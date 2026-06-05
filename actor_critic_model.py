import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import threading
import os
import subprocess


class ACNet(nn.Module):
    def __init__(self, input_dim=21, output_dim=10):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


class ACAgent:
    def __init__(
        self,
        state_dim=21,
        action_dim=10,
        gamma=0.99,
        lr=3e-4,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.lock = threading.Lock()
        self.train_lock = threading.Lock()

        self.states  = []
        self.actions = []
        self.rewards = []

        self.total_simulations = 0
        self.reward_summation  = 0
        self.checkpoint_interval = 100
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.ac_net    = ACNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, log_prob, value, source_id):
        # log_prob, value, next_state kept for API compatibility with flask_server; not stored
        with self.lock:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
        self.reward_summation += reward

    def _compute_returns(self, rewards):
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def train(self, source_id):
        with self.train_lock:
            with self.lock:
                if not self.states:
                    return
                states  = list(self.states);  self.states.clear()
                actions = list(self.actions); self.actions.clear()
                rewards = list(self.rewards); self.rewards.clear()

            returns = self._compute_returns(rewards)

            s_t   = torch.FloatTensor(states).to(self.device)
            a_t   = torch.LongTensor(actions).to(self.device)
            ret_t = torch.FloatTensor(returns).to(self.device)

            logits, values_pred = self.ac_net(s_t)
            dist        = Categorical(logits=logits)
            log_probs   = dist.log_prob(a_t)
            entropy     = dist.entropy().mean()

            advantage = ret_t - values_pred.squeeze()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            actor_loss  = -(log_probs * advantage.detach()).mean()
            critic_loss = nn.MSELoss()(values_pred.squeeze(), ret_t)
            loss        = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.total_simulations += 1
            print("SIMULATION NUMBER:", self.total_simulations)

            if self.total_simulations >= 2000 and self.total_simulations % self.checkpoint_interval == 0:
                self.save_checkpoint()

            with open("loss_log.txt", "a") as f:
                f.write(f"{loss.item():.4f},{self.reward_summation:.2f}\n")
            self.reward_summation = 0

    def act(self, state):
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.ac_net(s_t)
            dist     = Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.total_simulations}.pth")
        torch.save(self.ac_net.state_dict(), path)
        print(f"[Checkpoint] Saved to {path}")
        try:
            subprocess.Popen(["python", "./validation/validate_ac.py", path])
        except Exception as e:
            print(f"[Validation Error] {e}")

    def load_checkpoint(self, path):
        self.ac_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded checkpoint from {path}")
