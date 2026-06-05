from flask_server import create_flask_app
from actor_critic_model import ACAgent
import threading
import subprocess
import time

NUM_INSTANCES = 1
BASE_PORT = 6000
batch_size = 50
lb_type = 4
epochs = 8000

jar_path = "AC-10VM-TRAIN.jar"

agent = ACAgent()

fallback_app = create_flask_app(agent, BASE_PORT)

fallback_thread = threading.Thread(target=fallback_app.run, kwargs={"port": BASE_PORT})
fallback_thread.daemon = True
fallback_thread.start()
print(f"🛡️ Fallback server running on port {BASE_PORT}")
time.sleep(2)

port = str(BASE_PORT)
proc = subprocess.Popen(["java", "-jar", jar_path, port, str(batch_size), 
                         str(lb_type), str(epochs)])
time.sleep(1)


proc.wait()
