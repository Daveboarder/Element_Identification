import subprocess
import sys

subprocess.run([sys.executable, "-m", "mlflow", "server", "--backend-store-uri", "./mlruns", "--port", "5000"])
