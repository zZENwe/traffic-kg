"""Load secrets from .env file."""
import os

def _load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

NEO4J_URI = os.environ['NEO4J_URI']
NEO4J_AUTH = (os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
DEEPSEEK_KEY = os.environ['DEEPSEEK_KEY']
