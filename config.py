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

def _get(key):
    val = os.environ.get(key, '')
    if not val:
        raise RuntimeError(f"Missing {key}. Copy .env.example to .env and fill in your credentials.")
    return val

NEO4J_URI = _get('NEO4J_URI')
NEO4J_AUTH = (_get('NEO4J_USER'), _get('NEO4J_PASSWORD'))
DEEPSEEK_KEY = _get('DEEPSEEK_KEY')
