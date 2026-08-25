import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_candidate
from src.utils import ARTIFACTS_DIR

def bootstrap() -> None:
    """Bootstrap the environment by training the initial model if none exists."""
    print("Checking for existing artifacts...")
    model_path = ARTIFACTS_DIR / "model.pkl"
    
    if model_path.exists():
        print(f"Found existing model at {model_path}. Skipping training.")
        return

    print("No model found. Starting bootstrap training...")
    try:
        # We assume environment variables are set via Docker Compose or .env
        result = train_candidate(persist_local_artifacts=True)
        print(f"Bootstrap successful! Model version: {result['version']}, Run ID: {result['run_id']}")
    except Exception as e:
        print(f"Bootstrap failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    bootstrap()
