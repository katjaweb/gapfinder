import sys
from pathlib import Path

# Add 04-testing/ to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))