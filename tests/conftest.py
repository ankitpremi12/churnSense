import sys
from pathlib import Path

# Make project root and src/ importable in all tests
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
