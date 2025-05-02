"""使用するPATHの一覧"""
from pathlib import Path

ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG = ROOT/"src"/"config"
DEFAULT_DATA_DIR = ROOT/"data"
DEFAULT_REPOS_DIR = DEFAULT_DATA_DIR/"repos"