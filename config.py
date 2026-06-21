import os
from pathlib import Path

# Project root is the directory containing this config file
PROJECT_ROOT = Path(__file__).resolve().parent

# The data folder is a sibling to the project folder
DATA_ROOT = PROJECT_ROOT.parent / "EE200_course_project_data_2026"

# Specific data paths
Q1_DATA_PATH = DATA_ROOT / "Q1_data"
Q2_DATA_PATH = DATA_ROOT / "Q2_data"
Q3_DEMO_PATH = DATA_ROOT / "Q3_demo"

# Ensure directories exist where needed (creating output directories)
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# The path to the downloaded song database for Q3
SONG_DATABASE_PATH = ASSETS_DIR / "song_database"

def get_path(relative_path: str) -> Path:
    """Helper to get an absolute path relative to the project root."""
    return PROJECT_ROOT / relative_path

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT} (Exists: {DATA_ROOT.exists()})")
    print(f"Q1 Data: {Q1_DATA_PATH} (Exists: {Q1_DATA_PATH.exists()})")
    print(f"Q2 Data: {Q2_DATA_PATH} (Exists: {Q2_DATA_PATH.exists()})")
