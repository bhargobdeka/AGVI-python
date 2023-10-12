import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "AGVI_python"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    "main.py",
    "requirements.txt",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath) # create a Path object
    [filedir, filename] = os.path.split(filepath) # split path into directory and filename
    if filedir !="": # if directory does not exist, create it
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): # if file does not exist or is empty, create it
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File {filepath} already exists")




