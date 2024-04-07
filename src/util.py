from pathlib import Path


def mkdir(dir_path: Path):
    if dir_path.is_dir():
        dir_path.mkdir()
