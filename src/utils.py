from pathlib import Path


def find_project_root(markers=("pyproject.toml", ".git")) -> Path:
    for parent in Path(__file__).parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Project root not found")
