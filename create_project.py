from pathlib import Path

PROJECT_NAME = "SR_FLOW"

root = Path.cwd()

project = root / PROJECT_NAME

folders = [
    project,
    project / "data",
    project / "reports",
    project / "output",
    project / "logs",
    project / "strategies"
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

files = {
    "run.py": "",
    "config.py": "",
    "strategy.py": "",
    "optimizer.py": "",
    "report.py": "",
    "README.txt": "Standalone Backtesting Project\n"
}

for name, content in files.items():
    file = project / name
    if not file.exists():
        file.write_text(content, encoding="utf-8")

print("=" * 50)
print("PROJECT CREATED")
print(project)
print("=" * 50)