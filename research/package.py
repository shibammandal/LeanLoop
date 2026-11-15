"""Build the submission bundle: dist/leanloop-submission.zip.

Includes the code, tests, write-up, demo assets and OUR saved result files. Leaves out everything that is large, third-party or
not ours to redistribute: data/ (other people's benchmark data is re-downloaded by the documented commands), vendor/, tools/
(pi and its node_modules), scratch/, caches and virtual environments.

usage: python research/package.py
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["LICENSE", "README.md", "SUBMISSION.md", "pyproject.toml", "pytest.ini", "compare.py", "run_pi.py", "run_leanloop.py"]
DIRS = ["src", "tests", "research", "demo", "results", "tasks", "pi-agent"]
SKIP_NAMES = ("safetune_install.log", "st_debug2.log")      # pip/debug logs that contain local machine paths
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", "node_modules", "dist", "leanloop.egg-info"}


# Result files record the temporary folders the harnesses ran in ("C:\Users\<name>\AppData\Local\Temp\..."). The originals stay as they
# are; the copies in the bundle have the user name inside a Users path blanked, in any of its escaped or slash forms.
USER_IN_PATH = re.compile(r"(?i)(Users[\/]+)Shibam")
TEXT_SUFFIXES = {".jsonl", ".json", ".log", ".md", ".txt", ".tsv"}


def bundle_bytes(p: Path, root: Path) -> bytes:
    data = p.read_bytes()
    if p.relative_to(root).parts[0] == "results" and p.suffix in TEXT_SUFFIXES:
        data = USER_IN_PATH.sub(r"\g<1>user", data.decode("utf-8", errors="replace")).encode("utf-8")
    return data


def main():
    out = ROOT / "dist" / "leanloop-submission.zip"
    out.parent.mkdir(exist_ok=True)
    paths = [ROOT / f for f in FILES if (ROOT / f).exists()]
    for d in DIRS:
        paths += [p for p in (ROOT / d).rglob("*") if p.is_file()]
    paths = sorted({p for p in paths if not (SKIP_PARTS & set(p.relative_to(ROOT).parts)) and p.suffix != ".pyc" and p.name not in SKIP_NAMES})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.writestr((Path("leanloop") / p.relative_to(ROOT)).as_posix(), bundle_bytes(p, ROOT))
    print(f"{len(paths)} files -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
