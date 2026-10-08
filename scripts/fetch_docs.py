"""Fetch a pinned snapshot of the Kubernetes docs (concepts + tasks) into data/raw.

Sparse, blobless clone of github.com/kubernetes/website, restricted to
content/en/docs/concepts and content/en/docs/tasks, checked out at a fixed commit.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/kubernetes/website.git"
COMMIT = "ba4362b3dcf7089084e809fc1c8fd4c84831fc8d"  # main as of 2026-10-08
SPARSE_PATHS = ["content/en/docs/concepts", "content/en/docs/tasks"]
LICENSE_FILES = ["LICENSE", "LICENSE-code"]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLONE_DIR = RAW_DIR / "k8s-website"


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def clone() -> None:
    if CLONE_DIR.exists():
        print(f"Reusing existing clone at {CLONE_DIR}")
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {REPO_URL} (sparse, blobless)...")
        git("clone", "--filter=blob:none", "--no-checkout", "--sparse", REPO_URL, str(CLONE_DIR))
    git("sparse-checkout", "set", *SPARSE_PATHS, cwd=CLONE_DIR)
    try:
        git("checkout", "--quiet", COMMIT, cwd=CLONE_DIR)
    except subprocess.CalledProcessError:
        git("fetch", "origin", COMMIT, cwd=CLONE_DIR)
        git("checkout", "--quiet", COMMIT, cwd=CLONE_DIR)


def write_metadata() -> None:
    head = git("rev-parse", "HEAD", cwd=CLONE_DIR)
    if head != COMMIT:
        sys.exit(f"Checked-out commit {head} does not match pinned {COMMIT}")
    (RAW_DIR / "COMMIT.txt").write_text(f"{REPO_URL}\n{head}\n")

    parts = [f"License files from {REPO_URL} at commit {head}\n"]
    for name in LICENSE_FILES:
        path = CLONE_DIR / name
        if path.exists():
            parts.append(f"\n===== {name} =====\n\n{path.read_text()}")
    (RAW_DIR / "LICENSE_NOTES.txt").write_text("".join(parts))


def report() -> None:
    files = [
        p
        for sub in SPARSE_PATHS
        for p in (CLONE_DIR / sub).rglob("*")
        if p.is_file()
    ]
    md = [p for p in files if p.suffix == ".md"]
    total = sum(p.stat().st_size for p in files)
    md_total = sum(p.stat().st_size for p in md)
    print(f"Commit: {COMMIT}")
    print(f"Files: {len(files)} total ({total / 1e6:.2f} MB)")
    print(f"Markdown: {len(md)} files ({md_total / 1e6:.2f} MB)")


def main() -> None:
    if shutil.which("git") is None:
        sys.exit("git is required")
    clone()
    write_metadata()
    report()


if __name__ == "__main__":
    main()
