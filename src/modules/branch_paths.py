from pathlib import Path
import os
import re
import subprocess
from typing import Optional


def project_root() -> Path:
    # src/modules/branch_paths.py -> project root is two levels up.
    return Path(__file__).resolve().parents[2]


def get_current_branch(repo_dir: Optional[Path] = None) -> str:
    env_branch = os.getenv("GITHUB_REF_NAME") or os.getenv("CI_COMMIT_REF_NAME")
    if env_branch:
        return env_branch

    root = project_root() if repo_dir is None else Path(repo_dir)
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if branch == "HEAD":
            return "detached"
        return branch
    except Exception:
        return "unknown"


def safe_branch_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def branch_dir(base_subdir: str, create: bool = True) -> Path:
    branch = safe_branch_name(get_current_branch())
    out = project_root() / base_subdir / branch
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def latest_in_branch(base_subdir: str, pattern: str) -> Optional[Path]:
    folder = branch_dir(base_subdir, create=False)
    if not folder.exists():
        return None

    matches = list(folder.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)
