"""CLI launcher: ``divar-streamlit`` → ``streamlit run streamlit_app.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit UI via ``streamlit run`` as a child process."""
    app_file = Path(__file__).resolve().parent / "streamlit_app.py"
    cmd = ["streamlit", "run", str(app_file), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))
