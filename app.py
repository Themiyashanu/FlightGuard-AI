"""Project helper entrypoint.

Usage:
    python app.py

This launches the Streamlit dashboard.
"""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=False)

