"""Smoke tests for xssharden package scaffolding (Task 1)."""

import subprocess
import sys


def test_package_version():
    """xssharden exposes __version__."""
    import xssharden

    assert hasattr(xssharden, "__version__")
    assert isinstance(xssharden.__version__, str)
    assert len(xssharden.__version__) > 0


def test_cli_help_exits_successfully():
    """python -m xssharden --help exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, "-m", "xssharden", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI exited with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()
