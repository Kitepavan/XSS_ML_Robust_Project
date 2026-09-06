"""Deterministic robustness report generation.

Renders independent four-arm hardening evaluations to Markdown or HTML
without inventing results. Payloads stay inert strings throughout and
raw payload text is never included in reports.
"""

from xssharden.reporting.report import (
    ARM_NAMES,
    SUPPORTED_SUFFIXES,
    render_html,
    render_markdown,
    write_report,
)

__all__ = [
    "ARM_NAMES",
    "SUPPORTED_SUFFIXES",
    "render_html",
    "render_markdown",
    "write_report",
]
