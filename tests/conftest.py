"""Pytest configuration: re-export the shared fixtures from ``_fixtures``."""

from __future__ import annotations

from _fixtures import PLAIN, ROOT, SAMPLE, plain, plan, sample  # noqa: F401

__all__ = ["PLAIN", "ROOT", "SAMPLE", "plan", "plain", "sample"]
