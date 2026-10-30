"""Pytest configuration: re-export the shared fixtures from ``_fixtures``."""

from __future__ import annotations

import os

# Set dummy credentials for offline testing so tests never fail on missing cloud keys in CI
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-offline-key")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-offline-key")
os.environ.setdefault("GEMINI_API_KEY", "sk-test-offline-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-offline-key")

from _fixtures import PLAIN, ROOT, SAMPLE, plain, plan, sample

__all__ = ["PLAIN", "ROOT", "SAMPLE", "plan", "plain", "sample"]
