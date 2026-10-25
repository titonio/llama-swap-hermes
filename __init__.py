"""llama-swap provider profile plugin for Hermes Agent.

Hermes discovers this directory (``plugin.yaml`` + ``__init__.py``) under
``$HERMES_HOME/plugins/model-providers/`` and imports it; the import itself performs
the registration, which is Hermes' discovery contract. No logic here — the profile and
its hooks live in ``profile.py``.
"""

import os

from providers import register_provider

try:  # Under Hermes the plugin dir IS a package, so the relative import is the only
    # correct one. The absolute fallback serves flat test imports only, and the order
    # matters: `profile` is a real stdlib module in this interpreter
    # (lib/python3.14/profile.py), so had the absolute form won the race the failure
    # would be a confusing AttributeError on a profiler, not this clean ImportError.
    from .profile import build_profile
except ImportError:  # standalone import (tests, scripts): repo root on sys.path
    from profile import build_profile

llama_swap = build_profile(os.environ)

register_provider(llama_swap)

__all__ = ["llama_swap"]
