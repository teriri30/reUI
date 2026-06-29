"""Shared pytest environment setup."""
import os

from runtime_env import configure_geospatial_environment


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REUI_DISABLE_AUTO_START", "1")
configure_geospatial_environment()
