"""Translation system for MinerU2PPT GUI.

This package provides internationalization support through JSON-based translation files.
"""

from .translator import Translator, get_translator, get_language

__all__ = ['Translator', 'get_translator', 'get_language']