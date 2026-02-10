import json
import os
import locale
from pathlib import Path


class Translator:
    """Translation management class that loads translations from JSON files."""
    
    def __init__(self, translations_dir=None):
        """Initialize the translator.
        
        Args:
            translations_dir: Path to directory containing translation JSON files.
                            If None, uses the 'translations' directory relative to this file.
        """
        if translations_dir is None:
            translations_dir = Path(__file__).parent
        
        self.translations_dir = Path(translations_dir)
        self.translations = {}
        self.current_language = None
        
        # Load all available translations
        self._load_translations()
        
        # Set default language
        self.set_language(self._detect_system_language())
    
    def _load_translations(self):
        """Load all translation JSON files from the translations directory."""
        if not self.translations_dir.exists():
            raise FileNotFoundError(f"Translations directory not found: {self.translations_dir}")
        
        for json_file in self.translations_dir.glob("*.json"):
            lang_code = json_file.stem  # filename without extension
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                print(f"Loaded {lang_code} translations from {json_file}")
            except Exception as e:
                print(f"Warning: Failed to load translations from {json_file}: {e}")
    
    def _detect_system_language(self):
        """Detect the system language and return appropriate language code."""
        try:
            # Use the recommended approach instead of deprecated getdefaultlocale()
            lang_code = locale.getlocale()[0] or locale.getencoding() or 'en'
            if not lang_code:
                # Fallback to environment variables
                lang_code = os.environ.get('LANG', os.environ.get('LANGUAGE', 'en'))
            return 'zh' if lang_code and lang_code.lower().startswith('zh') else 'en'
        except Exception: 
            return 'en'
    
    def set_language(self, lang_code):
        """Set the current language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'zh')
        """
        if lang_code in self.translations:
            self.current_language = lang_code
        else:
            print(f"Warning: Language '{lang_code}' not found, falling back to 'en'")
            self.current_language = 'en' if 'en' in self.translations else list(self.translations.keys())[0]
    
    def get_available_languages(self):
        """Get list of available language codes."""
        return list(self.translations.keys())
    
    def get_text(self, key, fallback=None):
        """Get translated text for the given key.
        
        Args:
            key: Translation key
            fallback: Fallback text if key not found (defaults to key itself)
        
        Returns:
            Translated text or fallback
        """
        if self.current_language and self.current_language in self.translations:
            text = self.translations[self.current_language].get(key)
            if text is not None:
                return text
        
        # Fallback to English if current language doesn't have the key
        if 'en' in self.translations and self.current_language != 'en':
            text = self.translations['en'].get(key)
            if text is not None:
                return text
        
        # Return fallback or key itself
        return fallback if fallback is not None else key
    
    def get(self, key, default=None):
        """Get translated text (backward compatibility method)."""
        return self.get_text(key, fallback=default)
    
    def __getitem__(self, key):
        """Allow dictionary-style access to translations."""
        return self.get_text(key)
    
    def get_current_language(self):
        """Get the current language code."""
        return self.current_language


# Global translator instance
_translator_instance = None

def get_translator():
    """Get the global translator instance."""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = Translator()
    return _translator_instance

def get_language():
    """Get current language code (for backward compatibility)."""
    return get_translator().get_current_language()