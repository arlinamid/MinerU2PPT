# Translation System Documentation

The MinerU2PPT application now uses an external JSON-based translation system instead of hardcoded translations in the GUI code.

## Structure

```
translations/
├── __init__.py          # Package initialization
├── translator.py        # Translator class implementation
├── en.json             # English translations (76 keys)
├── hu.json             # Hungarian translations (76 keys)
└── zh.json             # Chinese translations (76 keys)
```

## Key Classes

### `Translator`
- **Location**: `translations/translator.py`
- **Purpose**: Manages translation loading, language detection, and text retrieval
- **Features**:
  - Automatic system language detection
  - JSON-based translation files
  - Fallback to English if translation missing
  - Dictionary-style access (`translator['key']`)
  - Method-based access (`translator.get_text('key')`)

### Usage in GUI

```python
from translations import get_translator

# Get the global translator instance
translator = get_translator()

# Access translations
app_title = translator['app_title']
# or
app_title = translator.get_text('app_title')

# Change language programmatically
translator.set_language('hu')  # Switch to Hungarian
```

### **Language Selection Interface**
The GUI includes a dropdown in the top-left corner allowing users to:
- **Select from available languages**: English, Magyar, 中文
- **Switch languages instantly** without restart
- **Override system language detection**

## Migration from Old System

The old system had hardcoded `TRANSLATIONS` dictionary in `gui.py`. The new system:

1. **Moved** all translations to external JSON files
2. **Created** a `Translator` class for management
3. **Updated** GUI code to use the new translator instance
4. **Maintained** backward compatibility with existing access patterns

## Adding New Translations

1. **Add to JSON files**: Update both `en.json` and `zh.json`
2. **Use in code**: Access via `translator['new_key']`

## Language Support

- **English** (`en`): Default fallback language (76 keys)
- **Hungarian** (`hu`): Full translation support (76 keys) 
- **Chinese** (`zh`): Full translation support (76 keys)
- **Extensible**: Add new languages by creating new JSON files

## Benefits

- **Maintainability**: Translations separated from code
- **Extensibility**: Easy to add new languages
- **Performance**: Translations loaded once at startup
- **Flexibility**: Can switch languages at runtime
- **Clean Code**: Removes large translation dictionaries from source files