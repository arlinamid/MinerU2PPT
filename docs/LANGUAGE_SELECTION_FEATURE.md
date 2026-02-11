# Language Selection Feature

The GUI now includes a language selection dropdown that allows users to manually choose their preferred language.

## Feature Overview

### **Language Selection Dropdown**
- **Location**: Top-left corner of the main window, next to the mode switch button
- **Options**: English, Magyar (Hungarian), 中文 (Chinese)
- **Behavior**: Instantly updates all UI elements when selection changes

### **Dynamic UI Updates**
When a user selects a different language:
1. **Window title** changes immediately
2. **All labels, buttons, and text** update to the new language
3. **Content area rebuilds** with translated text
4. **Mode selection** (Single/Batch) maintains current state

### **Technical Implementation**

#### **Language Mapping**
```python
language_names = {
    'en': 'English',
    'hu': 'Magyar', 
    'zh': '中文'
}
```

#### **Key Methods**
- **`_get_language_options()`**: Returns display names for combobox
- **`_get_language_code()`**: Converts display name to language code
- **`_on_language_change()`**: Handles language selection events
- **`_update_ui_text()`**: Rebuilds UI with new translations

#### **UI Structure**
- **Top Controls Frame**: Persistent (language selector + mode button)
- **Content Frame**: Rebuilt when language changes
- **Translation Keys**: 77 keys per language file

### **Supported Languages**

| Code | Display Name | Translation File | Keys |
|------|-------------|------------------|------|
| `en` | English | `translations/en.json` | 77 |
| `hu` | Magyar | `translations/hu.json` | 77 |
| `zh` | 中文 | `translations/zh.json` | 77 |

### **User Experience**
- **Automatic Detection**: Default language based on system locale
- **Manual Override**: Users can select any supported language
- **Instant Feedback**: UI updates immediately without restart
- **State Preservation**: Current mode and settings maintained during switch

### **Benefits**
- ✅ **User Control**: Manual language selection regardless of system settings
- ✅ **Real-time Switching**: No application restart required
- ✅ **Consistent Experience**: All UI elements update synchronously
- ✅ **Extensible**: Easy to add new languages by creating JSON files