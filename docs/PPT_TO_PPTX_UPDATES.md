# PPT to PPTX Terminology Updates

This document summarizes the changes made to standardize terminology from "PPT" to "PPTX" throughout the application.

## Files Updated

### **Translation Files**
- **`translations/en.json`**: Updated app title from "File to PPT Converter" to "File to PPTX Converter"
- **`translations/hu.json`**: Updated app title from "Fájl PPT Konvertáló" to "Fájl PPTX Konvertáló"
- **`translations/zh.json`**: Updated app title from "MinerU 转 PPT 转换器" to "MinerU 转 PPTX 转换器"

### **GUI Files**
- **`streamlit_gui.py`**: Updated both English and Chinese app titles to use "PPTX"
  - English: "🚀 MinerU2PPT Converter" → "🚀 MinerU2PPTX Converter"
  - Chinese: "🚀 MinerU2PPT 转换器" → "🚀 MinerU2PPTX 转换器"

### **Command Line Interface**
- **`main.py`**: 
  - Updated argument parser description: "MinerU PDF/Image to PPT Converter" → "MinerU PDF/Image to PPTX Converter"
  - Updated output help text: "Path to output PPT file" → "Path to output PPTX file"

### **Documentation Files**
- **`README.md`**: 
  - Updated main title: "MinerU to PPT Converter" → "MinerU to PPTX Converter"
  - Updated CLI example: `--output <path_to_ppt>` → `--output <path_to_pptx>`
- **`README_zh.md`**: 
  - Updated main title: "MinerU 转 PPT 转换器" → "MinerU 转 PPTX 转换器"
  - Updated CLI example: `--output <ppt输出路径>` → `--output <pptx输出路径>`
- **`CLAUDE.md`**: Updated CLI example: `--output <path_to_ppt>` → `--output <path_to_pptx>`

## Rationale

### **Why PPTX over PPT?**
- **Modern Standard**: PPTX is the current PowerPoint format (Office 2007+)
- **Accurate Representation**: The tool actually generates `.pptx` files, not `.ppt` files
- **User Clarity**: Users expect PPTX format in modern applications
- **Technical Accuracy**: The code uses `python-pptx` library which creates PPTX format

### **What Remains Unchanged**
- **Function names**: `convert_mineru_to_ppt()` - kept for API compatibility
- **Parameter names**: `output_ppt_path` - kept for code consistency
- **Project folder name**: `MinerU2PPT` - kept to avoid breaking existing installations
- **Internal references**: Most internal code references remain as-is

## Impact

### **User-Facing Changes**
- ✅ **Window titles** now show "PPTX" for accuracy
- ✅ **Documentation** reflects modern terminology
- ✅ **CLI help text** uses correct file format
- ✅ **All languages** consistently updated (EN/HU/ZH)

### **Technical Compatibility**
- ✅ **No breaking changes** to existing APIs
- ✅ **Translation system** remains fully functional
- ✅ **GUI compilation** successful
- ✅ **All existing functionality** preserved

## Verification

All changes have been tested and verified:
- Translation system loads correctly in all languages
- GUI compiles without errors
- App titles display correctly
- No functional changes to core conversion logic