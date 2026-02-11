# 📋 Changelog

All notable changes to MinerU2PPTX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Additional AI providers integration
- More language support (German, French, Spanish)
- Auto-updater functionality
- Cloud processing options

---

## [2.0.1] - 2026-02-11

### Added
- **PDF Image Extractor**: New utility to extract images from original PDF files using MinerU JSON bounding boxes — works offline without downloading from CDN
- **CLI Sub-commands**: Restructured CLI with `convert` and `extract-images` sub-commands
  - `cli.exe convert --json ... --input ... --output ...` — PPTX conversion
  - `cli.exe extract-images --json ... --input ... --dpi 200` — image extraction
  - `cli.exe --version` — version display
- **Standalone CLI Executable**: New `cli.exe` console-mode executable for automation and scripting
- **Image Extractor Dialog**: GUI dialog to extract images from PDFs with progress tracking and cancellation support
- **Work Folder Selector**: Switchable output folder for extracted images in the GUI
- **CLI Version Flag**: `--version` flag displays `MinerU2PPTX 2.0.1`

### Changed
- **Author Updated**: Arlinamid (Rózsavölgyi János)
- **Version Bumped**: 2.0.1 across all version files, translations, and executables
- **Build System Overhauled**: Both `cli.spec` and `MinerU2PPTX.spec` now exclude ~20 unused heavy packages (scipy, pandas, pyarrow, sqlalchemy, etc.)
- **Executable Size Reduced**: ~47-48% smaller (170 MB → 87-90 MB) by stripping unused transitive dependencies
- **Build Scripts Updated**: `build.bat` and `build_exe.py` now build both `cli.exe` and `MinerU2PPTX.exe`

### Fixed
- **Spec File Paths**: Corrected `docs/CREDITS_AND_DIFFERENCES.md` and `docs/ABOUT_SUPPORT_FEATURE.md` paths in `MinerU2PPTX.spec` (were pointing to root instead of `docs/` folder)

---

## [2.0.0] - 2026-02-10

### 🎉 Major Release - Complete Rewrite with AI Enhancement

This version represents a complete transformation from a basic conversion tool to a professional-grade application with enterprise features.

### ✨ Added

#### 🤖 AI Integration
- **4 AI Providers**: OpenAI GPT, Google Gemini, Anthropic Claude, Groq support
- **Smart Text Correction**: 60-90% accuracy improvement for OCR documents
- **Intelligent Batching**: Automatic text chunking to prevent token limit issues
- **Persistent Caching**: Reduces API costs through intelligent text caching system
- **Fallback Systems**: Graceful degradation when AI services unavailable
- **Configuration Management**: Secure API key storage with encryption
- **Model Discovery**: Dynamic model fetching and selection for each provider

#### 🌍 Multi-Language Support
- **3 Languages**: Complete interface in English, Hungarian (Magyar), Chinese (中文)
- **Dynamic Language Switching**: Real-time language changes without restart
- **Cultural Adaptation**: Language-specific fonts and formatting optimization
- **Translation System**: JSON-based translation management with automatic detection
- **Complete Localization**: All UI elements, dialogs, and messages translated

#### 💻 Professional GUI Enhancements
- **About Dialog**: Developer information with GitHub profile integration
- **Dynamic Photo Loading**: GitHub avatar fetching with fallback systems
- **Support Integration**: Buy Me a Coffee links throughout the application
- **Post-Conversion Support**: Automatic support dialog after successful conversions
- **Language Selection**: Real-time UI language switching dropdown
- **Enhanced UX**: Better error messages, tooltips, and user guidance

#### 🎯 Advanced Conversion Features
- **Page Selection**: Convert single pages or ranges (e.g., 1-5, 10-15 format)
- **Smart Text Rendering**: Collision detection and enhanced character alignment
- **Intelligent Bullet Points**: Context-aware bullet point detection and formatting
- **Enhanced Batch Processing**: Multiple files with individual task configuration
- **Watermark Removal**: Improved detection and removal of headers/footers
- **Debug Mode**: Generate diagnostic images for troubleshooting

#### 🏗️ Development & Build System
- **Automated Build System**: PyInstaller integration with cross-platform scripts
- **Environment Management**: Automated setup scripts for development
- **Package Management**: Intelligent dependency checking and upgrades
- **Version Management**: Comprehensive version info for Windows executables
- **Cross-Platform Support**: Build scripts for Windows, Linux, and macOS

#### 📚 Comprehensive Documentation
- **Interactive User Guide**: Modern responsive HTML guide with 3-language support
- **AI Setup Guides**: Detailed instructions for all 4 AI providers with API key links
- **Build Instructions**: Complete development setup and deployment documentation
- **Feature Documentation**: Detailed guides for all new capabilities
- **Translation Documentation**: Guidelines for adding new languages
- **Troubleshooting Guide**: Solutions for common issues and optimization tips

#### 🌐 Alternative Interfaces
- **Streamlit Web GUI**: Complete browser-based interface alternative
- **Web API Ready**: Architecture supports future REST API implementation
- **Responsive Design**: All interfaces work on desktop, tablet, and mobile

### 🔧 Enhanced

#### Core Conversion Engine
- **Improved Text Processing**: Better handling of complex layouts and formatting
- **Enhanced Image Handling**: Optimized image extraction and positioning
- **Font Management**: Better Unicode support and font selection
- **Performance Optimization**: Faster processing through intelligent caching
- **Memory Management**: Reduced memory usage for large documents
- **Error Recovery**: More robust error handling and recovery mechanisms

#### User Experience
- **Drag & Drop**: Enhanced file drag-and-drop functionality
- **Progress Tracking**: Better progress indicators for long-running operations
- **Status Updates**: Real-time status updates during conversion process
- **File Validation**: Comprehensive input file validation and error reporting
- **Output Management**: Improved output file naming and organization

### 🏛️ Project & Attribution
- **Proper Attribution**: Complete attribution to JuniverseCoder's original MinerU2PPT
- **Code Headers**: Professional code documentation with inspiration credits
- **Project Documentation**: Comprehensive README with feature comparisons
- **Credits System**: Detailed comparison of original vs enhanced features
- **Open Source Compliance**: Proper licensing and attribution throughout

### 🔒 Security & Privacy
- **Local Processing**: All conversion happens locally, no data sent to third parties
- **Secure API Storage**: Encrypted storage of AI API keys
- **Privacy First**: Optional AI features, works completely offline if preferred
- **No Telemetry**: No usage tracking or data collection

### 🎨 Technical Architecture
- **Modular Design**: Clean separation of concerns with pluggable architecture
- **Service Abstraction**: Abstract base classes for AI services and components
- **Configuration System**: Flexible JSON-based configuration management
- **Event-Driven GUI**: Responsive interface with proper event handling
- **Threading**: Non-blocking operations with proper thread management
- **Resource Management**: Efficient handling of system resources

---

## [1.x.x] - Historical Versions

### Note
This enhanced version builds upon the foundation of [JuniverseCoder's original MinerU2PPT](https://github.com/JuniverseCoder/MinerU2PPT). 

**Original Features (1.x.x baseline):**
- Basic PDF/Image to PPTX conversion using MinerU JSON
- Simple GUI with drag-and-drop functionality  
- Basic batch processing mode
- Fundamental watermark removal
- Core PyInstaller executable creation

**Enhancement Factor:** This v2.0.0 represents approximately **10x feature expansion** with:
- 4 AI providers vs 0
- 3 languages vs 1
- Professional GUI vs basic interface
- Advanced documentation vs minimal README
- Comprehensive build system vs basic PyInstaller
- Enterprise features vs hobby tool functionality

---

## 🙏 Acknowledgments

### Original Inspiration
- **[JuniverseCoder](https://github.com/JuniverseCoder)** - Creator of the original MinerU2PPT concept
- **[MinerU Team](https://mineru.net/)** - Excellent PDF extraction service
- **Open Source Community** - Libraries and tools that made this enhancement possible

### Technology Stack
- **Python 3.11** - Core programming language
- **Tkinter** - Desktop GUI framework
- **PyInstaller** - Executable packaging
- **OpenAI, Google, Anthropic, Groq** - AI service providers
- **Streamlit** - Web interface framework
- **Various Python Libraries** - Supporting the rich functionality

---

## 📈 Version Schema

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

**Current Status:** v2.0.1 (Feature update with CLI and image extraction)

---

**Developer:** [Arlinamid (Rózsavölgyi János)](https://github.com/arlinamid)  
**Repository:** https://github.com/arlinamid/MinerU2PPT  
**Support:** [Buy Me a Coffee](https://buymeacoffee.com/arlinamid)  
**Original Inspiration:** [JuniverseCoder/MinerU2PPT](https://github.com/JuniverseCoder/MinerU2PPT)