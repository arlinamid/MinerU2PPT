# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Commands

### Environment Setup
- **Python Version**: Python 3.10+ recommended
- **Virtual Environment**: Always use `python -m venv venv` and activate before installing
- **Install Dependencies**: `pip install -r requirements.txt`
- **Upgrade Packages**: `pip install --upgrade pip setuptools wheel`
- **AI Packages**: Uses latest `google-genai` (replaces deprecated `google-generativeai`)

### Execution
- **Run Tkinter GUI**: `python gui.py` (Traditional desktop GUI)
- **Run Streamlit GUI**: `python run_streamlit.py` or `streamlit run streamlit_gui.py` (Modern web-based GUI)
- **Run CLI**: `python main.py --json <path_to_json> --input <path_to_input> --output <path_to_pptx> [OPTIONS]`
 - `--no-watermark`: Erase elements marked as `discarded_blocks`.
 - `--debug-images`: Generate diagnostic images in the `tmp/` folder.
 - `--enable-ai`: Enable AI text correction using configured providers.
 - `--disable-ai`: Disable AI text correction for this conversion.
 - `--ai-provider <provider>`: Specify AI provider (OpenAI, Google Gemini, Anthropic Claude, Groq).
 - `--ai-model <model>`: Specify AI model to use for text correction.
 - `--api-key <key>`: API key for the specified AI provider.
 - `--correction-type <type>`: Type of correction (grammar_spelling, formatting, both).

### Packaging
- **Create Executable**: `pyinstaller --windowed --onefile --name MinerU2PPT gui.py`

## AI Text Correction Features

### Overview
MinerU2PPT now includes advanced AI-powered text correction capabilities that can automatically improve the quality of extracted text during the PDF/image to PowerPoint conversion process.

### Supported AI Providers
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo, GPT-4o
- **Google Gemini**: gemini-pro, gemini-pro-vision
- **Anthropic Claude**: claude-3-sonnet, claude-3-opus, claude-3-haiku
- **Groq**: llama2-70b-4096, mixtral-8x7b-32768, gemma-7b-it

### Authentication Methods
- **OpenAI**: API key authentication
- **Google Gemini**: CLI-based authentication with API key
- **Anthropic Claude**: API key authentication  
- **Groq**: API key authentication

### Correction Types
- **Grammar & Spelling**: Fixes grammatical errors and spelling mistakes
- **Formatting**: Improves text structure and consistency
- **Both**: Combines grammar/spelling and formatting corrections

### Configuration
- **GUI**: Access AI settings via the "AI Settings" button in the main interface
- **CLI**: Use command-line arguments for one-time configuration
- **Config File**: Settings are persistently stored in `ai_config.json`

### Key Features
- **Multi-Provider Support**: Choose from multiple AI providers based on your needs
- **JSON Response Format**: Enforced structured responses with MIME type application/json
- **Batch Text Processing**: Efficiently processes multiple texts by pages/sections
- **Smart Batch Sizing**: Automatically determines optimal batch size based on text amount
- **Async Processing**: Non-blocking text correction for better performance
- **Error Handling**: Graceful fallback to original text if correction fails
- **Confidence Scoring**: AI providers return confidence scores for corrections
- **Change Tracking**: Optional logging of all text corrections applied
- **Original Text Backup**: Automatically backup original text before correction
- **Cache System**: Avoids redundant corrections with intelligent caching

## Code Architecture

### High-Level Structure
- **`gui.py`**: The main entry point for end-users. A `tkinter`-based GUI that provides a user-friendly interface for the conversion process.
- **`main.py`**: The entry point for the command-line interface (CLI).
- **`converter/`**: Core conversion logic.
 - **`generator.py`**: Contains `PPTGenerator` and `PageContext`. It orchestrates the entire conversion from a source (PDF/image) and a MinerU JSON file to a PPTX presentation.
 - **`utils.py`**: Low-level helpers for image processing, color analysis, and segmentation.
 - **`ai_services.py`**: AI service abstraction layer supporting multiple providers (OpenAI, Gemini, Claude, Groq).
 - **`config.py`**: Configuration management for AI services, API keys, and correction settings.

### Known Issues and Solutions
- **Text Quality Issues**: See `TEXT_CREATION_ISSUES_ANALYSIS.md` and `TEXT_ISSUES_SUMMARY_AND_FIXES.md` for comprehensive analysis of text rendering problems and their solutions.

### Key Implementation Details
- **Unified Input**: The core logic in `convert_mineru_to_ppt` handles both PDF and single-image files as input, using the same sophisticated MinerU JSON-driven pipeline for both.
- **Stateful Page Processing**: A `PageContext` class holds the state for each page, including the original image, a progressively cleaned background, and a list of all detected elements.
- **Two-Phase Conversion**: Page processing is split into two phases:
  1.  **Analysis**: Elements from the JSON are processed to extract their data and populate the `PageContext`. A clean background is created by inpainting the area under each element.
  2.  **Rendering**: The cleaned background is rendered, followed by images, and finally text. This ensures correct Z-order layering.
- **Watermark/Footer Handling**: Elements marked as `discarded_blocks` in the JSON are handled based on the `remove_watermark` option.
- **Advanced Text Processing**:
 - **Bullet Point Correction**: A heuristic prepends a bullet character (`•`) if the first two detected `raw_chars` in a text block have different colors.
 - **Single-Line Textbox Widening**: Single-line textboxes are widened by 20% during rendering to prevent unwanted wrapping.
 - **AI Correction Caching**: Intelligent caching system stores AI corrections to avoid redundant API calls and improve performance.
- **AI Text Correction Integration**:
  - **Batch Processing Pipeline**: Three-phase processing: text collection → batch AI correction → element rendering
  - **Smart Batching**: Automatically groups texts by optimal batch size based on total character count
  - **JSON Response Handling**: All AI providers enforce structured JSON responses with robust parsing and fallbacks
  - **Page-Level Optimization**: Processes all texts on a page together for better context and efficiency
  - **Cache-First Architecture**: Corrected texts are cached and reused to avoid redundant API calls
  - **Provider Abstraction**: The `AIServiceManager` manages multiple AI services through a unified interface
  - **Error Recovery**: AI correction failures gracefully fall back to original text without interrupting conversion
  - **Context-Aware Correction**: AI services receive contextual information about text elements and page structure
  - **Span-Level Correction**: Complex text elements with multiple spans are intelligently updated while preserving structure
- **GUI Logic**:
  - **Single and Batch Modes**: The GUI supports two modes of operation. Users can switch between converting a single file and managing a list of files for batch processing.
  - **Modal Task Dialog**: In batch mode, a modal `AddTaskDialog` is used to add new conversion tasks. This dialog includes its own file browsers and drag-and-drop functionality.
  - **Dynamic UI**: The main window's UI changes dynamically based on the selected mode. Options that are not relevant for batch mode (like debugging) are hidden to simplify the interface.
  - **Per-Task Options**: In batch mode, options like "Remove Watermark" are configured individually for each task within the `AddTaskDialog`.
  - **Internationalization (i18n)**: Auto-detects OS language for English or Chinese UI.
  - **Drag and Drop**: `tkinterdnd2` is used for file inputs in both the main window and the task dialog.
  - **Asynchronous Processing**: The conversion process runs in a separate thread to keep the GUI responsive, for both single and batch conversions.

### Streamlit GUI Features
- **Modern Web Interface**: Browser-based GUI accessible from any device with a web browser
- **Dual Interface Options**: 
  - **Tkinter GUI** (`gui.py`): Traditional desktop application with native OS integration
  - **Streamlit GUI** (`streamlit_gui.py`): Web-based interface with real-time updates and modern UI
- **Feature Parity**: Both GUIs provide the same core functionality:
  - Single and batch conversion modes
  - Complete AI text correction configuration
  - Real-time progress tracking and logging
  - Cache management with visual statistics
  - Internationalization (English/Chinese)
- **Streamlit-Specific Features**:
  - **File Upload/Download**: Direct browser-based file handling
  - **Real-time Updates**: Automatic UI refresh on state changes
  - **Visual Cache Management**: Interactive statistics and controls
  - **Responsive Design**: Adapts to different screen sizes
  - **Server Deployment**: Can run on remote servers for team access
  - **Session Persistence**: Maintains state during browser session
- **Launch Options**:
  - **Quick Start**: `python run_streamlit.py` (recommended)
  - **Windows**: Double-click `run_streamlit.bat`
  - **Direct**: `streamlit run streamlit_gui.py`
  - **Server**: `streamlit run streamlit_gui.py --server.address 0.0.0.0`
