# 🚀 MinerU2PPT Streamlit GUI

A modern, web-based interface for converting PDFs and images to PowerPoint presentations with AI-powered text correction.

## Features

### 🎯 **Dual Mode Operation**
- **Single File Mode**: Convert individual files with full control
- **Batch Mode**: Process multiple files efficiently with task queue management

### 🤖 **AI Text Correction**
- **Multiple Providers**: OpenAI, Google Gemini, Anthropic Claude, Groq
- **Correction Types**: Grammar & Spelling, Formatting, or Both
- **Real-time Authentication**: Test API keys instantly
- **Dynamic Model Selection**: Fetch latest available models
- **Smart Caching**: Avoid duplicate corrections with intelligent caching

### 🌐 **Internationalization**
- **English** and **Chinese** language support
- **Auto-detection** based on system locale
- **Consistent UI** across languages

### 📊 **Advanced Features**
- **Cache Management**: View stats, clear by provider, or clear all
- **Real-time Logging**: Progress tracking and error reporting
- **File Upload**: Drag-and-drop support for all file types
- **Download Integration**: Direct PPTX download from browser
- **Debug Mode**: Generate diagnostic images for troubleshooting

## Installation & Setup

### 1. Install Dependencies
```bash
# Install Streamlit and other requirements
pip install -r requirements.txt

# Or install Streamlit separately
pip install streamlit>=1.32.0
```

### 2. Launch Options

#### Option A: Python Script (Recommended)
```bash
python run_streamlit.py
```

#### Option B: Windows Batch File
```bash
# Double-click or run from command line
run_streamlit.bat
```

#### Option C: Direct Streamlit Command
```bash
streamlit run streamlit_gui.py
```

### 3. Access the Interface
The web interface will automatically open at: **http://localhost:8501**

## Using the Streamlit GUI

### 📁 **Single File Mode**

1. **Select Conversion Mode**
   - Click **"Single File Conversion"** button at the top

2. **Upload Files**
   - **Input File**: Upload PDF or image file (.pdf, .png, .jpg, etc.)
   - **MinerU JSON**: Upload the JSON file from MinerU extractor
   - **Output Name**: Specify the PPTX filename

3. **Configure Options**
   - ✅ **Remove Watermark**: Strip footer/watermark elements
   - 🐛 **Debug Images**: Generate diagnostic images in tmp/ folder

4. **AI Text Correction** (Optional)
   - ✅ **Enable AI**: Turn on AI-powered text correction
   - **Provider**: Choose OpenAI, Gemini, Claude, or Groq
   - **API Key**: Enter your API key (stored securely)
   - **Model**: Select from available models
   - **Type**: Grammar & Spelling, Formatting, or Both
   - 🧪 **Test Auth**: Verify API key works
   - 💾 **Save Config**: Store settings for future use

5. **Start Conversion**
   - Click **🎯 Start Conversion**
   - Monitor progress in real-time
   - Download PPTX when complete

### 🔄 **Batch Mode**

1. **Select Batch Mode**
   - Click **"Batch Processing"** button at the top

2. **Add Tasks**
   - **Add Task** section allows multiple file uploads
   - Configure options per task (watermark, debug)
   - Click **➕ Add to Queue** for each task

3. **Manage Queue**
   - View all tasks in the **Task List**
   - Delete unwanted tasks with 🗑️ button
   - See status: ⏳ Pending, 🔄 Processing, ✅ Complete, ❌ Error

4. **Process Batch**
   - Click **🚀 Start Batch Processing**
   - Monitor overall progress
   - Each completed file becomes available for download

### 🧠 **AI Configuration**

#### Supported Providers

| Provider | Models | Features |
|----------|---------|----------|
| **OpenAI** | GPT-4, GPT-3.5-turbo | High accuracy, fast |
| **Google Gemini** | Gemini Pro, Flash | Advanced reasoning |
| **Anthropic Claude** | Claude-3.5 Sonnet | Excellent for text |
| **Groq** | Mixtral, LLaMA | Ultra-fast inference |

#### Setup Steps

1. **Get API Key**
   - Visit provider's website (links in help section)
   - Create account and generate API key
   - Copy the key (keep it secure!)

2. **Configure in GUI**
   - Select provider from dropdown
   - Paste API key in password field
   - Click **Test Authentication**
   - Click **Refresh Models** to load available models
   - Select preferred model
   - Choose correction type (grammar, formatting, both)
   - Click **Save Configuration**

3. **Use AI Correction**
   - Enable AI checkbox in conversion options
   - AI will automatically correct text during conversion
   - View corrections in the logs
   - Corrections are cached to save API costs

### 📊 **Cache Management**

The AI text correction system includes intelligent caching:

#### **Cache Statistics**
- **Total Cached Texts**: Number of corrected texts stored
- **Cache Size**: Memory usage in MB
- **AI Providers**: Number of different providers used
- **Hit Rate**: Percentage of cache hits vs. new corrections

#### **Cache Actions**
- **📊 View Recent Corrections**: See latest AI corrections
- **🧹 Clear Current Provider**: Remove cache for selected AI provider
- **🗑️ Clear All Cache**: Remove all cached corrections

#### **Benefits**
- **Cost Savings**: Avoid re-correcting identical texts
- **Speed**: Instant retrieval of previously corrected texts
- **Persistence**: Cache survives application restarts
- **Smart Management**: Automatic cleanup of old/large entries

### 📝 **Conversion Logs**

Real-time logging shows:
- **File Upload Status**: Confirms successful uploads
- **Conversion Progress**: Step-by-step processing updates  
- **AI Corrections**: Details of text changes made
- **Error Messages**: Clear error descriptions with solutions
- **Completion Status**: Success confirmation with file info

#### **Log Features**
- **Timestamps**: Each entry shows exact time
- **Color Coding**: Info (ℹ️), Success (✅), Warning (⚠️), Error (❌)
- **Auto-scrolling**: Latest entries appear at top
- **History**: Last 50 entries retained
- **Clear Function**: 🧹 Clear Logs button

### 🆘 **Help & Troubleshooting**

#### **Built-in Help Sections**
- **MinerU JSON**: Information about required JSON format
- **Supported Formats**: List of accepted file types
- **System Information**: Python version, directories, etc.

#### **Common Issues**

1. **"No files uploaded"**
   - Make sure both input file AND JSON file are uploaded
   - Check file types are supported

2. **AI Authentication Failed**
   - Verify API key is correct
   - Check internet connection
   - Ensure account has sufficient credits

3. **Conversion Failed**
   - Check JSON file is valid MinerU output
   - Verify input file isn't corrupted
   - Look at detailed logs for specific error

4. **Streamlit Won't Start**
   - Install requirements: `pip install -r requirements.txt`
   - Try: `python run_streamlit.py`
   - Check Python version is 3.8+

## Comparison: Streamlit vs. Tkinter GUI

| Feature | Streamlit GUI | Tkinter GUI |
|---------|---------------|-------------|
| **Interface** | Modern web-based | Desktop application |
| **Accessibility** | Any device with browser | Local machine only |
| **File Handling** | Upload/download | File system browser |
| **Real-time Updates** | Automatic refresh | Manual polling |
| **Deployment** | Can run on server | Desktop only |
| **Dependencies** | Streamlit + web browser | tkinterdnd2 |
| **AI Configuration** | Web forms | Dialog windows |
| **Batch Processing** | Interactive queue | List management |
| **Logs** | Real-time scrolling | Text widget |
| **Cache Management** | Visual stats/controls | Basic buttons |

## Advanced Usage

### 🖥️ **Server Deployment**

Run on a server for team access:
```bash
# Run on all interfaces
streamlit run streamlit_gui.py --server.address 0.0.0.0 --server.port 8501

# With authentication
streamlit run streamlit_gui.py --server.enableCORS false --server.enableXsrfProtection false
```

### 🎨 **Customization**

The GUI supports theme customization:
```bash
# Light theme (default)
streamlit run streamlit_gui.py --theme.base light

# Dark theme
streamlit run streamlit_gui.py --theme.base dark

# Custom colors
streamlit run streamlit_gui.py --theme.primaryColor "#FF6B6B" --theme.backgroundColor "#FFFFFF"
```

### 🔧 **Configuration Files**

Settings are stored in:
- **AI Configuration**: `ai_config.json`
- **Text Cache**: `ai_text_cache.json`
- **Streamlit Config**: `.streamlit/config.toml` (optional)

## Technical Details

### **Architecture**
- **Frontend**: Streamlit web components
- **Backend**: Same conversion engine as Tkinter GUI
- **State Management**: Streamlit session state
- **File Handling**: Temporary file system with cleanup
- **Async Support**: Asyncio for AI operations

### **Security**
- **API Keys**: Stored in session state (not persistent by default)
- **File Uploads**: Temporary storage with automatic cleanup
- **Network**: Local-only by default (localhost:8501)

### **Performance**
- **Streaming**: Real-time progress updates
- **Caching**: Intelligent AI correction caching
- **Memory**: Automatic cleanup of large logs
- **Concurrency**: Non-blocking AI operations

---

## 🚀 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Launch**: `python run_streamlit.py`
3. **Access**: Open http://localhost:8501
4. **Configure**: Set up AI providers (optional)
5. **Convert**: Upload files and click Start!

The Streamlit GUI provides the same powerful conversion capabilities as the desktop version, but with a modern web interface that's accessible from any device with a web browser.

---

## 🔧 Recent Fixes (February 2026)

### Service Registration Issues - FIXED ✅

All "Service not registered" errors have been completely resolved:

- ✅ **Fixed**: `Failed to fetch models: Service Google Gemini not registered`
- ✅ **Fixed**: `Error saving configuration: Service gemini not registered`  
- ✅ **Fixed**: `Authentication failed: Service gemini not registered`
- ✅ **Fixed**: AI services now auto-register when first accessed
- ✅ **Fixed**: Provider name mapping ensures seamless operation
- ✅ **Fixed**: Model fetching works correctly for all providers

**What This Means for Users:**
- No more manual service registration required
- AI provider switching works seamlessly
- Model fetching is now reliable
- Configuration saving works correctly
- Authentication flow is smooth

The Streamlit GUI now provides a fully robust experience for AI-powered text correction!