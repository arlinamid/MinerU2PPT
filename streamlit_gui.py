#!/usr/bin/env python3
"""
Streamlit GUI for MinerU2PPT - Modern web-based interface
"""

import streamlit as st
import os
import sys
import tempfile
import shutil
import asyncio
import threading
import time
import locale
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import core conversion logic
from converter.generator import convert_mineru_to_ppt
from converter.config import ai_config
from converter.ai_services import ai_manager
from converter.cache_manager import global_cache_manager

# Configure Streamlit page
st.set_page_config(
    page_title="MinerU2PPT Converter",
    page_icon="📄➡️📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/MinerU2PPT',
        'Report a bug': 'https://github.com/your-repo/MinerU2PPT/issues',
        'About': 'MinerU2PPT - Convert PDFs and images to PowerPoint presentations with AI text correction'
    }
)

# --- Internationalization ---
TRANSLATIONS = {
    "en": {
        "app_title": "🚀 MinerU2PPTX Converter",
        "subtitle": "Convert PDFs and Images to PowerPoint presentations with AI-powered text correction",
        "mode_selection": "Conversion Mode",
        "single_mode": "Single File Conversion",
        "batch_mode": "Batch Processing",
        "file_inputs": "File Inputs",
        "input_file_label": "📁 Input File (PDF/Image)",
        "json_file_label": "📋 MinerU JSON File",
        "output_file_label": "💾 Output PPTX File",
        "conversion_options": "Conversion Options", 
        "remove_watermark": "Remove Watermark/Footer elements",
        "debug_images": "Generate Debug Images",
        "page_selection": "Page Selection",
        "all_pages": "All Pages",
        "single_page": "Single Page",
        "page_range": "Page Range",
        "page_number": "Page Number",
        "from_page": "From Page",
        "to_page": "To Page",
        "ai_correction": "AI Text Correction",
        "enable_ai": "Enable AI Text Correction",
        "ai_provider": "AI Provider",
        "ai_model": "Model",
        "api_key": "API Key",
        "correction_type": "Correction Type",
        "test_auth": "Test Authentication",
        "save_config": "Save AI Configuration",
        "start_conversion": "🎯 Start Conversion",
        "conversion_progress": "Conversion Progress",
        "batch_tasks": "Batch Tasks",
        "add_task": "➕ Add Task",
        "task_list": "Task List",
        "start_batch": "🚀 Start Batch Processing",
        "cache_management": "Cache Management",
        "cache_stats": "Cache Statistics",
        "clear_cache": "Clear Cache",
        "logs": "Conversion Logs",
        "help_section": "Help & Information",
        "json_help": "About MinerU JSON Files",
        "json_help_text": "This tool requires a JSON file from the MinerU PDF/Image Extractor. The JSON contains structured data about text blocks, images, and layout information extracted from your PDF or image file.",
        "supported_formats": "Supported Input Formats",
        "formats_list": "• PDF files (.pdf)\n• Image files (.png, .jpg, .jpeg, .bmp, .tiff)",
        "correction_grammar": "Grammar & Spelling",
        "correction_formatting": "Formatting Only", 
        "correction_both": "Grammar & Formatting",
        "providers": {
            "openai": "OpenAI",
            "gemini": "Google Gemini",
            "claude": "Anthropic Claude",
            "groq": "Groq"
        },
        "auth_success": "✅ Authentication successful!",
        "auth_failed": "❌ Authentication failed: {}",
        "config_saved": "✅ Configuration saved successfully!",
        "conversion_complete": "✅ Conversion completed successfully!",
        "conversion_error": "❌ Conversion failed: {}",
        "no_files": "Please upload the required files",
        "processing": "Processing...",
        "task_added": "Task added to queue",
        "batch_complete": "✅ Batch processing completed!",
        "cache_cleared": "✅ Cache cleared successfully!",
    },
    "zh": {
        "app_title": "🚀 MinerU2PPTX 转换器",
        "subtitle": "将 PDF 和图片转换为 PowerPoint 演示文稿，支持 AI 文本校正",
        "mode_selection": "转换模式",
        "single_mode": "单文件转换",
        "batch_mode": "批量处理",
        "file_inputs": "文件输入",
        "input_file_label": "📁 输入文件 (PDF/图片)",
        "json_file_label": "📋 MinerU JSON 文件", 
        "output_file_label": "💾 输出 PPTX 文件",
        "conversion_options": "转换选项",
        "remove_watermark": "移除水印/页脚元素",
        "debug_images": "生成调试图片",
        "page_selection": "页面选择",
        "all_pages": "全部页面",
        "single_page": "单页",
        "page_range": "页面范围",
        "page_number": "页码",
        "from_page": "起始页",
        "to_page": "结束页",
        "ai_correction": "AI 文本校正",
        "enable_ai": "启用 AI 文本校正",
        "ai_provider": "AI 提供商",
        "ai_model": "模型",
        "api_key": "API 密钥",
        "correction_type": "校正类型",
        "test_auth": "测试认证",
        "save_config": "保存 AI 配置",
        "start_conversion": "🎯 开始转换",
        "conversion_progress": "转换进度",
        "batch_tasks": "批量任务",
        "add_task": "➕ 添加任务",
        "task_list": "任务列表",
        "start_batch": "🚀 开始批量处理",
        "cache_management": "缓存管理",
        "cache_stats": "缓存统计",
        "clear_cache": "清除缓存",
        "logs": "转换日志",
        "help_section": "帮助信息",
        "json_help": "关于 MinerU JSON 文件",
        "json_help_text": "此工具需要 MinerU PDF/图片提取器生成的 JSON 文件。JSON 包含从您的 PDF 或图片文件中提取的文本块、图像和布局信息的结构化数据。",
        "supported_formats": "支持的输入格式",
        "formats_list": "• PDF 文件 (.pdf)\n• 图片文件 (.png, .jpg, .jpeg, .bmp, .tiff)",
        "correction_grammar": "语法和拼写",
        "correction_formatting": "仅格式化",
        "correction_both": "语法和格式化",
        "providers": {
            "openai": "OpenAI",
            "gemini": "Google Gemini",
            "claude": "Anthropic Claude", 
            "groq": "Groq"
        },
        "auth_success": "✅ 认证成功！",
        "auth_failed": "❌ 认证失败：{}",
        "config_saved": "✅ 配置保存成功！",
        "conversion_complete": "✅ 转换完成！",
        "conversion_error": "❌ 转换失败：{}",
        "no_files": "请上传必需的文件",
        "processing": "处理中...",
        "task_added": "任务已添加到队列",
        "batch_complete": "✅ 批量处理完成！",
        "cache_cleared": "✅ 缓存清除成功！",
    }
}

def get_language() -> str:
    """Detect system language"""
    try:
        system_locale = locale.getlocale()[0]
        if system_locale and system_locale.lower().startswith('zh'):
            return 'zh'
    except:
        pass
    return 'en'

def get_full_provider_name(short_name: str) -> str:
    """Convert short provider name to full name used by AI services"""
    provider_mapping = {
        'openai': 'OpenAI',
        'gemini': 'Google Gemini', 
        'claude': 'Anthropic Claude',
        'groq': 'Groq'
    }
    return provider_mapping.get(short_name, short_name)

def get_short_provider_name(full_name: str) -> str:
    """Convert full provider name to short name used by GUI"""
    reverse_mapping = {
        'OpenAI': 'openai',
        'Google Gemini': 'gemini',
        'Anthropic Claude': 'claude', 
        'Groq': 'groq'
    }
    return reverse_mapping.get(full_name, full_name.lower())

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        'language': get_language(),
        'conversion_mode': 'single',
        'ai_enabled': False,
        'ai_provider': 'openai',
        'ai_model': '',
        'api_key': '',
        'correction_type': 'grammar_spelling',
        'remove_watermark': True,
        'debug_images': False,
        'page_selection_mode': 'all',
        'single_page': 1,
        'from_page': 1,
        'to_page': 1,
        'page_selection': {"mode": "all"},
        'batch_tasks': [],
        'conversion_logs': [],
        'conversion_in_progress': False,
        'ai_config_loaded': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_ai_config():
    """Load AI configuration into session state"""
    if not st.session_state.ai_config_loaded:
        try:
            st.session_state.ai_enabled = ai_config.is_text_correction_enabled()
            
            # Convert full provider name to short name for GUI
            full_provider = ai_config.get_active_provider() or 'OpenAI'
            st.session_state.ai_provider = get_short_provider_name(full_provider)
            
            st.session_state.correction_type = ai_config.get_correction_type() or 'grammar_spelling'
            
            # Load API key for current provider (using full name)
            provider_key = ai_config.get_api_key(full_provider)
            st.session_state.api_key = provider_key
            
            # Load configured model for current provider (using full name)
            configured_model = ai_config.get_model(full_provider)
            if configured_model:
                st.session_state.ai_model = configured_model
            else:
                st.session_state.ai_model = ''
                
            st.session_state.ai_config_loaded = True
        except Exception as e:
            st.error(f"Error loading AI configuration: {e}")

async def save_ai_configuration():
    """Save AI configuration from session state"""
    try:
        # Convert short provider name to full name
        full_provider_name = get_full_provider_name(st.session_state.ai_provider)
        
        # Update config settings
        ai_config.set_text_correction_enabled(st.session_state.ai_enabled)
        ai_config.set_active_provider(full_provider_name)
        ai_config.set_correction_type(st.session_state.correction_type)
        
        # Update API key for current provider (using full name)
        ai_config.set_api_key(full_provider_name, st.session_state.api_key)
        
        # Update model selection for current provider (using full name)
        if st.session_state.ai_model:
            ai_config.set_model(full_provider_name, st.session_state.ai_model)
        
        # Save to file
        ai_config.save_config()
        
        # Update the AI manager (using full name)
        await ai_manager.set_active_service(full_provider_name)
        
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False

def log_message(message: str, level: str = "info"):
    """Add a message to the conversion logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Initialize logs if not present
    if 'conversion_logs' not in st.session_state:
        st.session_state.conversion_logs = []
    
    # Add log entry
    st.session_state.conversion_logs.append({
        'timestamp': timestamp,
        'message': message,
        'level': level
    })
    
    # Limit log entries to prevent memory issues
    if len(st.session_state.conversion_logs) > 1000:
        st.session_state.conversion_logs = st.session_state.conversion_logs[-500:]
    
    # Note: Don't call st.rerun() here as it can cause infinite loops
    # The UI will update when the user triggers an action or state changes

def render_header(i18n: Dict[str, str]):
    """Render the application header"""
    st.title(i18n['app_title'])
    st.markdown(f"*{i18n['subtitle']}*")
    st.divider()

def render_mode_selection(i18n: Dict[str, str]):
    """Render conversion mode selection"""
    st.subheader(i18n['mode_selection'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            i18n['single_mode'], 
            type="primary" if st.session_state.conversion_mode == 'single' else "secondary",
            use_container_width=True
        ):
            st.session_state.conversion_mode = 'single'
            st.rerun()
    
    with col2:
        if st.button(
            i18n['batch_mode'],
            type="primary" if st.session_state.conversion_mode == 'batch' else "secondary", 
            use_container_width=True
        ):
            st.session_state.conversion_mode = 'batch'
            st.rerun()

def render_file_inputs(i18n: Dict[str, str]):
    """Render file input section for single mode"""
    st.subheader(i18n['file_inputs'])
    
    # Input file uploader
    input_file = st.file_uploader(
        i18n['input_file_label'],
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload your PDF file or image to convert"
    )
    
    # JSON file uploader  
    json_file = st.file_uploader(
        i18n['json_file_label'],
        type=['json'],
        help="Upload the MinerU JSON file containing extracted text and layout data"
    )
    
    # Output filename
    output_filename = st.text_input(
        i18n['output_file_label'],
        value="output.pptx",
        help="Specify the name for your output PowerPoint file"
    )
    
    return input_file, json_file, output_filename

def render_conversion_options(i18n: Dict[str, str]):
    """Render conversion options"""
    st.subheader(i18n['conversion_options'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.remove_watermark = st.checkbox(
            i18n['remove_watermark'],
            value=st.session_state.remove_watermark,
            help="Remove watermark and footer elements marked as discarded_blocks in the JSON"
        )
    
    with col2:
        st.session_state.debug_images = st.checkbox(
            i18n['debug_images'],
            value=st.session_state.debug_images,
            help="Generate diagnostic images in the tmp/ folder for debugging"
        )
    
    # Page Selection
    st.subheader(i18n['page_selection'])
    
    page_mode = st.radio(
        "Select pages to convert:",
        options=["all", "single", "range"],
        format_func=lambda x: {
            "all": i18n['all_pages'],
            "single": i18n['single_page'],
            "range": i18n['page_range']
        }[x],
        key="page_selection_mode",
        horizontal=True
    )
    
    if page_mode == "single":
        st.session_state.single_page = st.number_input(
            i18n['page_number'],
            min_value=1,
            value=getattr(st.session_state, 'single_page', 1),
            step=1,
            key="single_page_input"
        )
    elif page_mode == "range":
        col_from, col_to = st.columns(2)
        with col_from:
            st.session_state.from_page = st.number_input(
                i18n['from_page'],
                min_value=1,
                value=getattr(st.session_state, 'from_page', 1),
                step=1,
                key="from_page_input"
            )
        with col_to:
            st.session_state.to_page = st.number_input(
                i18n['to_page'],
                min_value=1,
                value=getattr(st.session_state, 'to_page', 1),
                step=1,
                key="to_page_input"
            )
    
    # Store page selection in session state
    st.session_state.page_selection = get_page_selection_from_ui(page_mode)

def get_page_selection_from_ui(page_mode: str) -> Dict[str, Any]:
    """Get page selection configuration from UI state"""
    if page_mode == "all":
        return {"mode": "all"}
    elif page_mode == "single":
        return {"mode": "single", "page": getattr(st.session_state, 'single_page', 1)}
    elif page_mode == "range":
        from_page = getattr(st.session_state, 'from_page', 1)
        to_page = getattr(st.session_state, 'to_page', 1)
        if from_page > to_page:
            st.error(f"From page ({from_page}) must be less than or equal to To page ({to_page})")
            return {"mode": "all"}  # Fallback
        return {"mode": "range", "from": from_page, "to": to_page}
    
    return {"mode": "all"}  # Fallback

def render_ai_settings(i18n: Dict[str, str]):
    """Render AI text correction settings"""
    st.subheader(i18n['ai_correction'])
    
    # Load AI config if not already loaded
    load_ai_config()
    
    # Enable/Disable AI
    st.session_state.ai_enabled = st.checkbox(
        i18n['enable_ai'],
        value=st.session_state.ai_enabled,
        help="Enable AI-powered text correction for better grammar and formatting"
    )
    
    if st.session_state.ai_enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            # Provider selection
            st.session_state.ai_provider = st.selectbox(
                i18n['ai_provider'],
                options=['openai', 'gemini', 'claude', 'groq'],
                format_func=lambda x: i18n['providers'][x],
                index=['openai', 'gemini', 'claude', 'groq'].index(st.session_state.ai_provider)
            )
            
            # Correction type
            st.session_state.correction_type = st.selectbox(
                i18n['correction_type'],
                options=['grammar_spelling', 'formatting', 'both'],
                format_func=lambda x: i18n.get(f'correction_{x}', x),
                index=['grammar_spelling', 'formatting', 'both'].index(st.session_state.correction_type)
            )
        
        with col2:
            # API Key input
            st.session_state.api_key = st.text_input(
                i18n['api_key'],
                value=st.session_state.api_key,
                type="password",
                help=f"Enter your {i18n['providers'][st.session_state.ai_provider]} API key"
            )
            
            # Model selection - get available models from AI manager
            available_models = []
            try:
                # Convert short provider name to full name
                full_provider_name = get_full_provider_name(st.session_state.ai_provider)
                
                # Try to get available models from the AI manager
                if hasattr(ai_manager, 'services') and full_provider_name in ai_manager.services:
                    service = ai_manager.services[full_provider_name]
                    if hasattr(service, 'get_available_models'):
                        available_models = service.get_available_models()
            except Exception as e:
                log_message(f"Error getting available models: {str(e)}", "warning")
                available_models = []
            
            if available_models:
                current_model = st.session_state.ai_model if st.session_state.ai_model in available_models else available_models[0]
                st.session_state.ai_model = st.selectbox(
                    i18n['ai_model'],
                    options=available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0
                )
            else:
                st.info("Models will be available after authentication")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(i18n['test_auth'], type="secondary"):
                with st.spinner("Testing authentication..."):
                    asyncio.run(test_ai_authentication(i18n))
        
        with col2:
            if st.button(i18n['save_config'], type="secondary"):
                with st.spinner("Saving configuration..."):
                    if asyncio.run(save_ai_configuration()):
                        st.success(i18n['config_saved'])
                        time.sleep(1)
                        st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Models", type="secondary"):
                with st.spinner("Fetching models..."):
                    asyncio.run(refresh_ai_models(i18n))

async def test_ai_authentication(i18n: Dict[str, str]):
    """Test AI authentication"""
    try:
        # Convert short provider name to full name
        full_provider_name = get_full_provider_name(st.session_state.ai_provider)
        
        # Save current config temporarily
        await save_ai_configuration()
        
        # Try to authenticate (using full provider name)
        await ai_manager.set_active_service(full_provider_name)
        active_service = ai_manager.get_active_service()
        
        if active_service:
            await active_service.authenticate()
            st.success(i18n['auth_success'])
            
            # Fetch models after successful authentication
            await refresh_ai_models(i18n, show_message=False)
        else:
            st.error("Service not available")
            
    except Exception as e:
        st.error(i18n['auth_failed'].format(str(e)))

async def refresh_ai_models(i18n: Dict[str, str], show_message: bool = True):
    """Refresh available AI models"""
    try:
        # Convert short provider name to full name
        full_provider_name = get_full_provider_name(st.session_state.ai_provider)
        
        await ai_manager.set_active_service(full_provider_name)
        active_service = ai_manager.get_active_service()
        
        if active_service:
            await active_service.authenticate_and_fetch_models()
            models = active_service.get_available_models()
            
            if models:
                # Set first model as default if none selected
                if not st.session_state.ai_model or st.session_state.ai_model not in models:
                    st.session_state.ai_model = models[0]
                    # Save the selected model to config (using full provider name)
                    ai_config.set_model(full_provider_name, st.session_state.ai_model)
                    ai_config.save_config()
                
                if show_message:
                    st.success(f"Found {len(models)} models")
                    st.rerun()
            else:
                if show_message:
                    st.warning("No models found")
        else:
            if show_message:
                st.error("Service not authenticated")
                
    except Exception as e:
        if show_message:
            st.error(f"Failed to fetch models: {str(e)}")

def render_batch_mode(i18n: Dict[str, str]):
    """Render batch processing interface"""
    st.subheader(i18n['batch_tasks'])
    
    # Add task section
    with st.expander(i18n['add_task'], expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            batch_input = st.file_uploader("Input File", type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'], key="batch_input")
            batch_json = st.file_uploader("JSON File", type=['json'], key="batch_json")
        
        with col2:
            batch_output = st.text_input("Output Filename", value="output.pptx", key="batch_output")
            
            col_a, col_b = st.columns(2)
            with col_a:
                batch_watermark = st.checkbox("Remove Watermark", value=True, key="batch_watermark")
            with col_b:
                batch_debug = st.checkbox("Debug Images", value=False, key="batch_debug")
                
            # Page selection for batch task
            st.write("**Page Selection:**")
            batch_page_mode = st.radio(
                "Pages to convert:",
                options=["all", "single", "range"],
                format_func=lambda x: {
                    "all": i18n['all_pages'],
                    "single": i18n['single_page'], 
                    "range": i18n['page_range']
                }[x],
                key="batch_page_mode",
                horizontal=True
            )
            
            batch_page_selection = {"mode": "all"}
            if batch_page_mode == "single":
                page_num = st.number_input("Page Number", min_value=1, value=1, key="batch_single_page")
                batch_page_selection = {"mode": "single", "page": page_num}
            elif batch_page_mode == "range":
                col_from, col_to = st.columns(2)
                with col_from:
                    from_page = st.number_input("From Page", min_value=1, value=1, key="batch_from_page")
                with col_to:
                    to_page = st.number_input("To Page", min_value=1, value=1, key="batch_to_page")
                
                if from_page <= to_page:
                    batch_page_selection = {"mode": "range", "from": from_page, "to": to_page}
                else:
                    st.error("'From' page must be less than or equal to 'To' page")
                    batch_page_selection = {"mode": "all"}
        
        if st.button("➕ Add to Queue", type="primary", use_container_width=True):
            if batch_input and batch_json and batch_output:
                # Save uploaded files temporarily
                input_path = save_uploaded_file(batch_input, "batch_input")
                json_path = save_uploaded_file(batch_json, "batch_json")
                
                task = {
                    'input_file': batch_input.name,
                    'json_file': batch_json.name,
                    'output_file': batch_output,
                    'input_path': input_path,
                    'json_path': json_path,
                    'remove_watermark': batch_watermark,
                    'debug_images': batch_debug,
                    'page_selection': batch_page_selection,
                    'status': 'pending'
                }
                
                st.session_state.batch_tasks.append(task)
                st.success(i18n['task_added'])
                st.rerun()
            else:
                st.error(i18n['no_files'])
    
    # Task list
    if st.session_state.batch_tasks:
        st.subheader(i18n['task_list'])
        
        for i, task in enumerate(st.session_state.batch_tasks):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    status_icon = {"pending": "⏳", "processing": "🔄", "completed": "✅", "error": "❌"}.get(task['status'], "⏳")
                    st.write(f"{status_icon} **{task['input_file']}** → {task['output_file']}")
                
                with col2:
                    options = []
                    if task['remove_watermark']:
                        options.append("No Watermark")
                    if task['debug_images']:
                        options.append("Debug")
                    
                    # Add page selection info
                    page_sel = task.get('page_selection', {"mode": "all"})
                    if page_sel['mode'] == 'single':
                        options.append(f"Page {page_sel['page']}")
                    elif page_sel['mode'] == 'range':
                        options.append(f"Pages {page_sel['from']}-{page_sel['to']}")
                    elif page_sel['mode'] == 'all':
                        options.append("All Pages")
                    
                    st.caption(", ".join(options) if options else "Standard")
                
                with col3:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete task"):
                        st.session_state.batch_tasks.pop(i)
                        st.rerun()
                
                st.divider()
        
        # Start batch processing
        if st.button(i18n['start_batch'], type="primary", use_container_width=True):
            if not st.session_state.conversion_in_progress:
                start_batch_processing(i18n)
            else:
                st.warning("Conversion already in progress")
    else:
        st.info("No tasks in queue. Add tasks above to start batch processing.")

def save_uploaded_file(uploaded_file, prefix: str) -> str:
    """Save uploaded file to temporary directory and return path"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{prefix}_{uploaded_file.name}")
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return file_path

def render_cache_management(i18n: Dict[str, str]):
    """Render cache management interface"""
    st.subheader(i18n['cache_management'])
    
    # Get cache stats
    try:
        cache_stats = global_cache_manager.get_cache_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cached Texts", cache_stats['total_entries'])
        
        with col2:
            st.metric("Cache Size", f"{cache_stats.get('memory_usage_mb', 0):.1f} MB")
        
        with col3:
            providers = len(cache_stats.get('providers', {}))
            st.metric("AI Providers", providers)
        
        with col4:
            # Calculate hit rate from recent activity if available
            total = cache_stats.get('total_entries', 0)
            recent = cache_stats.get('recent_24h', 0)
            hit_rate = (recent / total * 100) if total > 0 else 0
            st.metric("Recent Activity", f"{hit_rate:.1f}%")
        
        # Provider breakdown
        providers_dict = cache_stats.get('providers', {})
        if providers_dict:
            st.write("**Cache by Provider:**")
            for provider, count in providers_dict.items():
                st.write(f"• {provider}: {count} texts")
        
        # Cache actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Recent Corrections", use_container_width=True):
                recent = global_cache_manager.get_recent_corrections(10)
                if recent:
                    st.write("**Recent Corrections:**")
                    for correction in recent:
                        with st.expander(f"{correction['original'][:50]}..."):
                            st.write(f"**Original:** {correction['original']}")
                            st.write(f"**Corrected:** {correction['corrected']}")
                            st.write(f"**Provider:** {correction['provider']}")
                            st.write(f"**Date:** {correction['timestamp']}")
                else:
                    st.info("No recent corrections found")
        
        with col2:
            if st.button("🧹 Clear Current Provider", use_container_width=True):
                if st.session_state.ai_enabled and st.session_state.ai_provider:
                    # Convert short provider name to full name
                    full_provider_name = get_full_provider_name(st.session_state.ai_provider)
                    global_cache_manager.clear_by_provider(full_provider_name)
                    st.success(f"Cleared {st.session_state.ai_provider} cache")
                    st.rerun()
                else:
                    st.warning("No AI provider selected")
        
        with col3:
            if st.button("🗑️ Clear All Cache", use_container_width=True):
                global_cache_manager.clear_all()
                st.success(i18n['cache_cleared'])
                st.rerun()
                
    except Exception as e:
        st.error(f"Error loading cache stats: {e}")

def render_logs(i18n: Dict[str, str]):
    """Render conversion logs"""
    st.subheader(i18n['logs'])
    
    if st.session_state.conversion_logs:
        # Show number of log entries
        st.write(f"**{len(st.session_state.conversion_logs)} log entries** (showing last 50)")
        
        # Show logs in reverse chronological order with better formatting
        with st.container():
            # Create a scrollable area for logs
            log_text_all = []
            
            for log_entry in reversed(st.session_state.conversion_logs[-50:]):  # Show last 50 entries
                level_icon = {
                    'info': 'ℹ️',
                    'success': '✅', 
                    'warning': '⚠️',
                    'error': '❌'
                }.get(log_entry['level'], 'ℹ️')
                
                # Format each log entry
                log_line = f"{level_icon} `{log_entry['timestamp']}` {log_entry['message']}"
                st.markdown(log_line)
            
            # Separator
            st.divider()
        
        # Clear logs button
        if st.button("🧹 Clear Logs"):
            st.session_state.conversion_logs = []
            st.rerun()
    else:
        st.info("No conversion logs yet. Start a conversion to see progress here.")

def render_help_section(i18n: Dict[str, str]):
    """Render help and information section"""
    st.subheader(i18n['help_section'])
    
    with st.expander(i18n['json_help'], expanded=False):
        st.write(i18n['json_help_text'])
        if st.button("🔗 Open MinerU Website"):
            st.markdown("[MinerU PDF Extractor](https://github.com/opendatalab/MinerU)")
    
    with st.expander(i18n['supported_formats'], expanded=False):
        st.write(i18n['formats_list'])
    
    with st.expander("🔧 System Information", expanded=False):
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Working Directory:** {os.getcwd()}")
        st.write(f"**Temp Directory:** {tempfile.gettempdir()}")

def start_conversion(input_file, json_file, output_filename: str, i18n: Dict[str, str]):
    """Start single file conversion"""
    if not input_file or not json_file:
        st.error(i18n['no_files'])
        return
    
    st.session_state.conversion_in_progress = True
    
    try:
        # Save uploaded files temporarily
        input_path = save_uploaded_file(input_file, "input")
        json_path = save_uploaded_file(json_file, "json")
        
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="mineru2ppt_")
        output_path = os.path.join(output_dir, output_filename)
        
        log_message(f"Starting conversion: {input_file.name} → {output_filename}")
        
        # Create containers for live updates
        progress_container = st.container()
        status_container = st.container()
        log_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            
        with status_container:
            status_text = st.empty()
        
        # Update progress manually with live logging
        with progress_container:
            progress_bar.progress(0.1)
        with status_container:
            status_text.text("🔄 Initializing conversion...")
        log_message("Initializing conversion...")
        
        # Show current logs
        with log_container:
            if st.session_state.conversion_logs:
                latest_log = st.session_state.conversion_logs[-1]
                st.info(f"Latest: {latest_log['message']}")
        
        with progress_container:
            progress_bar.progress(0.3)
        with status_container:
            status_text.text("📁 Processing files...")
        log_message("Processing files...")
        
        with progress_container:
            progress_bar.progress(0.5)
        with status_container:
            status_text.text("🔄 Running conversion...")
        log_message("Running conversion engine...")
        
        # Run conversion
        try:
            success = convert_mineru_to_ppt(
                json_path=json_path,
                input_path=input_path,
                output_ppt_path=output_path,
                remove_watermark=st.session_state.remove_watermark,
                debug_images=st.session_state.debug_images,
                enable_ai_correction=st.session_state.ai_enabled,
                page_selection=st.session_state.page_selection
            )
        except Exception as conversion_error:
            success = False
            log_message(f"Conversion engine error: {str(conversion_error)}", "error")
            with status_container:
                status_text.text("❌ Conversion engine error")
        
        if success:
            with progress_container:
                progress_bar.progress(1.0)
            with status_container:
                status_text.text("✅ Conversion completed successfully!")
            log_message("Conversion completed successfully!", "success")
            
            # Provide download link
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PPTX",
                        data=f.read(),
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        type="primary"
                    )
                log_message("PPTX file ready for download", "success")
            else:
                st.error("Output file was not created")
                log_message("Output file was not created", "error")
        else:
            with progress_container:
                progress_bar.progress(0.8)
            with status_container:
                status_text.text("❌ Conversion failed")
            st.error("Conversion failed - check logs for details")
            log_message("Conversion failed", "error")
        
        # Cleanup temporary files
        try:
            os.unlink(input_path)
            os.unlink(json_path)
            if success and os.path.exists(output_path):
                # Keep output file for download, clean it up later
                pass
            else:
                shutil.rmtree(output_dir)
        except:
            pass
            
    except Exception as e:
        st.error(i18n['conversion_error'].format(str(e)))
        log_message(f"Conversion error: {str(e)}", "error")
    
    finally:
        st.session_state.conversion_in_progress = False

def start_batch_processing(i18n: Dict[str, str]):
    """Start batch processing of tasks"""
    st.session_state.conversion_in_progress = True
    
    log_message("Starting batch processing", "info")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tasks = len(st.session_state.batch_tasks)
    
    for i, task in enumerate(st.session_state.batch_tasks):
        if task['status'] != 'pending':
            continue
            
        task['status'] = 'processing'
        st.rerun()
        
        progress = i / total_tasks
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{total_tasks}: {task['input_file']}")
        
        try:
            # Create output directory
            output_dir = tempfile.mkdtemp(prefix="mineru2ppt_batch_")
            output_path = os.path.join(output_dir, task['output_file'])
            
            log_message(f"Processing task {i+1}: {task['input_file']}")
            
            # Run conversion
            success = convert_mineru_to_ppt(
                json_path=task['json_path'],
                input_path=task['input_path'],
                output_ppt_path=output_path,
                remove_watermark=task['remove_watermark'],
                debug_images=task['debug_images'],
                enable_ai_correction=st.session_state.ai_enabled,
                page_selection=task.get('page_selection', {"mode": "all"})
            )
            
            if success:
                task['status'] = 'completed'
                task['output_path'] = output_path
                log_message(f"✅ Task {i+1} completed: {task['input_file']}", "success")
            else:
                task['status'] = 'error'
                log_message(f"❌ Task {i+1} failed: {task['input_file']}", "error")
                
        except Exception as e:
            task['status'] = 'error'
            log_message(f"❌ Task {i+1} error: {str(e)}", "error")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Batch processing completed!")
    log_message(i18n['batch_complete'], "success")
    
    st.session_state.conversion_in_progress = False
    st.rerun()

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Get translations
    i18n = TRANSLATIONS[st.session_state.language]
    
    # Language selector in sidebar
    with st.sidebar:
        language = st.selectbox(
            "🌐 Language / 语言",
            options=['en', 'zh'],
            format_func=lambda x: '🇺🇸 English' if x == 'en' else '🇨🇳 中文',
            index=['en', 'zh'].index(st.session_state.language)
        )
        
        if language != st.session_state.language:
            st.session_state.language = language
            st.rerun()
        
        st.divider()
        
        # Quick actions in sidebar
        st.write("**🚀 Quick Actions**")
        
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
        
        if st.button("🧹 Clear All Data", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'language':
                    del st.session_state[key]
            st.rerun()
        
        st.divider()
        
        # Cache management in sidebar
        render_cache_management(i18n)
    
    # Main content
    render_header(i18n)
    
    # Mode selection
    render_mode_selection(i18n)
    
    if st.session_state.conversion_mode == 'single':
        # Single file mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File inputs
            input_file, json_file, output_filename = render_file_inputs(i18n)
            
            # Conversion options
            render_conversion_options(i18n)
            
            # Start conversion button
            if st.button(i18n['start_conversion'], type="primary", use_container_width=True):
                if not st.session_state.conversion_in_progress:
                    start_conversion(input_file, json_file, output_filename, i18n)
                else:
                    st.warning("Conversion already in progress")
        
        with col2:
            # AI settings
            render_ai_settings(i18n)
            
            # Help section
            render_help_section(i18n)
    
    else:
        # Batch mode
        render_batch_mode(i18n)
        
        # AI settings for batch
        with st.expander("🤖 AI Settings for Batch", expanded=False):
            render_ai_settings(i18n)
    
    # Logs section (always visible at bottom)
    st.divider()
    render_logs(i18n)

if __name__ == "__main__":
    main()