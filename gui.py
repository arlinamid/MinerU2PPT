#!/usr/bin/env python3
"""
MinerU2PPTX GUI Application

This enhanced version is inspired by and extends the original MinerU2PPT project:
https://github.com/JuniverseCoder/MinerU2PPT

Author: Arlinamid (Rózsavölgyi János)
GitHub: https://github.com/arlinamid
Support: https://buymeacoffee.com/arlinamid

Enhancements include:
- AI-powered text correction (OpenAI, Gemini, Claude, Groq)
- Multi-language support (English, Hungarian, Chinese)
- Advanced text rendering with collision detection
- Professional About/Support dialogs
- Comprehensive error handling and caching
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import queue
import sys
import os
import shutil
import webbrowser
import locale
import asyncio
import logging
from tkinterdnd2 import DND_FILES, TkinterDnD
import urllib.request
from PIL import Image, ImageTk
from converter.generator import convert_mineru_to_ppt
from converter.config import ai_config
from converter.ai_services import ai_manager
from converter.cache_manager import global_cache_manager
from converter.image_downloader import ImageExtractor
from translations import get_translator

logger = logging.getLogger(__name__)

class QueueHandler:
    def __init__(self, queue): self.queue = queue
    def write(self, text): self.queue.put(text)
    def flush(self): pass

class AddTaskDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.i18n = parent.i18n
        self.title(self.i18n['add_task_title'])
        self.geometry("600x350")

        self.input_path = tk.StringVar()
        self.json_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.remove_watermark = tk.BooleanVar(value=True)
        
        # Page selection variables
        self.page_selection_mode = tk.StringVar(value="all")  # "all", "single", "range"
        self.single_page_num = tk.StringVar(value="1")
        self.page_range_from = tk.StringVar(value="1")
        self.page_range_to = tk.StringVar(value="1")
        
        self.result = None

        self._create_widgets()
        self.transient(parent)
        self.grab_set()
        self.wait_window(self)

    def _create_widgets(self):
        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.grid_columnconfigure(1, weight=1)

        tk.Label(frame, text=self.i18n['input_file_label']).grid(row=0, column=0, sticky="w", pady=5)
        input_entry = tk.Entry(frame, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        input_entry.drop_target_register(DND_FILES)
        input_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.input_path))
        tk.Button(frame, text=self.i18n['browse_button'], command=self._browse_input).grid(row=0, column=2, padx=5)

        tk.Label(frame, text=self.i18n['json_file_label']).grid(row=1, column=0, sticky="w", pady=5)
        json_entry = tk.Entry(frame, textvariable=self.json_path)
        json_entry.grid(row=1, column=1, sticky="ew", padx=5)
        json_entry.drop_target_register(DND_FILES)
        json_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.json_path))
        tk.Button(frame, text=self.i18n['browse_button'], command=self._browse_json).grid(row=1, column=2, padx=5)

        tk.Label(frame, text=self.i18n['output_file_label']).grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(frame, text=self.i18n['save_as_button'], command=self._save_pptx).grid(row=2, column=2, padx=5)

        # Page Selection Frame
        page_frame = tk.LabelFrame(frame, text=self.i18n.get('page_selection_label', 'Page Selection'), padx=5, pady=5)
        page_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        page_frame.grid_columnconfigure(1, weight=1)
        
        # All Pages option
        tk.Radiobutton(page_frame, text=self.i18n.get('all_pages_option', 'All Pages'), 
                      variable=self.page_selection_mode, value="all",
                      command=self._on_page_selection_change).grid(row=0, column=0, sticky="w", padx=5)
        
        # Single Page option
        single_frame = tk.Frame(page_frame)
        single_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=5)
        tk.Radiobutton(single_frame, text=self.i18n.get('single_page_option', 'Single Page:'), 
                      variable=self.page_selection_mode, value="single",
                      command=self._on_page_selection_change).pack(side=tk.LEFT)
        self.single_page_entry = tk.Entry(single_frame, textvariable=self.single_page_num, width=5)
        self.single_page_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Page Range option
        range_frame = tk.Frame(page_frame)
        range_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5)
        tk.Radiobutton(range_frame, text=self.i18n.get('page_range_option', 'Page Range:'), 
                      variable=self.page_selection_mode, value="range",
                      command=self._on_page_selection_change).pack(side=tk.LEFT)
        tk.Label(range_frame, text=self.i18n.get('from_label', 'From:')).pack(side=tk.LEFT, padx=(10, 0))
        self.page_from_entry = tk.Entry(range_frame, textvariable=self.page_range_from, width=5)
        self.page_from_entry.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(range_frame, text=self.i18n.get('to_label', 'To:')).pack(side=tk.LEFT, padx=(10, 0))
        self.page_to_entry = tk.Entry(range_frame, textvariable=self.page_range_to, width=5)
        self.page_to_entry.pack(side=tk.LEFT, padx=(5, 0))

        options_frame = tk.Frame(frame)
        options_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")
        tk.Checkbutton(options_frame, text=self.i18n['remove_watermark_checkbox'], variable=self.remove_watermark).pack(side=tk.LEFT)

        buttons_frame = tk.Frame(frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=5)
        tk.Button(buttons_frame, text=self.i18n['ok_button'], command=self._on_ok, width=10).pack(side=tk.LEFT, padx=10)
        tk.Button(buttons_frame, text=self.i18n['cancel_button'], command=self.destroy, width=10).pack(side=tk.LEFT, padx=10)
        
        # Initially disable page selection entries
        self._on_page_selection_change()
    
    def _on_page_selection_change(self):
        """Handle page selection mode changes"""
        mode = self.page_selection_mode.get()
        
        # Enable/disable entry fields based on selection
        if mode == "single":
            self.single_page_entry.config(state="normal")
            self.page_from_entry.config(state="disabled")
            self.page_to_entry.config(state="disabled")
        elif mode == "range":
            self.single_page_entry.config(state="disabled")
            self.page_from_entry.config(state="normal")
            self.page_to_entry.config(state="normal")
        else:  # "all"
            self.single_page_entry.config(state="disabled")
            self.page_from_entry.config(state="disabled")
            self.page_to_entry.config(state="disabled")

    def _on_drop(self, event, var):
        filepath = event.data.strip('{}')
        var.set(filepath)
        if var == self.input_path:
            self._set_default_output_path(filepath)

    def _set_default_output_path(self, in_path):
        if not self.output_path.get() and in_path:
            self.output_path.set(os.path.splitext(in_path)[0] + ".pptx")

    def _browse_input(self):
        filetypes = [("Supported Files", "*.pdf *.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes, parent=self)
        if path: self.input_path.set(path); self._set_default_output_path(path)

    def _browse_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")], parent=self)
        if path: self.json_path.set(path)

    def _save_pptx(self):
        path = filedialog.asksaveasfilename(defaultextension=".pptx", filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")], parent=self)
        if path: self.output_path.set(path)

    def _on_ok(self):
        input_f, json_f, output_f = self.input_path.get(), self.json_path.get(), self.output_path.get()
        if not all([input_f, json_f, output_f]):
            messagebox.showerror(self.i18n['error_title'], self.i18n['error_all_paths'], parent=self)
            return
        
        # Validate page selection
        page_info = self._get_page_selection_info()
        if page_info is None:
            return  # Error already shown
        
        self.result = {
            "input": input_f, 
            "json": json_f, 
            "output": output_f, 
            "remove_watermark": self.remove_watermark.get(),
            "page_selection": page_info
        }
        self.destroy()
    
    def _get_page_selection_info(self):
        """Get and validate page selection information"""
        mode = self.page_selection_mode.get()
        
        if mode == "all":
            return {"mode": "all"}
        elif mode == "single":
            try:
                page_num = int(self.single_page_num.get().strip())
                if page_num < 1:
                    raise ValueError("Page number must be positive")
                return {"mode": "single", "page": page_num}
            except ValueError:
                messagebox.showerror(self.i18n['error_title'], 
                                   "Please enter a valid page number (positive integer)", parent=self)
                return None
        elif mode == "range":
            try:
                from_page = int(self.page_range_from.get().strip())
                to_page = int(self.page_range_to.get().strip())
                if from_page < 1 or to_page < 1:
                    raise ValueError("Page numbers must be positive")
                if from_page > to_page:
                    raise ValueError("'From' page must be less than or equal to 'To' page")
                return {"mode": "range", "from": from_page, "to": to_page}
            except ValueError as e:
                messagebox.showerror(self.i18n['error_title'], 
                                   f"Please enter valid page numbers: {str(e)}", parent=self)
                return None
        
        return {"mode": "all"}  # Fallback


class AISettingsDialog(tk.Toplevel):
    """Dialog for configuring AI text correction settings"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.i18n = parent.i18n
        self.title(self.i18n['ai_config_title'])
        self.geometry("650x580")
        self.resizable(False, False)
        
        self.transient(parent)
        self.grab_set()
        
        # Variables
        self.enable_ai = tk.BooleanVar(value=ai_config.is_text_correction_enabled())
        self.selected_provider = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.api_key = tk.StringVar()
        self.correction_type = tk.StringVar(value=ai_config.get_correction_type())
        # Set initial correction type value
        initial_type = ai_config.get_correction_type()
        self.correction_type.set(initial_type)
        self.auto_apply = tk.BooleanVar(value=ai_config.is_auto_apply_enabled())
        self.show_changes = tk.BooleanVar(value=ai_config.should_show_changes())
        self.backup_original = tk.BooleanVar(value=ai_config.should_backup_original())
        
        # Provider models will be fetched dynamically after authentication
        self.provider_models = {
            self.i18n['ai_provider_openai']: [],
            self.i18n['ai_provider_gemini']: [],
            self.i18n['ai_provider_claude']: [],
            self.i18n['ai_provider_groq']: []
        }
        
        self._create_widgets()
        self._load_current_config()
        self._on_provider_changed()
        
        # Center on parent
        self.update_idletasks()
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        x = parent_x + (parent_width - self.winfo_width()) // 2
        y = parent_y + (parent_height - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _create_widgets(self):
        """Create all dialog widgets"""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main enable/disable checkbox
        enable_frame = ttk.Frame(main_frame)
        enable_frame.pack(fill=tk.X, pady=(0, 10))
        
        enable_cb = ttk.Checkbutton(
            enable_frame,
            text=self.i18n['enable_ai_checkbox'],
            variable=self.enable_ai,
            command=self._on_enable_changed
        )
        enable_cb.pack(anchor=tk.W)
        
        # Provider configuration frame
        self.config_frame = ttk.LabelFrame(main_frame, text=self.i18n['ai_provider_label'], padding=10)
        self.config_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Provider selection
        provider_frame = ttk.Frame(self.config_frame)
        provider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(provider_frame, text=self.i18n['ai_provider_label']).pack(side=tk.LEFT)
        self.provider_combo = ttk.Combobox(
            provider_frame,
            textvariable=self.selected_provider,
            values=list(self.provider_models.keys()),
            state="readonly",
            width=20
        )
        self.provider_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.provider_combo.bind('<<ComboboxSelected>>', lambda e: self._on_provider_changed())
        
        # Model selection
        model_frame = ttk.Frame(self.config_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text=self.i18n['ai_model_label']).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            state="readonly",
            width=25
        )
        self.model_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Refresh models button
        self.refresh_models_button = ttk.Button(
            model_frame,
            text="↻",  # Refresh symbol
            command=self._refresh_models,
            width=3
        )
        self.refresh_models_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # API Key
        key_frame = ttk.Frame(self.config_frame)
        key_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(key_frame, text=self.i18n['api_key_label']).pack(side=tk.LEFT)
        self.key_entry = ttk.Entry(
            key_frame,
            textvariable=self.api_key,
            show="*",
            width=30
        )
        self.key_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # API Key helper buttons
        key_buttons_frame = ttk.Frame(key_frame)
        key_buttons_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.get_key_button = ttk.Button(
            key_buttons_frame,
            text=self.i18n['get_api_key_button'],
            command=self._open_auth_url
        )
        self.get_key_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.auth_help_button = ttk.Button(
            key_buttons_frame,
            text=self.i18n['auth_help_button'],
            command=self._show_auth_instructions,
            width=6
        )
        self.auth_help_button.pack(side=tk.LEFT)
        
        # Test authentication button
        test_frame = ttk.Frame(self.config_frame)
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.test_button = ttk.Button(
            test_frame,
            text=self.i18n['test_auth_button'],
            command=self._test_authentication
        )
        self.test_button.pack(side=tk.LEFT)
        
        self.auth_status_label = ttk.Label(test_frame, text="")
        self.auth_status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Correction settings
        settings_frame = ttk.LabelFrame(main_frame, text=self.i18n['correction_type_label'], padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Correction type
        correction_frame = ttk.Frame(settings_frame)
        correction_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(correction_frame, text=self.i18n['correction_type_label']).pack(side=tk.LEFT)
        
        # Create a mapping for correction types
        self.correction_type_map = {
            self.i18n['correction_grammar']: "grammar_spelling",
            self.i18n['correction_formatting']: "formatting",
            self.i18n['correction_both']: "both"
        }
        self.correction_type_reverse_map = {v: k for k, v in self.correction_type_map.items()}
        
        correction_combo = ttk.Combobox(
            correction_frame,
            values=list(self.correction_type_map.keys()),
            state="readonly",
            width=25
        )
        correction_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Set current value
        current_type = ai_config.get_correction_type()
        display_name = self.correction_type_reverse_map.get(current_type, list(self.correction_type_map.keys())[0])
        correction_combo.set(display_name)
        
        # Bind to update the internal value
        correction_combo.bind('<<ComboboxSelected>>', self._on_correction_type_changed)
        
        self.correction_combo = correction_combo
        
        # Additional settings
        ttk.Checkbutton(
            settings_frame,
            text=self.i18n['auto_apply_checkbox'],
            variable=self.auto_apply
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(
            settings_frame,
            text=self.i18n['show_changes_checkbox'],
            variable=self.show_changes
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(
            settings_frame,
            text=self.i18n['backup_original_checkbox'],
            variable=self.backup_original
        ).pack(anchor=tk.W, pady=2)
        
        # Cache Management Section
        cache_frame = ttk.LabelFrame(main_frame, text="Cache Management", padding=10)
        cache_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Cache statistics label
        self.cache_stats_label = ttk.Label(
            cache_frame,
            text="Loading cache statistics..."
        )
        self.cache_stats_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Cache management buttons
        cache_buttons_frame = ttk.Frame(cache_frame)
        cache_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            cache_buttons_frame,
            text="View Cache Stats",
            command=self._show_cache_stats
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            cache_buttons_frame,
            text="Clear All Cache",
            command=self._clear_all_cache
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            cache_buttons_frame,
            text="Clear Current Provider",
            command=self._clear_provider_cache
        ).pack(side=tk.LEFT)
        
        # Update cache stats
        self._update_cache_stats()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text=self.i18n['save_ai_config_button'],
            command=self._save_config
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(
            button_frame,
            text=self.i18n['cancel_button'],
            command=self.destroy
        ).pack(side=tk.RIGHT)
    
    def _load_current_config(self):
        """Load current AI configuration"""
        active_provider = ai_config.get_active_provider()
        if active_provider:
            # Map internal provider names to display names
            provider_map = {
                "OpenAI": self.i18n['ai_provider_openai'],
                "Google Gemini": self.i18n['ai_provider_gemini'],
                "Anthropic Claude": self.i18n['ai_provider_claude'],
                "Groq": self.i18n['ai_provider_groq']
            }
            display_name = provider_map.get(active_provider, active_provider)
            if display_name in self.provider_models:
                self.selected_provider.set(display_name)
                model = ai_config.get_model(active_provider)
                if model:
                    self.selected_model.set(model)
                api_key = ai_config.get_api_key(active_provider)
                if api_key:
                    self.api_key.set(api_key)
    
    def _on_enable_changed(self):
        """Handle enable/disable checkbox change"""
        enabled = self.enable_ai.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Enable/disable all configuration widgets
        for widget in self.config_frame.winfo_children():
            self._set_widget_state_recursive(widget, state)
        
        # Specifically handle our custom buttons
        if hasattr(self, 'get_key_button'):
            self.get_key_button.configure(state=state)
        if hasattr(self, 'auth_help_button'):
            self.auth_help_button.configure(state=state)
        if hasattr(self, 'refresh_models_button'):
            self.refresh_models_button.configure(state=state)
    
    def _set_widget_state_recursive(self, widget, state):
        """Recursively set widget state"""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass  # Some widgets don't support state
        
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)
    
    def _on_provider_changed(self):
        """Handle provider selection change"""
        provider_display = self.selected_provider.get()
        self.auth_status_label.configure(text="")
        
        if provider_display:
            # Clear model list initially - will be populated after authentication
            self.model_combo.configure(values=[])
            self.selected_model.set("")
            
            # Map display name to internal name
            provider_map = {
                self.i18n['ai_provider_openai']: "OpenAI",
                self.i18n['ai_provider_gemini']: "Google Gemini",
                self.i18n['ai_provider_claude']: "Anthropic Claude",
                self.i18n['ai_provider_groq']: "Groq"
            }
            provider = provider_map.get(provider_display, provider_display)
            
            # Load saved API key for this provider
            saved_key = ai_config.get_api_key(provider)
            saved_model = ai_config.get_model(provider)
            
            if saved_key:
                self.api_key.set(saved_key)
                self.selected_model.set(saved_model or "")
                
                # Show hint to authenticate to get models
                self.auth_status_label.configure(
                    text="Click 'Test Authentication' to load models",
                    foreground="blue"
                )
                
                # If provider is enabled, try to fetch real models in background
                if ai_config.is_provider_enabled(provider):
                    self._refresh_models_background(provider)
            else:
                self.api_key.set("")
                self.auth_status_label.configure(
                    text="Enter API key and test authentication",
                    foreground="gray"
                )
    
    def _refresh_models_background(self, provider):
        """Refresh models in background without blocking UI"""
        def refresh_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create service if not exists
                api_key = ai_config.get_api_key(provider)
                if not api_key:
                    return
                
                service = ai_manager.create_service(provider, api_key)
                ai_manager.register_service(provider, service)
                
                # Authenticate and fetch models
                auth_success, models = loop.run_until_complete(ai_manager.authenticate_and_fetch_models(provider))
                
                # Update UI in main thread
                if models and models != service.get_available_models():  # Only update if different from defaults
                    self.after(0, lambda: self._update_model_list(models))
                
            except Exception as e:
                logger.debug(f"Background model refresh failed for {provider}: {e}")
            finally:
                loop.close()
        
        # Run in background thread
        threading.Thread(target=refresh_async, daemon=True).start()
    
    def _update_model_list(self, models):
        """Update model dropdown with new models"""
        current_selection = self.selected_model.get()
        self.model_combo.configure(values=models)
        
        # Try to keep current selection if it's still valid
        if current_selection in models:
            self.selected_model.set(current_selection)
        elif models:
            self.selected_model.set(models[0])
    
    def _refresh_models(self):
        """Manually refresh available models"""
        provider_display = self.selected_provider.get()
        api_key = self.api_key.get().strip()
        
        if not provider_display:
            messagebox.showwarning("Warning", "Please select a provider first")
            return
        
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first")
            return
        
        # Map display name to internal name
        provider_map = {
            self.i18n['ai_provider_openai']: "OpenAI",
            self.i18n['ai_provider_gemini']: "Google Gemini",
            self.i18n['ai_provider_claude']: "Anthropic Claude",
            self.i18n['ai_provider_groq']: "Groq"
        }
        provider = provider_map.get(provider_display, provider_display)
        
        # Disable button and show status
        self.refresh_models_button.configure(state=tk.DISABLED)
        self.auth_status_label.configure(text=self.i18n['fetching_models'], foreground="blue")
        self.update()
        
        def refresh_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create and authenticate service
                service = ai_manager.create_service(provider, api_key)
                ai_manager.register_service(provider, service)
                
                success, models = loop.run_until_complete(
                    ai_manager.authenticate_and_fetch_models(provider)
                )
                
                # Update UI in main thread
                self.after(0, lambda: self._refresh_complete(success, models, provider))
                
            except Exception as e:
                self.after(0, lambda: self._refresh_complete(False, [], provider, str(e)))
            finally:
                loop.close()
        
        # Run in background thread
        threading.Thread(target=refresh_async, daemon=True).start()
    
    def _refresh_complete(self, success, models, provider, error=None):
        """Handle model refresh completion"""
        self.refresh_models_button.configure(state=tk.NORMAL)
        
        if success and models:
            self._update_model_list(models)
            self.auth_status_label.configure(
                text=self.i18n['models_found'].format(len(models)),
                foreground="green"
            )
        else:
            error_msg = error or "Failed to fetch models"
            self.auth_status_label.configure(
                text=self.i18n['model_fetch_failed'].format(error_msg),
                foreground="red"
            )
    
    def _on_correction_type_changed(self, event):
        """Handle correction type selection change"""
        display_name = self.correction_combo.get()
        internal_value = self.correction_type_map.get(display_name, "grammar_spelling")
        self.correction_type.set(internal_value)
    
    def _test_authentication(self):
        """Test authentication with selected provider"""
        if not self.enable_ai.get():
            return
        
        provider_display = self.selected_provider.get()
        api_key = self.api_key.get().strip()
        model = self.selected_model.get()
        
        if not all([provider_display, api_key, model]):
            messagebox.showwarning("Warning", "Please fill in all fields")
            return
        
        # Map display name to internal name
        provider_map = {
            self.i18n['ai_provider_openai']: "OpenAI",
            self.i18n['ai_provider_gemini']: "Google Gemini",
            self.i18n['ai_provider_claude']: "Anthropic Claude",
            self.i18n['ai_provider_groq']: "Groq"
        }
        provider = provider_map.get(provider_display, provider_display)
        
        # Disable test button and show status
        self.test_button.configure(state=tk.DISABLED)
        self.auth_status_label.configure(text="Testing...", foreground="blue")
        self.update()
        
        def test_auth_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Create service and register it
                service = ai_manager.create_service(provider, api_key, model)
                ai_manager.register_service(provider, service)
                
                # Authenticate and fetch available models
                success, available_models = loop.run_until_complete(
                    ai_manager.authenticate_and_fetch_models(provider)
                )
                
                # Update UI in main thread
                self.after(0, lambda: self._auth_test_complete(success, None, available_models if success else None))
                
            except Exception as e:
                self.after(0, lambda: self._auth_test_complete(False, str(e), None))
            finally:
                loop.close()
        
        # Run test in background thread
        threading.Thread(target=test_auth_async, daemon=True).start()
    
    def _auth_test_complete(self, success, error, available_models=None):
        """Handle authentication test completion"""
        self.test_button.configure(state=tk.NORMAL)
        
        if success:
            status_text = self.i18n['ai_auth_success']
            
            # Update model list if we got available models
            if available_models:
                self.model_combo.configure(values=available_models)
                # Select the first available model if current selection is not in the list
                current_model = self.selected_model.get()
                if current_model not in available_models:
                    self.selected_model.set(available_models[0] if available_models else "")
                status_text += f" ({len(available_models)} models found)"
                
            self.auth_status_label.configure(
                text=status_text,
                foreground="green"
            )
        else:
            error_msg = error or "Unknown error"
            if "api" in error_msg.lower() and "key" in error_msg.lower():
                error_msg = f"{error_msg}. Click 'Get API Key' for help."
            self.auth_status_label.configure(
                text=self.i18n['ai_auth_failed'].format(error_msg),
                foreground="red"
            )
    
    def _open_auth_url(self):
        """Open authentication URL for the selected provider"""
        provider_display = self.selected_provider.get()
        if not provider_display:
            messagebox.showwarning("Warning", "Please select a provider first")
            return
        
        # Map display name to internal name
        provider_map = {
            self.i18n['ai_provider_openai']: "OpenAI",
            self.i18n['ai_provider_gemini']: "Google Gemini",
            self.i18n['ai_provider_claude']: "Anthropic Claude",
            self.i18n['ai_provider_groq']: "Groq"
        }
        provider = provider_map.get(provider_display, provider_display)
        
        try:
            auth_url = ai_manager.get_auth_url(provider)
            if auth_url:
                webbrowser.open(auth_url)
            else:
                messagebox.showerror("Error", f"Authentication URL not available for {provider}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open authentication URL: {e}")
    
    def _show_auth_instructions(self):
        """Show detailed authentication instructions for the selected provider"""
        provider_display = self.selected_provider.get()
        if not provider_display:
            messagebox.showwarning("Warning", "Please select a provider first")
            return
        
        # Map display name to internal name
        provider_map = {
            self.i18n['ai_provider_openai']: "OpenAI",
            self.i18n['ai_provider_gemini']: "Google Gemini",
            self.i18n['ai_provider_claude']: "Anthropic Claude",
            self.i18n['ai_provider_groq']: "Groq"
        }
        provider = provider_map.get(provider_display, provider_display)
        
        try:
            instructions = ai_manager.get_auth_instructions(provider)
            
            # Create instructions dialog
            instructions_window = tk.Toplevel(self)
            instructions_window.title(f"{self.i18n['auth_instructions_title']} - {provider_display}")
            instructions_window.geometry("600x400")
            instructions_window.transient(self)
            instructions_window.grab_set()
            
            # Create text widget with scrollbar
            frame = ttk.Frame(instructions_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                font=("Consolas", 10),
                state="normal"
            )
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Insert instructions
            text_widget.insert("1.0", instructions)
            text_widget.configure(state="disabled")
            
            # Add close button
            button_frame = ttk.Frame(instructions_window)
            button_frame.pack(pady=(0, 10))
            
            ttk.Button(
                button_frame,
                text=self.i18n['ok_button'],
                command=instructions_window.destroy
            ).pack()
            
            # Center on parent
            instructions_window.update_idletasks()
            x = self.winfo_rootx() + (self.winfo_width() - instructions_window.winfo_width()) // 2
            y = self.winfo_rooty() + (self.winfo_height() - instructions_window.winfo_height()) // 2
            instructions_window.geometry(f"+{x}+{y}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show instructions: {e}")
    
    def _save_config(self):
        """Save AI configuration"""
        try:
            # Save main settings
            ai_config.set_text_correction_enabled(self.enable_ai.get())
            ai_config.set_correction_type(self.correction_type.get())
            ai_config.set_auto_apply_enabled(self.auto_apply.get())
            ai_config.set_show_changes(self.show_changes.get())
            ai_config.set_backup_original(self.backup_original.get())
            
            if self.enable_ai.get():
                provider_display = self.selected_provider.get()
                api_key = self.api_key.get().strip()
                model = self.selected_model.get()
                
                if provider_display and api_key and model:
                    # Map display name to internal name
                    provider_map = {
                        self.i18n['ai_provider_openai']: "OpenAI",
                        self.i18n['ai_provider_gemini']: "Google Gemini",
                        self.i18n['ai_provider_claude']: "Anthropic Claude",
                        self.i18n['ai_provider_groq']: "Groq"
                    }
                    provider = provider_map.get(provider_display, provider_display)
                    
                    # Save provider configuration
                    ai_config.set_api_key(provider, api_key)
                    ai_config.set_model(provider, model)
                    ai_config.set_provider_enabled(provider, True)
                    ai_config.set_active_provider(provider)
                    
                    # Synchronize AI manager active service if the service is registered
                    if provider in ai_manager.services:
                        ai_manager.set_active_service_sync(provider)
            
            # Save configuration to file
            ai_config.save_config()
            
            messagebox.showinfo("Success", self.i18n['ai_config_saved'])
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _update_cache_stats(self):
        """Update the cache statistics display"""
        try:
            stats = global_cache_manager.get_cache_stats()
            total_entries = stats.get('total_entries', 0)
            memory_mb = stats.get('memory_usage_mb', 0)
            recent_24h = stats.get('recent_24h', 0)
            
            stats_text = f"Cache: {total_entries} entries, {memory_mb:.1f} MB, {recent_24h} recent"
            self.cache_stats_label.configure(text=stats_text)
        except Exception as e:
            self.cache_stats_label.configure(text=f"Cache stats error: {e}")
    
    def _show_cache_stats(self):
        """Show detailed cache statistics in a dialog"""
        try:
            stats = global_cache_manager.get_cache_stats()
            recent_corrections = global_cache_manager.get_recent_corrections(5)
            
            # Create stats dialog
            stats_dialog = tk.Toplevel(self)
            stats_dialog.title("AI Text Correction Cache Statistics")
            stats_dialog.geometry("600x400")
            stats_dialog.resizable(True, True)
            stats_dialog.transient(self)
            stats_dialog.grab_set()
            
            # Main frame with scrollable text
            main_frame = ttk.Frame(stats_dialog, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create scrollable text widget
            text_widget = scrolledtext.ScrolledText(
                main_frame,
                wrap=tk.WORD,
                width=70,
                height=20,
                font=("Consolas", 9)
            )
            text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Build statistics text
            stats_text = []
            stats_text.append("=== AI Text Correction Cache Statistics ===\n")
            stats_text.append(f"Total cached corrections: {stats.get('total_entries', 0)}")
            stats_text.append(f"Memory usage: {stats.get('memory_usage_mb', 0):.2f} MB ({stats.get('memory_usage_chars', 0)} chars)")
            stats_text.append(f"Recent activity (24h): {stats.get('recent_24h', 0)} corrections")
            stats_text.append(f"Cache TTL: {stats.get('ttl_days', 30)} days")
            stats_text.append(f"Max cache size: {stats.get('max_size', 10000)} entries")
            stats_text.append(f"Cache file: {stats.get('cache_file', 'N/A')}")
            
            # Provider breakdown
            providers = stats.get('providers', {})
            if providers:
                stats_text.append(f"\n=== By Provider ===")
                for provider, count in providers.items():
                    stats_text.append(f"{provider}: {count} corrections")
            
            # Recent corrections
            if recent_corrections:
                stats_text.append(f"\n=== Recent Corrections (last 5) ===")
                for i, correction in enumerate(recent_corrections, 1):
                    stats_text.append(f"{i}. [{correction.get('provider', 'unknown')}] {correction.get('timestamp', '')}")
                    stats_text.append(f"   Original:  '{correction.get('original', '')}'")
                    stats_text.append(f"   Corrected: '{correction.get('corrected', '')}'")
                    stats_text.append(f"   Confidence: {correction.get('confidence', 0):.2f}")
                    stats_text.append("")
            
            # Insert text
            text_widget.insert(tk.END, "\n".join(stats_text))
            text_widget.configure(state=tk.DISABLED)  # Make read-only
            
            # Close button
            close_button = ttk.Button(
                main_frame,
                text="Close",
                command=stats_dialog.destroy
            )
            close_button.pack(pady=(5, 0))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show cache statistics: {e}")
    
    def _clear_all_cache(self):
        """Clear all cached text corrections"""
        try:
            result = messagebox.askyesno(
                "Confirm Clear Cache",
                "Are you sure you want to clear all cached text corrections?\n\nThis will remove all stored corrections from all providers."
            )
            
            if result:
                cleared_count = global_cache_manager.clear_all()
                messagebox.showinfo(
                    "Cache Cleared",
                    f"Successfully cleared {cleared_count} cached text corrections."
                )
                self._update_cache_stats()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear cache: {e}")
    
    def _clear_provider_cache(self):
        """Clear cached corrections for the currently selected provider"""
        try:
            provider_display = self.selected_provider.get()
            if not provider_display:
                messagebox.showwarning("Warning", "Please select a provider first")
                return
            
            # Map display name to internal name
            provider_map = {
                self.i18n['ai_provider_openai']: "OpenAI",
                self.i18n['ai_provider_gemini']: "Google Gemini",
                self.i18n['ai_provider_claude']: "Anthropic Claude",
                self.i18n['ai_provider_groq']: "Groq"
            }
            provider = provider_map.get(provider_display, provider_display)
            
            result = messagebox.askyesno(
                "Confirm Clear Provider Cache",
                f"Are you sure you want to clear cached corrections for {provider}?\n\nThis will only remove corrections made by this provider."
            )
            
            if result:
                cleared_count = global_cache_manager.clear_by_provider(provider)
                messagebox.showinfo(
                    "Provider Cache Cleared",
                    f"Successfully cleared {cleared_count} cached corrections for {provider}."
                )
                self._update_cache_stats()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear provider cache: {e}")


class AboutDialog(tk.Toplevel):
    """About dialog with developer info and support links"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.i18n = parent.i18n
        
        self.title(self.i18n['about_title'])
        self.geometry("500x650")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.winfo_screenheight() // 2) - (650 // 2)
        self.geometry(f"500x650+{x}+{y}")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the about dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # App icon/title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 20))
        
        app_label = ttk.Label(title_frame, text=self.i18n['about_app_name'], 
                             font=("Arial", 16, "bold"))
        app_label.pack()
        
        version_label = ttk.Label(title_frame, text=self.i18n['about_version'], 
                                 font=("Arial", 10))
        version_label.pack(pady=(5, 0))
        
        # Description
        desc_label = ttk.Label(main_frame, text=self.i18n['about_description'], 
                              wraplength=450, justify="center")
        desc_label.pack(pady=(0, 10))
        
        # Attribution to original project
        attribution_label = ttk.Label(main_frame, 
                                     text="Inspired by JuniverseCoder's MinerU2PPT",
                                     font=("Arial", 8, "italic"),
                                     foreground="gray")
        attribution_label.pack(pady=(0, 15))
        
        # Developer info
        dev_frame = ttk.LabelFrame(main_frame, text=self.i18n['about_developer'], padding="15")
        dev_frame.pack(fill="x", pady=(0, 20))
        
        # GitHub photo (placeholder or actual)
        photo_frame = ttk.Frame(dev_frame)
        photo_frame.pack(pady=(0, 15))
        
        try:
            # Try to load GitHub photo
            self._load_github_photo(photo_frame)
        except Exception as e:
            # Fallback to text if photo fails
            photo_label = ttk.Label(photo_frame, text="👤 Arlinamid (Rózsavölgyi János)", 
                                   font=("Arial", 14, "bold"))
            photo_label.pack()
        
        # GitHub info
        github_frame = ttk.Frame(dev_frame)
        github_frame.pack(fill="x", pady=(0, 10))
        
        github_label = ttk.Label(github_frame, text=self.i18n['about_github'])
        github_label.pack(side="left")
        
        github_link = ttk.Label(github_frame, text="arlinamid", 
                               foreground="blue", cursor="hand2")
        github_link.pack(side="left", padx=(5, 0))
        github_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/arlinamid"))
        
        # GitHub button
        github_btn = ttk.Button(dev_frame, text=self.i18n['github_button'],
                               command=lambda: webbrowser.open("https://github.com/arlinamid"))
        github_btn.pack(pady=(0, 10))
        
        # Support section
        support_frame = ttk.LabelFrame(main_frame, text=self.i18n['about_support'], padding="15")
        support_frame.pack(fill="x", pady=(0, 20))
        
        support_desc = ttk.Label(support_frame, 
                                text="If this tool helped you, consider buying me a coffee! ☕",
                                wraplength=450, justify="center")
        support_desc.pack(pady=(0, 15))
        
        # Buy me a coffee button
        coffee_btn = ttk.Button(support_frame, text=self.i18n['support_button'],
                               command=lambda: webbrowser.open("https://buymeacoffee.com/arlinamid"))
        coffee_btn.pack()
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", command=self.destroy)
        close_btn.pack(pady=(20, 0))
        
    def _load_github_photo(self, parent_frame):
        """Try to load GitHub profile photo"""
        try:
            # Download GitHub avatar
            url = "https://avatars.githubusercontent.com/arlinamid"
            urllib.request.urlretrieve(url, "temp_avatar.png")
            
            # Load and resize image
            image = Image.open("temp_avatar.png")
            image = image.resize((100, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Create label with photo
            photo_label = ttk.Label(parent_frame, image=photo)
            photo_label.image = photo  # Keep a reference
            photo_label.pack()
            
            # Clean up temp file
            try:
                os.remove("temp_avatar.png")
            except:
                pass
                
        except Exception:
            # Fallback to emoji
            photo_label = ttk.Label(parent_frame, text="👤", font=("Arial", 48))
            photo_label.pack()


class SupportDialog(tk.Toplevel):
    """Support dialog shown after conversion completion"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.i18n = parent.i18n
        
        self.title(self.i18n['support_dialog_title'])
        self.geometry("400x300")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.winfo_screenheight() // 2) - (300 // 2)
        self.geometry(f"400x300+{x}+{y}")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the support dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Success icon
        success_label = ttk.Label(main_frame, text="✅", font=("Arial", 48))
        success_label.pack(pady=(0, 20))
        
        # Message
        message_label = ttk.Label(main_frame, text=self.i18n['support_message'],
                                 wraplength=350, justify="center")
        message_label.pack(pady=(0, 30))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack()
        
        # Buy me a coffee button
        coffee_btn = ttk.Button(buttons_frame, text=self.i18n['support_button'],
                               command=self._open_coffee_link)
        coffee_btn.pack(side="left", padx=(0, 10))
        
        # GitHub button
        github_btn = ttk.Button(buttons_frame, text=self.i18n['github_button'],
                               command=self._open_github_link)
        github_btn.pack(side="left", padx=(10, 0))
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", command=self.destroy)
        close_btn.pack(pady=(30, 0))
        
    def _open_coffee_link(self):
        """Open Buy Me a Coffee link"""
        webbrowser.open("https://buymeacoffee.com/arlinamid")
        self.destroy()
        
    def _open_github_link(self):
        """Open GitHub link"""
        webbrowser.open("https://github.com/arlinamid")
        self.destroy()


class ImageExtractorDialog(tk.Toplevel):
    """Dialog for extracting images from PDF using MinerU JSON bounding boxes."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.i18n = parent.i18n
        self.title(self.i18n['image_downloader_title'])
        self.geometry("500x450")
        self.resizable(True, True)
        self.minsize(500, 400)

        self.cancel_event = threading.Event()
        self.extraction_thread = None

        self.transient(parent)
        self.grab_set()
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self._create_widgets()
        self.focus_set()

    def _create_widgets(self):
        main_frame = tk.Frame(self, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # -- File info ------------------------------------------------
        info_frame = tk.LabelFrame(main_frame, text="Files", padx=10, pady=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        for label, var in [
            ("JSON File:", self.parent.json_path),
            ("Original PDF/Image:", self.parent.input_path),
            (self.i18n['work_folder_label'], self.parent.work_folder_path),
        ]:
            tk.Label(info_frame, text=label, font=("Arial", 9, "bold")).pack(anchor=tk.W)
            value = var.get() or "Not selected"
            tk.Label(info_frame, text=value, fg="blue", wraplength=440).pack(anchor=tk.W, padx=10, pady=(0, 4))

        # -- Options --------------------------------------------------
        options_frame = tk.LabelFrame(main_frame, text="Options", padx=10, pady=5)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Checkbutton(options_frame, text=self.i18n['overwrite_images_checkbox'],
                        variable=self.parent.overwrite_images).pack(anchor=tk.W)

        # -- Progress -------------------------------------------------
        progress_frame = tk.LabelFrame(main_frame, text="Progress", padx=10, pady=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.status_label = tk.Label(progress_frame, text="Ready", fg="green")
        self.status_label.pack(pady=(0, 5))
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # -- Buttons --------------------------------------------------
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)

        self.start_button = tk.Button(
            btn_frame, text=self.i18n['download_images_button'],
            command=self._start_extraction,
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=20)
        self.start_button.pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text=self.i18n['cancel_button'],
                  command=self._cancel, padx=20).pack(side=tk.RIGHT, padx=5)

    # -- helpers ------------------------------------------------------
    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def _cancel(self):
        if self.extraction_thread and self.extraction_thread.is_alive():
            self._log("Cancelling...")
            self.cancel_event.set()
            self.status_label.config(text="Cancelling...", fg="orange")
        else:
            self.destroy()

    # -- extraction ---------------------------------------------------
    def _start_extraction(self):
        json_path = self.parent.json_path.get()
        original_path = self.parent.input_path.get()
        work_folder = self.parent.work_folder_path.get()

        if not json_path:
            messagebox.showerror("Error", self.i18n['error_no_json_for_download'])
            return
        if not original_path or not os.path.exists(original_path):
            messagebox.showerror("Error", "Please select the original PDF/image file.")
            return
        if not work_folder:
            messagebox.showerror("Error", self.i18n['error_no_work_folder'])
            return

        self.start_button.config(state=tk.DISABLED, text="Extracting...")
        self.status_label.config(text="Extracting from PDF...", fg="orange")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.cancel_event.clear()

        self.extraction_thread = threading.Thread(target=self._run_extraction, daemon=True)
        self.extraction_thread.start()

    def _run_extraction(self):
        try:
            if self.cancel_event.is_set():
                self._on_cancelled()
                return

            json_path = self.parent.json_path.get()
            original_path = self.parent.input_path.get()
            work_folder = self.parent.work_folder_path.get()
            overwrite = self.parent.overwrite_images.get()

            self._log("Starting image extraction from PDF...")
            self._log(f"JSON: {os.path.basename(json_path)}")
            self._log(f"PDF:  {os.path.basename(original_path)}")
            self._log(f"Output: {work_folder}")

            extractor = ImageExtractor(work_folder)
            result = extractor.extract_images_from_pdf(
                json_path, original_path, overwrite, cancel_event=self.cancel_event)

            if "error" in result:
                self._log(f"Error: {result['error']}")
            else:
                self._log("Extraction completed!")
                self._log(f"Total images found: {result['total_images']}")
                self._log(f"Extracted: {result['extracted']}")
                self._log(f"Skipped: {result['skipped']}")
                self._log(f"Errors: {result['errors']}")

            def done():
                self.start_button.config(state=tk.NORMAL, text=self.i18n['download_images_button'])
                if "error" not in result and result['total_images'] > 0:
                    self.status_label.config(text="Extraction completed!", fg="green")
                    messagebox.showinfo(
                        self.i18n['download_complete_title'],
                        f"Extracted: {result['extracted']} images\n"
                        f"Skipped: {result['skipped']} images\n"
                        f"Errors: {result['errors']}\n"
                        f"Folder: {result['work_folder']}")
                elif result['total_images'] == 0 and "error" not in result:
                    self.status_label.config(text="No images found in JSON", fg="orange")
                else:
                    self.status_label.config(text="Extraction failed", fg="red")

            self.after(0, done)

        except Exception as e:
            if self.cancel_event.is_set():
                self._on_cancelled()
                return

            def err():
                self.start_button.config(state=tk.NORMAL, text=self.i18n['download_images_button'])
                self.status_label.config(text="Extraction failed", fg="red")
                self._log(f"Error: {e}")
                messagebox.showerror("Error", f"Extraction failed: {e}")
            self.after(0, err)

    def _on_cancelled(self):
        def reset():
            self.start_button.config(state=tk.NORMAL, text=self.i18n['download_images_button'])
            self.status_label.config(text="Cancelled", fg="red")
            self._log("Cancelled by user.")
        self.after(0, reset)


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.translator = get_translator()
        self.i18n = self.translator  # For backward compatibility with existing code
        self.title(self.i18n['app_title'])
        self.geometry("700x600")
        self.debug_folder_path = os.path.join(os.getcwd(), "tmp")
        self.input_path, self.json_path, self.output_path = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.work_folder_path = tk.StringVar(value=os.path.join(os.getcwd(), "minerU_images"))
        self.remove_watermark, self.generate_debug = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.batch_mode = tk.BooleanVar(value=False)
        
        # Image extractor option
        self.overwrite_images = tk.BooleanVar(value=False)
        
        # Page selection variables
        self.page_selection_mode = tk.StringVar(value="all")  # "all", "single", "range"
        self.single_page_num = tk.StringVar(value="1")
        self.page_range_from = tk.StringVar(value="1")
        self.page_range_to = tk.StringVar(value="1")
        
        # Language selection
        current_lang_code = self.translator.get_current_language()
        language_names = {'en': 'English', 'hu': 'Magyar', 'zh': '中文'}
        current_display_name = language_names.get(current_lang_code, current_lang_code.upper())
        self.current_language = tk.StringVar(value=current_display_name)
        
        self.task_list = []
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self._create_widgets()
        self._poll_log_queue()
    
    def _browse_work_folder(self):
        """Browse and select work folder for image downloads."""
        folder = filedialog.askdirectory(
            title="Select Work Folder",
            initialdir=self.work_folder_path.get() or os.getcwd()
        )
        if folder:
            self.work_folder_path.set(folder)
    
    def _show_work_folder_help(self):
        """Show work folder help dialog."""
        messagebox.showinfo(
            self.i18n['work_folder_help_title'], 
            self.i18n['work_folder_help_text']
        )
    
    def _open_image_downloader(self):
        """Open the image extractor dialog."""
        ImageExtractorDialog(self)
    
    def _get_language_options(self):
        """Get available language options for the combobox."""
        language_names = {
            'en': 'English',
            'hu': 'Magyar',
            'zh': '中文'
        }
        available_langs = self.translator.get_available_languages()
        return [language_names.get(lang, lang.upper()) for lang in sorted(available_langs)]
    
    def _get_language_code(self, display_name):
        """Convert display name back to language code."""
        name_to_code = {
            'English': 'en',
            'Magyar': 'hu',
            '中文': 'zh'
        }
        return name_to_code.get(display_name, display_name.lower())
    
    def _on_language_change(self, event=None):
        """Handle language selection change."""
        display_name = self.current_language.get()
        lang_code = self._get_language_code(display_name)
        
        if lang_code != self.translator.get_current_language():
            self.translator.set_language(lang_code)
            self._update_ui_text()
    
    def _update_ui_text(self):
        """Update all UI text elements when language changes."""
        # Update window title
        self.title(self.i18n['app_title'])
        
        # Update language label
        self.language_label.config(text=self.i18n['language_label'])
        
        # Update about button
        self.about_button.config(text=self.i18n['about_button'])
        
        # Update mode switch button
        if self.batch_mode.get():
            self.mode_switch_button.config(text=self.i18n['single_mode_button'])
        else:
            self.mode_switch_button.config(text=self.i18n['batch_mode_button'])
        
        # Destroy and recreate content frame with new language
        if hasattr(self, 'content_frame'):
            self.content_frame.destroy()
        
        # Recreate content widgets with new language
        self._create_content_widgets()
        
        # Update mode display to match current mode
        current_batch_mode = self.batch_mode.get()
        if current_batch_mode:
            self.batch_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
            self.single_mode_frame.grid_remove()
            self.options_frame.grid_remove()
        else:
            self.single_mode_frame.grid()
            self.batch_frame.grid_remove()
            self.options_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="w")

    def _create_widgets(self):
        self.main_frame = tk.Frame(self, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Language and mode selection frame (persistent)
        self.top_controls_frame = tk.Frame(self.main_frame)
        self.top_controls_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        self.top_controls_frame.grid_columnconfigure(1, weight=1)
        
        # Language selection
        language_frame = tk.Frame(self.top_controls_frame)
        language_frame.grid(row=0, column=0, sticky="w")
        self.language_label = tk.Label(language_frame, text=self.i18n['language_label'])
        self.language_label.pack(side=tk.LEFT)
        
        self.language_combobox = ttk.Combobox(language_frame, textvariable=self.current_language, 
                                            values=self._get_language_options(), 
                                            state="readonly", width=12)
        self.language_combobox.pack(side=tk.LEFT, padx=(5, 0))
        self.language_combobox.bind('<<ComboboxSelected>>', self._on_language_change)
        
        # About button
        self.about_button = tk.Button(self.top_controls_frame, text=self.i18n['about_button'], 
                                     command=self._show_about_dialog, font=("Arial", 8))
        self.about_button.grid(row=0, column=1, sticky="e", padx=(0, 10))
        
        # Mode switch button
        self.mode_switch_button = tk.Button(self.top_controls_frame, text=self.i18n['batch_mode_button'], command=self._toggle_batch_mode)
        self.mode_switch_button.grid(row=0, column=2, sticky="e")
        
        # Create content widgets
        self._create_content_widgets()
    
    def _show_about_dialog(self):
        """Show the About dialog"""
        AboutDialog(self)
    
    def _show_support_dialog(self):
        """Show the Support dialog after successful conversion"""
        SupportDialog(self)
    
    def _create_content_widgets(self):
        """Create the main content widgets (everything except top controls)."""
        # Content frame for all widgets below the top controls
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(3, weight=1)

        # --- Single Mode Frame ---
        self.single_mode_frame = tk.Frame(self.content_frame)
        self.single_mode_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.single_mode_frame.grid_columnconfigure(1, weight=1)
        # (Content of single mode frame...)
        tk.Label(self.single_mode_frame, text=self.i18n['input_file_label']).grid(row=0, column=0, sticky="w", pady=2)
        input_entry = tk.Entry(self.single_mode_frame, textvariable=self.input_path, state="readonly")
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        input_entry.drop_target_register(DND_FILES); input_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.input_path))
        tk.Button(self.single_mode_frame, text=self.i18n['browse_button'], command=self._browse_input).grid(row=0, column=2, sticky="w")

        tk.Label(self.single_mode_frame, text=self.i18n['json_file_label']).grid(row=1, column=0, sticky="w", pady=2)
        json_entry = tk.Entry(self.single_mode_frame, textvariable=self.json_path, state="readonly")
        json_entry.grid(row=1, column=1, sticky="ew", padx=5)
        json_entry.drop_target_register(DND_FILES); json_entry.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, self.json_path))
        json_buttons_frame = tk.Frame(self.single_mode_frame)
        json_buttons_frame.grid(row=1, column=2, sticky="w")
        tk.Button(json_buttons_frame, text=self.i18n['browse_button'], command=self._browse_json).pack(side=tk.LEFT)
        tk.Button(json_buttons_frame, text=self.i18n['help_button'], command=self._show_json_help, width=2).pack(side=tk.LEFT)

        tk.Label(self.single_mode_frame, text=self.i18n['output_file_label']).grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(self.single_mode_frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        tk.Button(self.single_mode_frame, text=self.i18n['save_as_button'], command=self._save_pptx).grid(row=2, column=2, sticky="w")

        # Work folder selection
        tk.Label(self.single_mode_frame, text=self.i18n['work_folder_label']).grid(row=3, column=0, sticky="w", pady=2)
        work_folder_entry = tk.Entry(self.single_mode_frame, textvariable=self.work_folder_path, state="readonly")
        work_folder_entry.grid(row=3, column=1, sticky="ew", padx=5)
        work_folder_buttons_frame = tk.Frame(self.single_mode_frame)
        work_folder_buttons_frame.grid(row=3, column=2, sticky="w")
        tk.Button(work_folder_buttons_frame, text=self.i18n['work_folder_button'], command=self._browse_work_folder).pack(side=tk.LEFT)
        tk.Button(work_folder_buttons_frame, text=self.i18n['help_button'], command=self._show_work_folder_help, width=2).pack(side=tk.LEFT)

        # Page Selection Frame
        page_frame = tk.LabelFrame(self.single_mode_frame, text=self.i18n.get('page_selection_label', 'Page Selection'), padx=5, pady=5)
        page_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        page_frame.grid_columnconfigure(1, weight=1)
        
        # All Pages option
        tk.Radiobutton(page_frame, text=self.i18n.get('all_pages_option', 'All Pages'), 
                      variable=self.page_selection_mode, value="all",
                      command=self._on_page_selection_change).grid(row=0, column=0, sticky="w", padx=5)
        
        # Single Page option
        single_frame = tk.Frame(page_frame)
        single_frame.grid(row=1, column=0, columnspan=3, sticky="w", padx=5)
        tk.Radiobutton(single_frame, text=self.i18n.get('single_page_option', 'Single Page:'), 
                      variable=self.page_selection_mode, value="single",
                      command=self._on_page_selection_change).pack(side=tk.LEFT)
        self.single_page_entry = tk.Entry(single_frame, textvariable=self.single_page_num, width=5)
        self.single_page_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Page Range option
        range_frame = tk.Frame(page_frame)
        range_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5)
        tk.Radiobutton(range_frame, text=self.i18n.get('page_range_option', 'Page Range:'), 
                      variable=self.page_selection_mode, value="range",
                      command=self._on_page_selection_change).pack(side=tk.LEFT)
        tk.Label(range_frame, text=self.i18n.get('from_label', 'From:')).pack(side=tk.LEFT, padx=(10, 0))
        self.page_from_entry = tk.Entry(range_frame, textvariable=self.page_range_from, width=5)
        self.page_from_entry.pack(side=tk.LEFT, padx=(5, 0))
        tk.Label(range_frame, text=self.i18n.get('to_label', 'To:')).pack(side=tk.LEFT, padx=(10, 0))
        self.page_to_entry = tk.Entry(range_frame, textvariable=self.page_range_to, width=5)
        self.page_to_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Initially disable page selection entries
        self._on_page_selection_change()

        # --- Batch Mode Frame ---
        self.batch_frame = tk.Frame(self.content_frame)
        self.batch_frame.grid_columnconfigure(0, weight=1); self.batch_frame.grid_rowconfigure(0, weight=1)
        # (Content of batch mode frame...)
        task_list_frame = tk.LabelFrame(self.batch_frame, text=self.i18n['task_list_label'], padx=5, pady=5)
        task_list_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=5)
        task_list_frame.grid_columnconfigure(0, weight=1); task_list_frame.grid_rowconfigure(0, weight=1)
        self.task_listbox = tk.Listbox(task_list_frame, height=8)
        self.task_listbox.grid(row=0, column=0, sticky="nsew")
        task_scrollbar = tk.Scrollbar(task_list_frame, orient="vertical", command=self.task_listbox.yview)
        task_scrollbar.grid(row=0, column=1, sticky="ns"); self.task_listbox.config(yscrollcommand=task_scrollbar.set)
        batch_buttons_frame = tk.Frame(self.batch_frame)
        batch_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(5,0))
        tk.Button(batch_buttons_frame, text=self.i18n['add_task_button'], command=self._add_task).pack(side=tk.LEFT, padx=5)
        tk.Button(batch_buttons_frame, text=self.i18n['delete_task_button'], command=self._delete_task).pack(side=tk.LEFT, padx=5)

        # --- Options (will be managed dynamically) ---
        self.options_frame = tk.Frame(self.content_frame)
        self.options_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="w")
        self.remove_watermark_checkbox = tk.Checkbutton(self.options_frame, text=self.i18n['remove_watermark_checkbox'], variable=self.remove_watermark)
        self.remove_watermark_checkbox.pack(side=tk.LEFT, padx=5)
        self.debug_images_checkbox = tk.Checkbutton(self.options_frame, text=self.i18n['debug_images_checkbox'], variable=self.generate_debug, command=self._toggle_debug_button_visibility)
        self.debug_images_checkbox.pack(side=tk.LEFT, padx=5)

        # --- Actions and Log ---
        action_frame = tk.Frame(self.content_frame)
        action_frame.grid(row=2, column=0, columnspan=3, pady=10)
        action_frame.grid_columnconfigure(0, weight=1)
        button_container = tk.Frame(action_frame)
        button_container.grid(row=0, column=0)
        self.start_button = tk.Button(button_container, text=self.i18n['start_button'], command=self.start_conversion_thread)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.output_button = tk.Button(button_container, text=self.i18n['output_folder_button'], command=self._open_output_folder, state="disabled")
        self.output_button.pack(side=tk.LEFT, padx=10)
        self.debug_button = tk.Button(button_container, text=self.i18n['debug_folder_button'], command=self._open_debug_folder, state="disabled")
        self.ai_settings_button = tk.Button(button_container, text=self.i18n['ai_settings_button'], command=self._open_ai_settings)
        self.ai_settings_button.pack(side=tk.LEFT, padx=10)
        self.image_downloader_button = tk.Button(button_container, text=self.i18n['image_downloader_button'], command=self._open_image_downloader)
        self.image_downloader_button.pack(side=tk.LEFT, padx=10)

        log_frame = tk.LabelFrame(self.content_frame, text=self.i18n['log_label'], padx=5, pady=5)
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1); log_frame.grid_columnconfigure(0, weight=1)
        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled", wrap=tk.WORD, height=10)
        self.log_area.grid(row=0, column=0, sticky="nsew")

        self._toggle_batch_mode() # Set initial state to single mode
        self._toggle_batch_mode()

    def _toggle_batch_mode(self):
        is_batch = not self.batch_mode.get()
        self.batch_mode.set(is_batch)
        if is_batch:
            self.single_mode_frame.grid_remove()
            self.batch_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
            self.mode_switch_button.config(text=self.i18n['single_mode_button'])
            self.start_button.config(text=self.i18n['start_batch_button'])
            # Hide options not relevant to batch mode
            self.options_frame.grid_remove()
            self.debug_button.pack_forget()
        else:
            self.batch_frame.grid_remove()
            self.single_mode_frame.grid()
            self.mode_switch_button.config(text=self.i18n['batch_mode_button'])
            self.start_button.config(text=self.i18n['start_button'])
            # Show options for single mode
            self.options_frame.grid()
            self._toggle_debug_button_visibility()

    def _add_task(self):
        dialog = AddTaskDialog(self)
        if dialog.result:
            task = dialog.result
            self.task_list.append(task)
            suffix = "" if task['remove_watermark'] else " (Keep WM)"
            self.task_listbox.insert(tk.END, f"IN: {os.path.basename(task['input'])} -> OUT: {os.path.basename(task['output'])}{suffix}")

    def _delete_task(self):
        selected_indices = self.task_listbox.curselection()
        if not selected_indices: return
        for index in sorted(selected_indices, reverse=True):
            self.task_listbox.delete(index)
            del self.task_list[index]

    def _show_json_help(self):
        if messagebox.askokcancel(self.i18n['json_help_title'], self.i18n['json_help_text']):
            webbrowser.open_new("https://mineru.net/OpenSourceTools/Extractor")

    def _toggle_debug_button_visibility(self):
        if self.generate_debug.get() and not self.batch_mode.get():
            self.debug_button.pack(side=tk.LEFT, padx=10)
        else:
            self.debug_button.pack_forget()
    
    def _on_page_selection_change(self):
        """Handle page selection mode changes"""
        mode = self.page_selection_mode.get()
        
        # Enable/disable entry fields based on selection
        if mode == "single":
            self.single_page_entry.config(state="normal")
            self.page_from_entry.config(state="disabled")
            self.page_to_entry.config(state="disabled")
        elif mode == "range":
            self.single_page_entry.config(state="disabled")
            self.page_from_entry.config(state="normal")
            self.page_to_entry.config(state="normal")
        else:  # "all"
            self.single_page_entry.config(state="disabled")
            self.page_from_entry.config(state="disabled")
            self.page_to_entry.config(state="disabled")
    
    def _get_page_selection_info(self):
        """Get and validate page selection information for main GUI"""
        mode = self.page_selection_mode.get()
        
        if mode == "all":
            return {"mode": "all"}
        elif mode == "single":
            try:
                page_num = int(self.single_page_num.get().strip())
                if page_num < 1:
                    raise ValueError("Page number must be positive")
                return {"mode": "single", "page": page_num}
            except ValueError:
                messagebox.showerror(self.i18n['error_title'], 
                                   "Please enter a valid page number (positive integer)")
                return None
        elif mode == "range":
            try:
                from_page = int(self.page_range_from.get().strip())
                to_page = int(self.page_range_to.get().strip())
                if from_page < 1 or to_page < 1:
                    raise ValueError("Page numbers must be positive")
                if from_page > to_page:
                    raise ValueError("'From' page must be less than or equal to 'To' page")
                return {"mode": "range", "from": from_page, "to": to_page}
            except ValueError as e:
                messagebox.showerror(self.i18n['error_title'], 
                                   f"Please enter valid page numbers: {str(e)}")
                return None
        
        return {"mode": "all"}  # Fallback

    def _open_output_folder(self):
        output_file = self.output_path.get()
        if self.batch_mode.get() and self.task_list:
             output_file = self.task_list[-1]['output']
        if not output_file:
            messagebox.showinfo(self.i18n['info_title'], self.i18n['info_no_output']); return
        output_dir = os.path.dirname(output_file)
        if os.path.exists(output_dir): os.startfile(output_dir)
        else: messagebox.showerror(self.i18n['error_title'], self.i18n['error_dir_not_found'].format(output_dir))

    def _open_debug_folder(self):
        if os.path.exists(self.debug_folder_path): os.startfile(self.debug_folder_path)
        else: messagebox.showinfo(self.i18n['info_title'], self.i18n['info_debug_not_found'])
    
    def _open_ai_settings(self):
        """Open AI settings configuration dialog"""
        try:
            AISettingsDialog(self)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open AI settings: {e}")

    def _set_default_output_path(self, in_path):
        if not self.output_path.get(): self.output_path.set(os.path.splitext(in_path)[0] + ".pptx")

    def _on_drop(self, event, var):
        filepath = event.data.strip('{}')
        var.set(filepath)
        if var == self.input_path: self._set_default_output_path(filepath)

    def _browse_input(self):
        filetypes = [("Supported Files", "*.pdf *.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path: self.input_path.set(path); self._set_default_output_path(path)

    def _browse_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if path: self.json_path.set(path)

    def _save_pptx(self):
        path = filedialog.asksaveasfilename(defaultextension=".pptx", filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")])
        if path: self.output_path.set(path)

    def _poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get_nowait()
                self.log_area.config(state="normal"); self.log_area.insert(tk.END, record); self.log_area.see(tk.END); self.log_area.config(state="disabled")
            except queue.Empty: break
        self.after(100, self._poll_log_queue)

    def start_conversion_thread(self):
        if self.batch_mode.get():
            if not self.task_list:
                messagebox.showerror(self.i18n['error_title'], self.i18n['error_no_tasks'])
                return
            target_func, args = self._run_batch_conversion, ()
        else:
            input_file, json_f = self.input_path.get(), self.json_path.get()
            if not self.output_path.get() and input_file: self._set_default_output_path(input_file)
            output = self.output_path.get()
            if not all([input_file, json_f, output]):
                messagebox.showerror(self.i18n['error_title'], self.i18n['error_all_paths']); return
            
            # Validate page selection before starting thread
            page_selection = self._get_page_selection_info()
            if page_selection is None:
                return  # Invalid page selection, error already shown
                
            target_func, args = self._run_single_conversion, (json_f, input_file, output)

        self.start_button.config(state="disabled", text=self.i18n['converting_button'])
        self.output_button.config(state="disabled")
        if not self.batch_mode.get(): self.debug_button.config(state="disabled")
        self.log_area.config(state="normal"); self.log_area.delete(1.0, tk.END); self.log_area.config(state="disabled")

        threading.Thread(target=self._run_conversion_wrapper, args=(target_func, args), daemon=True).start()

    def _run_conversion_wrapper(self, conversion_func, args):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.queue_handler, self.queue_handler
        success = False
        try:
            conversion_func(*args)
            success = True
        except Exception as e:
            self.log_queue.put(self.i18n['log_error'].format(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.after(0, self._finalize_gui, success)

    def _run_single_conversion(self, json_path, input_path, output_path):
        page_selection = self._get_page_selection_info()
        if page_selection is None:
            return  # Invalid page selection, error already shown
        
        args = (json_path, input_path, output_path, self.remove_watermark.get(), self.generate_debug.get(), ai_config.is_text_correction_enabled(), page_selection)
        convert_mineru_to_ppt(*args)
        self.log_queue.put(self.i18n['log_success'])

    def _run_batch_conversion(self):
        self.log_queue.put(self.i18n['log_batch_start'])
        total_tasks = len(self.task_list)
        for i, task in enumerate(self.task_list):
            self.log_queue.put(self.i18n['log_task_start'].format(i + 1, total_tasks, os.path.basename(task['input'])))
            try:
                # Debug images are disabled for batch mode
                # Use page selection from task, or default to all pages
                page_selection = task.get('page_selection', {"mode": "all"})
                args = (task['json'], task['input'], task['output'], task['remove_watermark'], False, ai_config.is_text_correction_enabled(), page_selection)
                convert_mineru_to_ppt(*args)
                self.log_queue.put(self.i18n['log_task_complete'].format(os.path.basename(task['input'])))
            except Exception as e:
                self.log_queue.put(self.i18n['log_error'].format(e))
        self.log_queue.put(self.i18n['log_batch_complete'])

    def _finalize_gui(self, success):
        start_text = self.i18n['start_batch_button'] if self.batch_mode.get() else self.i18n['start_button']
        self.start_button.config(state="normal", text=start_text)
        if success:
            self.output_button.config(state="normal")
            if self.generate_debug.get() and not self.batch_mode.get():
                 self.debug_button.config(state="normal")
        messagebox.showinfo(self.i18n['complete_title'], self.i18n['msg_conversion_complete'])
        
        # Show support dialog after successful conversion
        if success:
            self.after(500, self._show_support_dialog)  # Show after a short delay

if __name__ == "__main__":
    app = App()
    app.mainloop()
