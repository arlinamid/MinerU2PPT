"""
Configuration Management for AI Services in MinerU2PPT

Handles storage and retrieval of:
- API keys for different AI providers
- Model selections
- Text correction preferences
- Authentication status
"""

import json
import os
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AIConfig:
    """Configuration manager for AI services"""
    
    def __init__(self, config_file: str = "ai_config.json"):
        # Determine base path for config file
        if getattr(sys, 'frozen', False):
            # If running as compiled executable, use the executable's directory
            base_path = os.path.dirname(sys.executable)
        else:
            # If running from source, use the project root
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.config_file = Path(base_path) / config_file
        self.config_data: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        # Return default configuration
        return {
            "providers": {
                "OpenAI": {
                    "api_key": "",
                    "model": "gpt-3.5-turbo",
                    "enabled": False
                },
                "Google Gemini": {
                    "api_key": "",
                    "model": "gemini-pro",
                    "enabled": False
                },
                "Anthropic Claude": {
                    "api_key": "",
                    "model": "claude-3-sonnet-20240229",
                    "enabled": False
                },
                "Groq": {
                    "api_key": "",
                    "model": "llama2-70b-4096",
                    "enabled": False
                }
            },
            "settings": {
                "active_provider": "",
                "correction_type": "grammar_spelling",  # grammar_spelling, formatting, both
                "enable_text_correction": True,
                "auto_apply_corrections": False,
                "show_correction_changes": True,
                "backup_original_text": True
            },
            "last_used": {
                "provider": "",
                "model": ""
            }
        }
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider"""
        return self.config_data.get("providers", {}).get(provider, {})
    
    def set_provider_config(self, provider: str, config: Dict[str, Any]):
        """Set configuration for a specific provider"""
        if "providers" not in self.config_data:
            self.config_data["providers"] = {}
        
        if provider not in self.config_data["providers"]:
            self.config_data["providers"][provider] = {}
        
        self.config_data["providers"][provider].update(config)
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider"""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("api_key", "")
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider"""
        self.set_provider_config(provider, {"api_key": api_key})
    
    def get_model(self, provider: str) -> str:
        """Get selected model for a provider"""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("model", "")
    
    def set_model(self, provider: str, model: str):
        """Set selected model for a provider"""
        self.set_provider_config(provider, {"model": model})
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled"""
        provider_config = self.get_provider_config(provider)
        return provider_config.get("enabled", False)
    
    def set_provider_enabled(self, provider: str, enabled: bool):
        """Enable/disable a provider"""
        self.set_provider_config(provider, {"enabled": enabled})
    
    def get_active_provider(self) -> str:
        """Get the currently active provider"""
        return self.config_data.get("settings", {}).get("active_provider", "")
    
    def set_active_provider(self, provider: str):
        """Set the active provider"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["active_provider"] = provider
    
    def get_correction_type(self) -> str:
        """Get the correction type setting"""
        return self.config_data.get("settings", {}).get("correction_type", "grammar_spelling")
    
    def set_correction_type(self, correction_type: str):
        """Set the correction type"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["correction_type"] = correction_type
    
    def is_text_correction_enabled(self) -> bool:
        """Check if text correction is enabled globally"""
        return self.config_data.get("settings", {}).get("enable_text_correction", True)
    
    def set_text_correction_enabled(self, enabled: bool):
        """Enable/disable text correction globally"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["enable_text_correction"] = enabled
    
    def is_auto_apply_enabled(self) -> bool:
        """Check if auto-apply corrections is enabled"""
        return self.config_data.get("settings", {}).get("auto_apply_corrections", False)
    
    def set_auto_apply_enabled(self, enabled: bool):
        """Enable/disable auto-apply corrections"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["auto_apply_corrections"] = enabled
    
    def should_show_changes(self) -> bool:
        """Check if correction changes should be shown"""
        return self.config_data.get("settings", {}).get("show_correction_changes", True)
    
    def set_show_changes(self, show: bool):
        """Set whether to show correction changes"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["show_correction_changes"] = show
    
    def should_backup_original(self) -> bool:
        """Check if original text should be backed up"""
        return self.config_data.get("settings", {}).get("backup_original_text", True)
    
    def set_backup_original(self, backup: bool):
        """Set whether to backup original text"""
        if "settings" not in self.config_data:
            self.config_data["settings"] = {}
        self.config_data["settings"]["backup_original_text"] = backup
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled providers"""
        enabled_providers = []
        for provider, config in self.config_data.get("providers", {}).items():
            if config.get("enabled", False) and config.get("api_key", "").strip():
                enabled_providers.append(provider)
        return enabled_providers
    
    def update_last_used(self, provider: str, model: str):
        """Update last used provider and model"""
        if "last_used" not in self.config_data:
            self.config_data["last_used"] = {}
        self.config_data["last_used"]["provider"] = provider
        self.config_data["last_used"]["model"] = model
    
    def get_last_used_provider(self) -> str:
        """Get last used provider"""
        return self.config_data.get("last_used", {}).get("provider", "")
    
    def get_last_used_model(self) -> str:
        """Get last used model"""
        return self.config_data.get("last_used", {}).get("model", "")
    
    def validate_provider_config(self, provider: str) -> bool:
        """Validate if a provider is properly configured"""
        provider_config = self.get_provider_config(provider)
        
        # Check if API key is provided
        api_key = provider_config.get("api_key", "").strip()
        if not api_key:
            return False
        
        # Check if model is selected
        model = provider_config.get("model", "").strip()
        if not model:
            return False
        
        return True
    
    def reset_provider_config(self, provider: str):
        """Reset a provider's configuration to defaults"""
        default_config = self._load_config()
        if provider in default_config.get("providers", {}):
            self.config_data["providers"][provider] = default_config["providers"][provider]
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration (without sensitive data like API keys)"""
        config_copy = self.config_data.copy()
        
        # Remove API keys for export
        for provider in config_copy.get("providers", {}):
            if "api_key" in config_copy["providers"][provider]:
                config_copy["providers"][provider]["api_key"] = "***HIDDEN***"
        
        return config_copy
    
    def import_config(self, config_data: Dict[str, Any], preserve_api_keys: bool = True):
        """Import configuration"""
        if preserve_api_keys:
            # Preserve existing API keys
            existing_keys = {}
            for provider in self.config_data.get("providers", {}):
                existing_keys[provider] = self.get_api_key(provider)
        
        self.config_data = config_data
        
        if preserve_api_keys:
            # Restore API keys
            for provider, api_key in existing_keys.items():
                if api_key and provider in self.config_data.get("providers", {}):
                    self.set_api_key(provider, api_key)


# Global configuration instance
ai_config = AIConfig()