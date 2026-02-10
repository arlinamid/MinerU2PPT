#!/usr/bin/env python3
"""
Cache manager for AI text corrections
Manages cached corrections across all pages and conversion sessions
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextCorrectionCacheManager:
    """Global cache manager for AI text corrections"""
    
    def __init__(self, cache_file: str = "ai_text_cache.json"):
        self.cache_file = os.path.join(os.path.dirname(__file__), "..", cache_file)
        self.cache: Dict[str, Dict] = {}  # text -> {corrected, timestamp, provider, confidence}
        self.lock = threading.Lock()
        self.max_cache_size = 10000  # Maximum number of cached entries
        self.cache_ttl_days = 30  # Cache time-to-live in days
        
        self.load_cache()
    
    def _get_cache_key(self, text: str, provider: str = "default") -> str:
        """Generate a cache key from text and provider"""
        # Use first 100 chars + hash for very long texts to avoid key length issues
        if len(text) > 100:
            import hashlib
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
            key_text = text[:100] + f"...#{text_hash}"
        else:
            key_text = text
        
        return f"{provider}:{key_text}"
    
    def get_corrected_text(self, original_text: str, provider: str = "default") -> Optional[str]:
        """Get corrected text from cache if available"""
        with self.lock:
            cache_key = self._get_cache_key(original_text, provider)
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry is still valid (not expired)
                entry_date = datetime.fromisoformat(entry.get('timestamp', '1900-01-01'))
                if datetime.now() - entry_date < timedelta(days=self.cache_ttl_days):
                    return entry.get('corrected_text')
                else:
                    # Entry expired, remove it
                    del self.cache[cache_key]
        
        return None
    
    def cache_correction(self, original_text: str, corrected_text: str, 
                        provider: str = "default", confidence: float = 1.0):
        """Cache a text correction"""
        with self.lock:
            # Don't cache if texts are identical
            if original_text == corrected_text:
                return
            
            cache_key = self._get_cache_key(original_text, provider)
            
            self.cache[cache_key] = {
                'original_text': original_text,
                'corrected_text': corrected_text,
                'provider': provider,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Clean up old/excess entries if cache is too large
            if len(self.cache) > self.max_cache_size:
                self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove old or excess cache entries"""
        try:
            # Remove expired entries first
            cutoff_date = datetime.now() - timedelta(days=self.cache_ttl_days)
            expired_keys = []
            
            for key, entry in self.cache.items():
                entry_date = datetime.fromisoformat(entry.get('timestamp', '1900-01-01'))
                if entry_date < cutoff_date:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            # If still too large, remove oldest entries
            if len(self.cache) > self.max_cache_size:
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].get('timestamp', '1900-01-01')
                )
                
                # Keep only the most recent entries
                entries_to_keep = sorted_entries[-self.max_cache_size:]
                self.cache = dict(entries_to_keep)
            
            logger.info(f"Cache cleanup completed. Current size: {len(self.cache)}")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def clear_all(self) -> int:
        """Clear all cached entries and return count of cleared entries"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.save_cache()
            logger.info(f"Cleared {count} cached text corrections")
            return count
    
    def clear_by_provider(self, provider: str) -> int:
        """Clear cached entries for a specific provider"""
        with self.lock:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{provider}:")]
            count = len(keys_to_remove)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            self.save_cache()
            logger.info(f"Cleared {count} cached text corrections for provider: {provider}")
            return count
    
    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        with self.lock:
            total_entries = len(self.cache)
            
            # Calculate memory usage (approximate)
            memory_usage = sum(
                len(entry.get('original_text', '')) + len(entry.get('corrected_text', ''))
                for entry in self.cache.values()
            )
            
            # Provider breakdown
            providers = {}
            for key, entry in self.cache.items():
                provider = entry.get('provider', 'unknown')
                providers[provider] = providers.get(provider, 0) + 1
            
            # Recent activity
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_entries = 0
            for entry in self.cache.values():
                entry_date = datetime.fromisoformat(entry.get('timestamp', '1900-01-01'))
                if entry_date > recent_cutoff:
                    recent_entries += 1
            
            return {
                'total_entries': total_entries,
                'memory_usage_chars': memory_usage,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'providers': providers,
                'recent_24h': recent_entries,
                'cache_file': self.cache_file,
                'max_size': self.max_cache_size,
                'ttl_days': self.cache_ttl_days
            }
    
    def get_recent_corrections(self, limit: int = 10) -> List[Dict]:
        """Get most recent corrections for display"""
        with self.lock:
            sorted_entries = sorted(
                self.cache.values(),
                key=lambda x: x.get('timestamp', '1900-01-01'),
                reverse=True
            )
            
            recent = []
            for entry in sorted_entries[:limit]:
                recent.append({
                    'original': entry.get('original_text', '')[:50] + "..." if len(entry.get('original_text', '')) > 50 else entry.get('original_text', ''),
                    'corrected': entry.get('corrected_text', '')[:50] + "..." if len(entry.get('corrected_text', '')) > 50 else entry.get('corrected_text', ''),
                    'provider': entry.get('provider', 'unknown'),
                    'timestamp': entry.get('timestamp', ''),
                    'confidence': entry.get('confidence', 0.0)
                })
            
            return recent
    
    def load_cache(self):
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached text corrections")
            else:
                logger.info("No existing cache file found, starting with empty cache")
        except Exception as e:
            logger.error(f"Error loading cache from {self.cache_file}: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.cache)} cached text corrections")
        except Exception as e:
            logger.error(f"Error saving cache to {self.cache_file}: {e}")
    
    def cleanup_and_save(self):
        """Perform cleanup and save cache"""
        with self.lock:
            self._cleanup_cache()
            self.save_cache()


# Global cache manager instance
global_cache_manager = TextCorrectionCacheManager()


def get_corrected_text(original_text: str, provider: str = "default") -> Optional[str]:
    """Convenience function to get corrected text from global cache"""
    return global_cache_manager.get_corrected_text(original_text, provider)


def cache_correction(original_text: str, corrected_text: str, provider: str = "default", confidence: float = 1.0):
    """Convenience function to cache a text correction in global cache"""
    global_cache_manager.cache_correction(original_text, corrected_text, provider, confidence)


def clear_all_cache() -> int:
    """Convenience function to clear all cached text corrections"""
    return global_cache_manager.clear_all()


def get_cache_stats() -> Dict:
    """Convenience function to get cache statistics"""
    return global_cache_manager.get_cache_stats()