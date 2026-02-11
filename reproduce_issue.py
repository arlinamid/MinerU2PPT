
import sys
import os
import asyncio
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gemini_auth():
    print("--- Testing Gemini Auth ---")
    
    # 1. Check Import
    try:
        import google.genai as genai
        print("[OK] import google.genai")
    except ImportError as e:
        print(f"[FAIL] import google.genai: {e}")
        return

    # 2. Load Config
    config_path = Path("ai_config.json")
    if not config_path.exists():
        print(f"[FAIL] ai_config.json not found at {config_path.absolute()}")
        return
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            api_key = config.get("providers", {}).get("Google Gemini", {}).get("api_key", "")
            if not api_key:
                print("[FAIL] No API key found in config for Google Gemini")
                return
            print(f"[OK] Found API Key: {api_key[:5]}...")
    except Exception as e:
        print(f"[FAIL] Error reading config: {e}")
        return

    # 3. Authenticate (Logic from GoogleGeminiService.authenticate)
    try:
        client = genai.Client(api_key=api_key)
        
        models_to_try = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash", 
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]
        
        for model in models_to_try:
            print(f"Testing model: {model}...")
            try:
                # Synchronous call wrapped in executor (simulating ai_services.py)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model,
                        contents="Hi"
                    ).text
                )
                print(f"[SUCCESS] Auth worked with {model}. Response: {response.strip()[:20]}")
                return
            except Exception as e:
                print(f"[WARN] Failed with {model}: {e}")
        
        print("[FAIL] All models failed.")

    except Exception as e:
        print(f"[FAIL] Client initialization or other error: {e}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_gemini_auth())
