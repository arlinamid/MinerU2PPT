#!/usr/bin/env python3
"""
MinerU2PPTX Command Line Interface

This enhanced version is inspired by and extends the original MinerU2PPT project:
https://github.com/JuniverseCoder/MinerU2PPT

Author: arlinamid
GitHub: https://github.com/arlinamid
"""

import argparse
import os
import sys
import shutil
from converter.generator import convert_mineru_to_ppt

def main():
    parser = argparse.ArgumentParser(description="MinerU PDF/Image to PPTX Converter")
    parser.add_argument("--json", required=True, help="Path to MinerU JSON file")
    parser.add_argument("--input", required=True, help="Path to original PDF/Image file")
    parser.add_argument("--output", required=True, help="Path to output PPTX file")
    parser.add_argument("--no-watermark", action="store_true", help="Remove watermarks from the output")
    parser.add_argument("--debug-images", action="store_true", help="Generate debug images in the tmp/ directory")
    
    # AI Text Correction Options
    ai_group = parser.add_argument_group('AI Text Correction')
    ai_group.add_argument("--enable-ai", action="store_true", help="Enable AI text correction")
    ai_group.add_argument("--disable-ai", action="store_true", help="Disable AI text correction")
    ai_group.add_argument("--ai-provider", choices=["OpenAI", "Google Gemini", "Anthropic Claude", "Groq"], 
                         help="AI provider for text correction")
    ai_group.add_argument("--ai-model", help="AI model to use for text correction")
    ai_group.add_argument("--api-key", help="API key for the AI provider")
    ai_group.add_argument("--correction-type", choices=["grammar_spelling", "formatting", "both"], 
                         default="grammar_spelling", help="Type of text correction to apply")

    args = parser.parse_args()

    if args.debug_images:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.makedirs("tmp")

    # Handle AI configuration
    enable_ai_correction = None
    if args.enable_ai:
        enable_ai_correction = True
        # Configure AI settings if provided
        if args.ai_provider and args.api_key:
            from converter.config import ai_config
            ai_config.set_api_key(args.ai_provider, args.api_key)
            ai_config.set_active_provider(args.ai_provider)
            ai_config.set_provider_enabled(args.ai_provider, True)
            if args.ai_model:
                ai_config.set_model(args.ai_provider, args.ai_model)
            if args.correction_type:
                ai_config.set_correction_type(args.correction_type)
            ai_config.set_text_correction_enabled(True)
            ai_config.save_config()
            print(f"AI text correction enabled with {args.ai_provider}")
    elif args.disable_ai:
        enable_ai_correction = False

    print(f"Converting {args.input} to {args.output}...")
    try:
        convert_mineru_to_ppt(args.json, args.input, args.output, 
                             remove_watermark=args.no_watermark, 
                             debug_images=args.debug_images,
                             enable_ai_correction=enable_ai_correction)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
