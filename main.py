#!/usr/bin/env python3
"""
MinerU2PPTX Command Line Interface v2.0.1

This enhanced version is inspired by and extends the original MinerU2PPT project:
https://github.com/JuniverseCoder/MinerU2PPT

Author: Arlinamid (Rózsavölgyi János)
GitHub: https://github.com/arlinamid
"""

import argparse
import os
import sys
import shutil

__version__ = "2.0.1"


def cmd_convert(args):
    """Handle the 'convert' sub-command."""
    from converter.generator import convert_mineru_to_ppt

    if args.debug_images:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.makedirs("tmp")

    # AI configuration
    enable_ai_correction = None
    if args.enable_ai:
        enable_ai_correction = True
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

    print(f"Converting {args.input} to {args.output} ...")
    try:
        convert_mineru_to_ppt(
            args.json, args.input, args.output,
            remove_watermark=args.no_watermark,
            debug_images=args.debug_images,
            enable_ai_correction=enable_ai_correction,
        )
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def cmd_extract_images(args):
    """Handle the 'extract-images' sub-command."""
    from converter.image_downloader import extract_images

    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    out_dir = args.output or "minerU_images"
    print(f"Extracting images from {args.input} using {args.json} ...")
    print(f"Output folder: {out_dir}  |  DPI: {args.dpi}  |  Overwrite: {args.overwrite}")

    result = extract_images(
        json_path=args.json,
        pdf_path=args.input,
        work_folder=out_dir,
        overwrite=args.overwrite,
        dpi=args.dpi,
    )

    print(f"\nDone — {result['extracted']}/{result['total_images']} images extracted, "
          f"{result['skipped']} skipped, {result['errors']} errors")
    print(f"Saved to: {result['work_folder']}")

    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="MinerU2PPTX",
        description="MinerU PDF/Image to PPTX Converter & Image Extractor",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── convert ──────────────────────────────────────────────────────
    p_conv = subparsers.add_parser("convert", help="Convert MinerU JSON + PDF/Image to PPTX")
    p_conv.add_argument("--json", required=True, help="Path to MinerU JSON file")
    p_conv.add_argument("--input", required=True, help="Path to original PDF/Image file")
    p_conv.add_argument("--output", required=True, help="Path to output PPTX file")
    p_conv.add_argument("--no-watermark", action="store_true", help="Remove watermarks")
    p_conv.add_argument("--debug-images", action="store_true", help="Generate debug images in tmp/")

    ai = p_conv.add_argument_group("AI Text Correction")
    ai.add_argument("--enable-ai", action="store_true", help="Enable AI text correction")
    ai.add_argument("--disable-ai", action="store_true", help="Disable AI text correction")
    ai.add_argument("--ai-provider", choices=["OpenAI", "Google Gemini", "Anthropic Claude", "Groq"])
    ai.add_argument("--ai-model", help="AI model name")
    ai.add_argument("--api-key", help="API key for the AI provider")
    ai.add_argument("--correction-type", choices=["grammar_spelling", "formatting", "both"],
                    default="grammar_spelling")

    # ── extract-images ───────────────────────────────────────────────
    p_img = subparsers.add_parser("extract-images", help="Extract images from PDF using MinerU JSON bounding boxes")
    p_img.add_argument("--json", required=True, help="Path to MinerU JSON file")
    p_img.add_argument("--input", required=True, help="Path to original PDF/Image file")
    p_img.add_argument("--output", default=None, help="Output folder (default: minerU_images)")
    p_img.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    p_img.add_argument("--overwrite", action="store_true", help="Overwrite existing images")

    # ── legacy mode (no sub-command, --json/--input/--output) ────────
    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "extract-images":
        cmd_extract_images(args)
    elif args.command is None:
        # If no sub-command, check for legacy --json flag (backwards compat)
        parser.print_help()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
