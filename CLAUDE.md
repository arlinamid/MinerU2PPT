# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- **Install Dependencies**: `pip install -r requirements.txt`
- **Python Version**: Python 3.10+ recommended

### Execution
- **Run Converter**: `python main.py --json <path_to_json> --pdf <path_to_pdf> --output <path_to_ppt>`
- **Run with Evaluation**: Add `--eval` flag to compare generated PPT with original PDF via SSIM.
- **Evaluation Output**: Comparison images are saved to `eval_output/comparison/` by default.

### Testing
- **Run Logic Tests**: `python tests/test_logic.py` (Tests color extraction and font size estimation)

## Code Architecture

### High-Level Structure
- **`main.py`**: Entry point and CLI logic. Orchestrates conversion and evaluation phases.
- **`converter/`**: Core conversion logic.
  - **`generator.py`**: Contains `PPTGenerator` which maps MinerU JSON elements (text, images, tables) to PPT shapes using `python-pptx`. Handles coordinate scaling and slide layout.
  - **`utils.py`**: Low-level helpers for PDF-to-image conversion (`pymupdf`), background/font color extraction (`opencv`), and font size estimation.
- **`evaluator/`**: Quality assurance tools.
  - **`checker.py`**: Exports PPT slides to images using Windows COM (`win32com.client`) and calculates Structural Similarity Index (SSIM) between original and generated pages.

### Key Implementation Details
- **Coordinate System**: MinerU coordinates are scaled from the input image size to PPT point-based coordinates.
- **Styling**: Uses "Microsoft YaHei" as the default font. Font size and color are dynamically estimated from the original document images within the bounding boxes.
- **Asset Handling**: Non-text elements (images, formulas, tables) are cropped from the original high-DPI page images and inserted as pictures into the PPT.
- **Windows Dependency**: The evaluation system (`evaluator/checker.py`) requires a Windows environment with PowerPoint installed to perform PPT-to-Image conversion via COM.
