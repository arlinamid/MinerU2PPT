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
- **Debugging**: The application automatically generates debug images in the `tmp/` directory for each page processed. This directory is cleared at the start of each run.

### Testing
- **Run Logic Tests**: `python tests/test_logic.py` (Tests color extraction and font size estimation)

## Code Architecture

### High-Level Structure
- **`main.py`**: Entry point and CLI logic. Orchestrates conversion, evaluation, and debug directory management.
- **`converter/`**: Core conversion logic.
  - **`generator.py`**: Contains `PPTGenerator` which maps MinerU JSON elements to PPT shapes. It orchestrates the advanced text processing pipeline.
  - **`utils.py`**: Low-level helpers for PDF-to-image conversion, advanced color/style analysis, and image projection.
- **`evaluator/`**: Quality assurance tools.
  - **`checker.py`**: Exports PPT slides to images using Windows COM and calculates SSIM.

### Key Implementation Details
- **Coordinate System**: MinerU coordinates are scaled to PPT point-based coordinates.
- **Asset Handling**: Non-text elements are cropped from high-DPI page images and inserted as pictures.
- **Advanced Text Processing**: A multi-stage pipeline accurately extracts and styles text character-by-character:
  1.  **Line Detection**: Uses a Y-axis projection to identify text lines. To handle varying font sizes, it defines an expanded vertical search area for each line that extends to the bottom of the previous line.
  2.  **Character Segmentation**: An X-axis projection segments characters. It can recover text of a different color from gaps by performing a recursive search, using a 5-pixel horizontal safety margin to prevent interference from primary font anti-aliasing.
  3.  **Noise Filtering**: Any potential character segment whose width is less than a fixed threshold is discarded as noise.
  4.  **Bounding Box Correction**: A dynamic programming algorithm (`_analyze_and_correct_bboxes`) aligns detected character fragments with the known text. It uses a fixed aspect ratio (0.3) for full-width punctuation to improve accuracy.
  5.  **Font Color Extraction**: A sophisticated algorithm (`extract_font_color`) determines the font color. It finds the most frequent color in a text block, defines a "cluster" of visually similar colors, and then selects the color from that cluster with the highest contrast against the background.
  6.  **Style Normalization**: After determining raw styles, two normalization passes (`_normalize_font_sizes` and `_normalize_colors`) unify styles that are visually similar, ensuring high consistency.
- **Windows Dependency**: The evaluation system requires a Windows environment with PowerPoint installed.
