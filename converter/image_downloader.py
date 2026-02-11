#!/usr/bin/env python3
"""
MinerU Image Extractor Utility

Extracts images from original PDF files using bounding box data from MinerU JSON.
Uses the same approach as the PPTX generator for reliable local extraction.

Author: Arlinamid (Rózsavölgyi János)
"""

import json
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extracts images from PDF using bounding boxes from MinerU JSON."""

    def __init__(self, work_folder: str = "minerU_images"):
        self.work_folder = Path(work_folder)
        self.work_folder.mkdir(parents=True, exist_ok=True)
        self.extracted_count = 0
        self.skipped_count = 0
        self.error_count = 0

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_pages(self, json_data: Dict) -> List[Dict]:
        """Return the list of page dicts from any MinerU JSON variant."""
        if isinstance(json_data, list):
            return json_data
        if "pdf_info" in json_data:
            return json_data["pdf_info"]
        if "pages" in json_data:
            return json_data["pages"]
        return [json_data]

    def extract_image_info_from_json(self, json_data: Dict) -> List[Dict]:
        """
        Walk through MinerU JSON and collect every image element with its
        bounding box, page index and a generated filename.
        """
        images: List[Dict] = []
        pages = self._parse_pages(json_data)
        logger.info(f"Found {len(pages)} pages to scan")

        for page_idx, page_data in enumerate(pages):
            # Gather image-type elements from all possible containers
            image_items: List[Dict] = []
            if "images" in page_data:
                image_items.extend(page_data["images"])
            for key in ("para_blocks", "blocks", "elements"):
                for item in page_data.get(key, []):
                    if item.get("type") == "image":
                        image_items.append(item)

            for img_idx, item in enumerate(image_items):
                bbox = item.get("bbox")
                if not bbox:
                    continue
                filename = f"page_{page_idx:03d}_img_{img_idx:03d}.jpg"
                images.append({
                    "page": page_idx,
                    "image_index": img_idx,
                    "bbox": bbox,
                    "filename": filename,
                })

            if image_items:
                logger.info(f"Page {page_idx}: {len(image_items)} image(s)")

        return images

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def extract_images_from_pdf(self, json_path: str, pdf_path: str,
                                overwrite: bool = False, dpi: int = 200,
                                cancel_event=None) -> Dict:
        """
        Render each PDF page and crop out images using JSON bounding boxes.

        Returns a stats dict with total_images, extracted, skipped, errors.
        """
        self.extracted_count = 0
        self.skipped_count = 0
        self.error_count = 0

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            image_infos = self.extract_image_info_from_json(json_data)
            if not image_infos:
                logger.warning("No images found in JSON file")
                return self._stats(0)

            pages = self._parse_pages(json_data)
            pdf_doc = fitz.open(pdf_path)

            # Group by page so we render each page only once
            by_page: Dict[int, List[Dict]] = {}
            for info in image_infos:
                by_page.setdefault(info["page"], []).append(info)

            logger.info(f"Extracting {len(image_infos)} images from {len(by_page)} pages")

            for page_num, page_images in by_page.items():
                if cancel_event and cancel_event.is_set():
                    logger.info("Extraction cancelled by user")
                    break

                if page_num >= len(pdf_doc):
                    logger.warning(f"Page {page_num} out of range (PDF has {len(pdf_doc)} pages)")
                    self.error_count += len(page_images)
                    continue

                # Render page
                page = pdf_doc[page_num]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # JSON coordinate space
                json_page = pages[page_num] if page_num < len(pages) else {}
                json_w, json_h = self._json_page_size(json_page)

                for info in page_images:
                    if cancel_event and cancel_event.is_set():
                        break
                    self._crop_and_save(img, info, pix.width, pix.height,
                                        json_w, json_h, overwrite)

            pdf_doc.close()
            return self._stats(len(image_infos))

        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return self._stats(0, error=str(e))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _json_page_size(page_data: Dict):
        """Return (width, height) from the JSON page metadata."""
        ps = page_data.get("page_size")
        if not ps and "page_info" in page_data:
            pi = page_data["page_info"]
            if isinstance(pi, dict):
                ps = (pi.get("width"), pi.get("height"))
        w = ps[0] if ps and ps[0] else 1376
        h = ps[1] if ps and ps[1] else 768
        return w, h

    def _crop_and_save(self, page_img: np.ndarray, info: Dict,
                       pw: int, ph: int, jw: int, jh: int,
                       overwrite: bool) -> Optional[str]:
        try:
            path = self.work_folder / info["filename"]
            if path.exists() and not overwrite:
                self.skipped_count += 1
                return str(path)

            x1, y1, x2, y2 = info["bbox"]
            px = lambda v, dim_img, dim_json: max(0, int(v * dim_img / dim_json))
            crop = page_img[px(y1, ph, jh):px(y2, ph, jh),
                            px(x1, pw, jw):px(x2, pw, jw)].copy()

            if crop.size == 0:
                self.error_count += 1
                return None

            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            self.extracted_count += 1
            logger.info(f"Extracted: {info['filename']}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to extract {info.get('filename', '?')}: {e}")
            self.error_count += 1
            return None

    def _stats(self, total: int, error: str = None) -> Dict:
        d: Dict = {
            "total_images": total,
            "extracted": self.extracted_count,
            "skipped": self.skipped_count,
            "errors": self.error_count,
            "work_folder": str(self.work_folder),
        }
        if error:
            d["error"] = error
        return d

    def set_work_folder(self, work_folder: str):
        self.work_folder = Path(work_folder)
        self.work_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def extract_images(json_path: str, pdf_path: str,
                   work_folder: str = "minerU_images",
                   overwrite: bool = False, dpi: int = 200,
                   cancel_event=None) -> Dict:
    """One-call helper to extract all images from a MinerU PDF."""
    extractor = ImageExtractor(work_folder)
    return extractor.extract_images_from_pdf(json_path, pdf_path, overwrite, dpi, cancel_event)
