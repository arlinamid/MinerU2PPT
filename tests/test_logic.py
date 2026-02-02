import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from converter.utils import extract_colors

def test_extract_colors():
    # Create a white image with a red text block
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    # Red block (255, 0, 0)
    img[45:55, 45:55] = [255, 0, 0]

    # BBox slightly larger than the red block to include white background at edges
    bbox = [40, 40, 60, 60]
    bg_color, font_color = extract_colors(img, bbox)

    print(f"Background color: {bg_color}")
    print(f"Font color: {font_color}")

    assert bg_color == (255, 255, 255)
    # Font color should be red
    assert font_color == (255, 0, 0)

if __name__ == "__main__":
    try:
        test_extract_colors()
        print("Tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
