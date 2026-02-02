import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from converter.generator import PPTGenerator

def test_line_detection():
    # Create a dummy generator
    gen = PPTGenerator("dummy.pptx")

    # Create a synthetic image: 200x200 white background
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Add two "lines" of text (black blocks)
    # Line 1: y from 20 to 40
    img[20:40, 20:180] = [0, 0, 0]
    # Line 2: y from 80 to 100
    img[80:100, 20:180] = [0, 0, 0]

    # BBox in JSON coordinates (let's say 1:1 for simplicity)
    bbox = [0, 0, 200, 200]
    coords = {
        'img_w': 200, 'img_h': 200,
        'json_w': 200, 'json_h': 200,
        'scale_x': 1, 'scale_y': 1
    }

    lines = gen._detect_lines_from_image(img, bbox, coords)

    print(f"Detected lines: {lines}")

    # Should detect 2 lines
    assert len(lines) == 2
    # Verify approximate y-coordinates
    assert abs(lines[0][0] - 20) < 2
    assert abs(lines[0][1] - 40) < 2
    assert abs(lines[1][0] - 80) < 2
    assert abs(lines[1][1] - 100) < 2

def test_line_detection_with_noise():
    gen = PPTGenerator("dummy.pptx")
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    # Line 1 (Black)
    img[20:40, 20:180] = [0, 0, 0]

    # Noise: a gray block that is far from white bg (255,255,255)
    # but also far from black font (0,0,0) or closer to bg than font?
    # bg = (255, 255, 255), font = (0, 0, 0)
    # gray = (200, 200, 200)
    # diff_bg = sqrt(3 * 55^2) = 95 > 30
    # diff_font = sqrt(3 * 200^2) = 346
    # diff_font > diff_bg, so it should be filtered out by (diff_font < diff_bg)
    img[120:140, 20:180] = [200, 200, 200]

    bbox = [0, 0, 200, 200]
    coords = {
        'img_w': 200, 'img_h': 200,
        'json_w': 200, 'json_h': 200,
        'scale_x': 1, 'scale_y': 1
    }

    lines = gen._detect_lines_from_image(img, bbox, coords)
    print(f"Detected lines with noise: {lines}")

    # Should only detect 1 line (the black one)
    assert len(lines) == 1
    assert abs(lines[0][0] - 20) < 2
    assert abs(lines[0][1] - 40) < 2

def test_font_height_detection():
    gen = PPTGenerator("dummy.pptx")
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Text block (black) from y=10 to y=25 (height 15)
    img[10:26, 10:90] = [0, 0, 0]

    bbox = [0, 0, 100, 40]
    coords = {
        'img_w': 100, 'img_h': 100,
        'json_w': 100, 'json_h': 100,
        'scale_x': 1, 'scale_y': 1
    }

    # The bbox provided to font height detection is the span bbox
    span_bbox = [10, 5, 90, 35] # includes the text block
    height = gen._detect_font_height_from_line(img, span_bbox, coords)
    print(f"Detected font height: {height}")

    # Should be around 16 pixels (10 to 25 inclusive)
    assert abs(height - 16) < 2

if __name__ == "__main__":
    try:
        test_line_detection()
        test_line_detection_with_noise()
        test_font_height_detection()
        print("Tests passed!")
        if os.path.exists("dummy.pptx"):
            try:
                os.remove("dummy.pptx")
            except:
                pass
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
