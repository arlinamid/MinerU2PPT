import argparse
import os
import sys
from converter.generator import convert_mineru_to_ppt
from converter.utils import pdf_to_images
from evaluator.checker import ppt_to_images, calculate_similarity, create_comparison

def main():
    parser = argparse.ArgumentParser(description="MinerU PDF/Image to PPT Converter")
    parser.add_argument("--json", required=True, help="Path to MinerU JSON file")
    parser.add_argument("--pdf", required=True, help="Path to original PDF/Image file")
    parser.add_argument("--output", required=True, help="Path to output PPT file")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after conversion")
    parser.add_argument("--debug", action="store_true", help="Generate an additional debug PPT without background cleaning.")
    parser.add_argument("--eval-dir", default="eval_output", help="Directory for evaluation images")

    args = parser.parse_args()

    print(f"Converting {args.pdf} to {args.output}...")
    try:
        convert_mineru_to_ppt(args.json, args.pdf, args.output, debug=args.debug)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

    if args.eval:
        print("Running evaluation...")
        try:
            # 1. Get original images
            orig_images = pdf_to_images(args.pdf)
            orig_dir = os.path.join(args.eval_dir, "original")
            if not os.path.exists(orig_dir):
                os.makedirs(orig_dir)

            import cv2
            orig_paths = []
            for i, img in enumerate(orig_images):
                p = os.path.join(orig_dir, f"page_{i+1}.png")
                cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                orig_paths.append(p)

            # 2. Convert PPT to images
            gen_dir = os.path.join(args.eval_dir, "generated")
            gen_paths = ppt_to_images(args.output, gen_dir)

            # 3. Compare
            comp_dir = os.path.join(args.eval_dir, "comparison")
            if not os.path.exists(comp_dir):
                os.makedirs(comp_dir)

            total_score = 0
            count = min(len(orig_paths), len(gen_paths))

            for i in range(count):
                score, _ = calculate_similarity(orig_paths[i], gen_paths[i])
                total_score += score
                comp_path = os.path.join(comp_dir, f"comp_{i+1}.png")
                create_comparison(orig_paths[i], gen_paths[i], comp_path, score)
                print(f"Page {i+1} SSIM: {score:.4f}")

            if count > 0:
                print(f"Average SSIM: {total_score / count:.4f}")
            print(f"Evaluation complete. Comparison images saved to {comp_dir}")

        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
