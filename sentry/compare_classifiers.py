import sys
import os
import glob
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import cv2
from itertools import combinations
import json
from datetime import datetime


def compare_ssim(img1_array, img2_array):
    """Structural Similarity Index - sensitive to structural changes."""
    score = ssim(img1_array, img2_array, multichannel=True, channel_axis=-1, data_range=1.0)
    return score


def compare_mse(img1_array, img2_array):
    """Mean Squared Error - lower is better (inverted for consistency)."""
    mse = np.mean((img1_array - img2_array) ** 2)
    similarity = 1 / (1 + mse * 100)
    return similarity


def compare_correlation(img1_array, img2_array):
    """Normalized Cross-Correlation - measures linear relationship."""
    flat1 = img1_array.flatten()
    flat2 = img2_array.flatten()
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    similarity = (correlation + 1) / 2
    return similarity


def compare_histogram(img1, img2):
    """Histogram comparison - compares color/intensity distributions."""
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2HSV)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2HSV)

    hist1 = cv2.calcHist([img1_cv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_cv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def compare_perceptual_hash(img1, img2):
    """Perceptual hashing - robust to minor variations."""
    img1_small = img1.resize((32, 32), Image.LANCZOS).convert('L')
    img2_small = img2.resize((32, 32), Image.LANCZOS).convert('L')

    arr1 = np.array(img1_small).flatten()
    arr2 = np.array(img2_small).flatten()

    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)

    hash1 = arr1 > mean1
    hash2 = arr2 > mean2

    hamming_distance = np.sum(hash1 != hash2)
    similarity = 1 - (hamming_distance / len(hash1))
    return similarity


def compare_feature_matching(img1, img2):
    """Feature-based matching using ORB."""
    try:
        img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)

        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        if des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        max_possible_matches = min(len(kp1), len(kp2))
        if max_possible_matches == 0:
            return 0.0

        similarity = len(matches) / max_possible_matches
        return similarity
    except Exception as e:
        return 0.0


def compare_two_images(img1_path, img2_path):
    """Compare two images using all 6 methods."""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_array = img_as_float(np.array(img1))
        img2_array = img_as_float(np.array(img2))

        # Ensure same size
        if img1_array.shape != img2_array.shape:
            img2 = img2.resize(img1.size, Image.LANCZOS)
            img2_array = img_as_float(np.array(img2))

        results = {
            'ssim': compare_ssim(img1_array, img2_array),
            'mse': compare_mse(img1_array, img2_array),
            'correlation': compare_correlation(img1_array, img2_array),
            'histogram': compare_histogram(img1, img2),
            'perceptual_hash': compare_perceptual_hash(img1, img2),
            'feature_matching': compare_feature_matching(img1, img2)
        }

        return results
    except Exception as e:
        print(f"  Error comparing {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}: {e}")
        return None


def analyze_library(library_path, output_file=None):
    """Analyze all images in library directory."""

    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(library_path, ext)))

    if len(image_files) == 0:
        print(f"No images found in {library_path}")
        return

    print(f"Found {len(image_files)} images in library")
    print(f"Will perform {len(list(combinations(range(len(image_files)), 2)))} comparisons")
    print()

    # Store all results
    all_comparisons = []
    method_stats = {
        'ssim': [],
        'mse': [],
        'correlation': [],
        'histogram': [],
        'perceptual_hash': [],
        'feature_matching': []
    }

    # Compare all pairs
    pairs = list(combinations(image_files, 2))
    total_pairs = len(pairs)

    for idx, (img1_path, img2_path) in enumerate(pairs, 1):
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)

        print(f"[{idx}/{total_pairs}] Comparing: {img1_name} vs {img2_name}")

        results = compare_two_images(img1_path, img2_path)

        if results:
            comparison = {
                'image1': img1_name,
                'image2': img2_name,
                'scores': results
            }
            all_comparisons.append(comparison)

            # Collect statistics
            for method, score in results.items():
                method_stats[method].append(score)

            # Print summary for this pair
            print(f"  Scores: " + " | ".join([f"{k}: {v:.3f}" for k, v in results.items()]))

        print()

    # Calculate statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    summary = {}
    for method, scores in method_stats.items():
        if scores:
            summary[method] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            }

            print(f"{method.upper().replace('_', ' ')}:")
            print(f"  Mean:   {summary[method]['mean']:.4f}")
            print(f"  Std:    {summary[method]['std']:.4f}")
            print(f"  Min:    {summary[method]['min']:.4f}")
            print(f"  Max:    {summary[method]['max']:.4f}")
            print(f"  Median: {summary[method]['median']:.4f}")
            print()

    # Find most similar and most different pairs
    print("=" * 80)
    print("TOP 5 MOST SIMILAR PAIRS (by each method)")
    print("=" * 80)
    print()

    for method in method_stats.keys():
        print(f"\n{method.upper().replace('_', ' ')}:")
        sorted_comparisons = sorted(all_comparisons,
                                    key=lambda x: x['scores'][method],
                                    reverse=True)
        for i, comp in enumerate(sorted_comparisons[:5], 1):
            print(f"  {i}. {comp['image1']} vs {comp['image2']}: {comp['scores'][method]:.4f}")

    print("\n" + "=" * 80)
    print("TOP 5 MOST DIFFERENT PAIRS (by each method)")
    print("=" * 80)
    print()

    for method in method_stats.keys():
        print(f"\n{method.upper().replace('_', ' ')}:")
        sorted_comparisons = sorted(all_comparisons,
                                    key=lambda x: x['scores'][method])
        for i, comp in enumerate(sorted_comparisons[:5], 1):
            print(f"  {i}. {comp['image1']} vs {comp['image2']}: {comp['scores'][method]:.4f}")

    # Save to JSON file
    if output_file:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'library_path': library_path,
            'total_images': len(image_files),
            'total_comparisons': len(all_comparisons),
            'summary_statistics': summary,
            'all_comparisons': all_comparisons
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n\nDetailed results saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python library_analyzer.py <library_directory> [output_json_file]")
        print("Example: python library_analyzer.py library_spectrograms/")
        print("Example: python library_analyzer.py library_spectrograms/ results.json")
        sys.exit(1)

    library_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(library_path):
        print(f"Error: Directory '{library_path}' not found")
        sys.exit(1)

    if not os.path.isdir(library_path):
        print(f"Error: '{library_path}' is not a directory")
        sys.exit(1)

    analyze_library(library_path, output_file)


if __name__ == "__main__":
    main()