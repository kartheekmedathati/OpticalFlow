"""
Evaluation script for the Wallach Psychophysical Optical Flow Benchmark.

Computes End-Point Error (EPE) and Angular Error (AE) per stimulus category,
and prints a leaderboard-ready summary.

Usage:
    python evaluate.py --predictions_dir /path/to/predictions --gt_dir /path/to/GT

Predictions should be .flo files matching the GT filenames.
"""

import os
import argparse
import numpy as np
from glob import glob
from collections import defaultdict

from utils.flow_io import flow_read
from utils.pcautils import epe_u_v


def angular_error(u_gt, v_gt, u_est, v_est):
    """Compute mean angular error in degrees between GT and estimated flow."""
    # Add 1 in the z-dimension for the 3D interpretation
    dot = u_gt * u_est + v_gt * v_est + 1.0
    mag_gt = np.sqrt(u_gt**2 + v_gt**2 + 1.0)
    mag_est = np.sqrt(u_est**2 + v_est**2 + 1.0)
    cos_angle = np.clip(dot / (mag_gt * mag_est), -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle)).mean()


def parse_stimulus_category(filename):
    """Extract the stimulus category from a filename."""
    base = os.path.basename(filename)
    for cat in ['hybridplaids', 'plaids', 'gratings', 'rectangles', 'lines', 'circles']:
        if base.startswith(cat):
            return cat
    return 'unknown'


def parse_subcategory(filename):
    """Extract sub-category info (radius, length, aspect ratio, etc.) from filename."""
    base = os.path.basename(filename)
    return base.split('_VX_')[0] if '_VX_' in base else base


def find_matching_predictions(gt_files, pred_dir):
    """Match GT files to prediction files by filename."""
    matched = []
    missing = []
    for gt_path in gt_files:
        pred_path = os.path.join(pred_dir, os.path.basename(gt_path))
        if os.path.exists(pred_path):
            matched.append((gt_path, pred_path))
        else:
            missing.append(gt_path)
    return matched, missing


def evaluate_pair(gt_path, pred_path):
    """Evaluate a single GT/prediction pair."""
    u_gt, v_gt = flow_read(gt_path)
    u_est, v_est = flow_read(pred_path)

    # Only evaluate on pixels where GT flow is non-zero
    mask = (u_gt != 0) | (v_gt != 0)
    if mask.sum() == 0:
        return None

    epe = np.sqrt((u_gt[mask] - u_est[mask])**2 + (v_gt[mask] - v_est[mask])**2).mean()
    ae = angular_error(u_gt[mask], v_gt[mask], u_est[mask], v_est[mask])

    return {'epe': epe, 'ae': ae}


def main():
    parser = argparse.ArgumentParser(description='Evaluate optical flow predictions on Wallach benchmark')
    parser.add_argument('--predictions_dir', required=True, help='Directory containing predicted .flo files')
    parser.add_argument('--gt_dir', required=True, help='Directory containing ground truth .flo files')
    parser.add_argument('--method_name', default='Method', help='Name of the method being evaluated')
    args = parser.parse_args()

    # Collect all GT .flo files
    gt_files = sorted(glob(os.path.join(args.gt_dir, '**/*.flo'), recursive=True))
    if not gt_files:
        print(f"No .flo files found in {args.gt_dir}")
        return

    print(f"Found {len(gt_files)} ground truth files")

    # Match with predictions
    matched, missing = find_matching_predictions(gt_files, args.predictions_dir)
    print(f"Matched: {len(matched)} | Missing predictions: {len(missing)}")

    if not matched:
        print("No matching predictions found. Ensure prediction filenames match GT filenames.")
        return

    # Evaluate per category
    results_by_cat = defaultdict(list)
    results_by_subcat = defaultdict(list)

    for gt_path, pred_path in matched:
        cat = parse_stimulus_category(gt_path)
        subcat = parse_subcategory(gt_path)
        result = evaluate_pair(gt_path, pred_path)
        if result is not None:
            results_by_cat[cat].append(result)
            results_by_subcat[subcat].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print(f"  WALLACH BENCHMARK RESULTS: {args.method_name}")
    print("=" * 70)
    print(f"\n{'Category':<20} {'Count':>6} {'EPE (px)':>10} {'AE (deg)':>10}")
    print("-" * 50)

    all_epes = []
    all_aes = []
    cat_order = ['circles', 'lines', 'rectangles', 'gratings', 'plaids', 'hybridplaids']

    for cat in cat_order:
        if cat not in results_by_cat:
            continue
        results = results_by_cat[cat]
        epes = [r['epe'] for r in results]
        aes = [r['ae'] for r in results]
        all_epes.extend(epes)
        all_aes.extend(aes)
        print(f"{cat:<20} {len(results):>6} {np.mean(epes):>10.4f} {np.mean(aes):>10.4f}")

    print("-" * 50)
    if all_epes:
        print(f"{'OVERALL':<20} {len(all_epes):>6} {np.mean(all_epes):>10.4f} {np.mean(all_aes):>10.4f}")

    # Detailed sub-category breakdown
    print(f"\n{'Sub-category':<50} {'Count':>6} {'EPE':>8} {'AE':>8}")
    print("-" * 75)
    for subcat in sorted(results_by_subcat.keys()):
        results = results_by_subcat[subcat]
        epes = [r['epe'] for r in results]
        aes = [r['ae'] for r in results]
        label = subcat[:48]
        print(f"{label:<50} {len(results):>6} {np.mean(epes):>8.4f} {np.mean(aes):>8.4f}")

    # Output markdown table for leaderboard
    print("\n\n--- Markdown for README leaderboard ---\n")
    parts = []
    for cat in cat_order:
        if cat in results_by_cat:
            results = results_by_cat[cat]
            parts.append(f"{np.mean([r['epe'] for r in results]):.4f}")
        else:
            parts.append("-")

    overall_epe = f"{np.mean(all_epes):.4f}" if all_epes else "-"
    overall_ae = f"{np.mean(all_aes):.4f}" if all_aes else "-"

    print(f"| - | {args.method_name} | learned | {' | '.join(parts)} | **{overall_epe}** | **{overall_ae}** |")


if __name__ == '__main__':
    main()
