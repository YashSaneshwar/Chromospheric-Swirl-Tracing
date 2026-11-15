"""Parameter sweep for the chromospheric swirl detection algorithm.

Place this file beside swirl_tracer.py and run:
    python parameter_sweep.py

Stages
------
Stage 1: Compare Canny smoothing (sigma) and high threshold (low threshold is
         held at a fixed fraction of the high threshold).
Stage 2: Using a selected Stage-1 configuration, compare maximum bridge
         separation and endpoint angle. Because cost terms are normalised by
         fixed reference scales (REF_*), sweeping these hard limits changes
         which candidates are admissible without rescaling candidate costs.
Stages 3-4: Relative cost-weight sweeps at fixed total weight.

Outputs: four 4x4 PNG mosaics, a CSV of summary metrics, and a shortlist TXT.
Inspect the visual mosaics before choosing a configuration, as overly permissive
bridging can inflate path lengths via false merges.
"""
from pathlib import Path
import csv
import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import swirl_tracer as tracker
from skimage.feature import canny

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FITS_FILE = tracker.FITS_FILE
FRAME_INDEX = tracker.FRAME_INDEX
OUTPUT_DIR = tracker.OUTPUT_DIR / 'parameter_sweep'

# Baseline values read from the tracker at import time.
BASE_SIGMA = tracker.CANNY_SIGMA
BASE_HIGH = tracker.CANNY_HIGH
BASE_LOW = tracker.CANNY_LOW
BASE_BRIDGE = tracker.MAX_BRIDGE
BASE_ANGLE = tracker.MAX_ANGLE

# 4 rows x 4 columns per mosaic: one output per parameter combination.
SIGMAS = [0.8, 1.2, 1.6, 2.0]
HIGH_THRESHOLDS = [0.12, 0.17, 0.22, 0.27]
BRIDGE_DISTANCES = [6.0, 9.0, 12.0, 15.0]
BRIDGE_ANGLES = [20.0, 30.0, 40.0, 50.0]

# Set to e.g. (1.2, 0.17) after inspecting the first mosaic.
# None uses the baseline Canny parameters for the bridge sweep.
BRIDGE_SWEEP_CANNY = None

# Weight sweeps: 4x4 each. Keep the sum of the weights equal to the
# baseline sum so that the fixed MAX_BRIDGE_COST remains comparable.
WEIGHT_FACTORS = [0.25, 0.5, 1.0, 2.0]
BASE_WEIGHTS = dict(distance=tracker.W_DISTANCE, angle=tracker.W_ANGLE,
                    curvature=tracker.W_CURVATURE, prediction=tracker.W_PREDICTION)
BASE_WEIGHT_SUM = sum(BASE_WEIGHTS.values())

# For stages 3 and 4, set a specific Canny/bridging pair if desired.
# None uses the baseline Canny, maximum bridge distance, and angle.
WEIGHT_SWEEP_CANNY = None
WEIGHT_SWEEP_BRIDGE = None


def normalized_weights(changes=None):
    """Apply relative weight multipliers while retaining baseline total weight."""
    changes = changes or {}
    weights = {key: value * changes.get(key, 1.0)
               for key, value in BASE_WEIGHTS.items()}
    scale = BASE_WEIGHT_SUM / sum(weights.values())
    return {key: value * scale for key, value in weights.items()}


# Number of longest paths drawn in every cell (e.g. 12 for a cleaner view).
# None draws all paths.
DRAW_LONGEST = None


def run_detector(processed, sigma, high, max_bridge, max_angle, weights=None):
    """Run the tracker algorithm with modified input parameters."""
    weights = normalized_weights() if weights is None else weights
    tracker.W_DISTANCE = float(weights['distance'])
    tracker.W_ANGLE = float(weights['angle'])
    tracker.W_CURVATURE = float(weights['curvature'])
    tracker.W_PREDICTION = float(weights['prediction'])
    tracker.MAX_BRIDGE = float(max_bridge)
    tracker.MAX_ANGLE = float(max_angle)
    low = high * (BASE_LOW / BASE_HIGH)
    edges = canny(processed, sigma=float(sigma),
                  low_threshold=float(low), high_threshold=float(high))
    graph = tracker.build_graph(edges)
    n_edge_pixels = graph.number_of_nodes()
    n_components = nx.number_connected_components(graph)
    bridges = tracker.find_bridges(graph, verbose=False)
    joined = graph.copy()
    for b in bridges:
        joined.add_edge(b['node_A'], b['node_B'],
                        weight=b['distance'], bridge=True, bridge_cost=b['cost'])
    paths = tracker.extract_paths(joined)
    lengths = [sum(float(np.hypot(b[0] - a[0], b[1] - a[1]))
                   for a, b in zip(path[:-1], path[1:])) for path in paths]
    result = dict(edges=edges, paths=paths, bridges=bridges,
                  **{f"w_{k}": v for k, v in weights.items()},
                  sigma=float(sigma), high=float(high), low=float(low),
                  max_bridge=float(max_bridge), max_angle=float(max_angle),
                  n_paths=len(paths), n_bridges=len(bridges),
                  n_fragments=n_components, n_edge_pixels=n_edge_pixels,
                  longest_px=max(lengths, default=0),
                  top5_total_px=sum(sorted(lengths, reverse=True)[:5]),
                  total_length_px=sum(lengths))
    return result


def draw_cell(ax, image, result):
    """Plot image with identical coordinate limits, coloured paths, and dashed gaps."""
    ax.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1)
    paths = result['paths'] if DRAW_LONGEST is None else result['paths'][:DRAW_LONGEST]
    colors = plt.get_cmap('tab20')
    for i, path in enumerate(paths):
        xy = np.array([tracker.xy(p) for p in path])
        ax.plot(xy[:, 0], xy[:, 1], lw=1.15, color=colors(i % 20))
    for bridge in result['bridges']:
        curve = tracker.bezier_bridge(bridge)
        ax.plot(curve[:, 0], curve[:, 1], '--', color='yellow', lw=1.25, alpha=0.85)
    h, w = image.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])


def save_mosaic(image, configs, row_values, col_values, row_key, col_key,
                heading, output):
    fig, axs = plt.subplots(len(row_values), len(col_values),
                            figsize=(4 * len(col_values), 4.35 * len(row_values)),
                            constrained_layout=True, squeeze=False)
    fig.suptitle(heading, fontsize=16, weight='bold')
    for i, row in enumerate(row_values):
        for j, col in enumerate(col_values):
            r = configs[(row, col)]
            ax = axs[i, j]
            draw_cell(ax, image, r)
            ax.set_title(f'{row_key}={row:g} | {col_key}={col:g}\n'
                         f"{r['n_paths']} paths, {r['n_bridges']} bridges, "
                         f"longest={r['longest_px']:.0f}px", fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{row_key}={row:g}', fontsize=10)
            if i == len(row_values) - 1:
                ax.set_xlabel(f'{col_key}={col:g}', fontsize=10)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)


def csv_record(stage, result):
    keys = ('sigma', 'low', 'high', 'max_bridge', 'max_angle', 'n_paths',
            'n_bridges', 'n_fragments', 'n_edge_pixels', 'longest_px',
            'top5_total_px', 'total_length_px',
            'w_distance', 'w_angle', 'w_curvature', 'w_prediction')
    return {'stage': stage, **{k: result[k] for k in keys}}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = tracker.load_frame(FITS_FILE, FRAME_INDEX)
    processed = tracker.preprocess(raw)
    records = []

    # Stage 1: Canny edge detection parameters (bridging kept at baseline).
    edge_results = {}
    print('\nSTAGE 1: Canny parameters')
    for sigma, high in itertools.product(SIGMAS, HIGH_THRESHOLDS):
        print(f'  sigma={sigma:g}, high={high:g}', flush=True)
        r = run_detector(processed, sigma, high, BASE_BRIDGE, BASE_ANGLE)
        edge_results[(sigma, high)] = r
        records.append(csv_record('canny', r))
    save_mosaic(processed, edge_results, SIGMAS, HIGH_THRESHOLDS,
                'sigma', 'high', 'Canny edge-parameter sweep',
                OUTPUT_DIR / '01_canny_mosaic.png')

    # Stage 2: Gap bridging limits (edge detection fixed).
    chosen_sigma, chosen_high = (BRIDGE_SWEEP_CANNY if BRIDGE_SWEEP_CANNY is not None
                                 else (BASE_SIGMA, BASE_HIGH))
    bridge_results = {}
    print('\nSTAGE 2: bridge parameters')
    for dist, angle in itertools.product(BRIDGE_DISTANCES, BRIDGE_ANGLES):
        print(f'  max bridge={dist:g}px, max angle={angle:g}deg', flush=True)
        r = run_detector(processed, chosen_sigma, chosen_high, dist, angle)
        bridge_results[(dist, angle)] = r
        records.append(csv_record('bridge', r))
    save_mosaic(processed, bridge_results, BRIDGE_DISTANCES, BRIDGE_ANGLES,
                'dist', 'angle',
                f'Gap-bridging sweep (Canny sigma={chosen_sigma:g}, high={chosen_high:g})',
                OUTPUT_DIR / '02_bridge_mosaic.png')

    # Stages 3 & 4: Relative bridge-cost weights.
    weight_sigma, weight_high = (WEIGHT_SWEEP_CANNY if WEIGHT_SWEEP_CANNY is not None
                                 else (BASE_SIGMA, BASE_HIGH))
    weight_bridge, weight_angle = (WEIGHT_SWEEP_BRIDGE if WEIGHT_SWEEP_BRIDGE is not None
                                   else (BASE_BRIDGE, BASE_ANGLE))
    weight_pairs = [('distance', 'angle'), ('curvature', 'prediction')]
    for stage_number, (row_term, col_term) in enumerate(weight_pairs, start=3):
        results = {}
        print(f'\nSTAGE {stage_number}: {row_term} versus {col_term} weights')
        for row_factor, col_factor in itertools.product(WEIGHT_FACTORS, WEIGHT_FACTORS):
            weights = normalized_weights({row_term: row_factor, col_term: col_factor})
            print(f'  {row_term} factor={row_factor:g}, {col_term} factor={col_factor:g}, '
                  f'weights={weights}', flush=True)
            r = run_detector(processed, weight_sigma, weight_high,
                             weight_bridge, weight_angle, weights=weights)
            results[(row_factor, col_factor)] = r
            records.append(csv_record(f'weights_{row_term}_{col_term}', r))
        save_mosaic(processed, results, WEIGHT_FACTORS, WEIGHT_FACTORS,
                    f'{row_term} x', f'{col_term} x',
                    f'Bridge-cost weights: {row_term} vs {col_term} '
                    '(factors relative to baseline; total weight fixed)',
                    OUTPUT_DIR / f'0{stage_number}_weights_{row_term}_{col_term}.png')

    csv_path = OUTPUT_DIR / 'sweep_metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    candidates = sorted(records, key=lambda r: r['top5_total_px'], reverse=True)[:8]
    with (OUTPUT_DIR / 'inspection_shortlist.txt').open('w') as f:
        f.write('Highest summed length of five longest detected paths.\n')
        f.write('Note: overly permissive bridging can inflate this metric via false merges.\n\n')
        for r in candidates:
            f.write(str(r) + '\n')
    print('\nSaved:', OUTPUT_DIR)
    print('  01_canny_mosaic.png')
    print('  02_bridge_mosaic.png')
    print('  03_weights_distance_angle.png')
    print('  04_weights_curvature_prediction.png')
    print('  sweep_metrics.csv')
    print('  inspection_shortlist.txt')


if __name__ == '__main__':
    main()
