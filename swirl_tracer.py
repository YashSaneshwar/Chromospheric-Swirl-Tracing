"""
Detects, traces, and reconnects fragmented chromospheric loops, fibrils, and swirl arms 
from high-resolution 2D solar images (such as SST/CRISP Ca II 8542 Å) into continuous, smooth paths.

---

### Step-by-Step Pipeline

1. **Image Preprocessing & Edge Detection**
   - Applies **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) and median filtering 
   to sharpen faint, low-contrast dark features.
   - Runs **Canny edge detection** to outline fibril and loop boundaries. It detects peaks in intensity
   change and assigns it a "node".

2. **Graph Construction**
   - Skeletonises detected edges down to 1-pixel-wide lines.
   - Builds a network graph using **$m$-adjacency** (mixed pixel neighbourhood) so each edge pixel 
     becomes a connected node.

3. **Endpoint Tangent & Curvature Analysis**
   - Identifies the loose ends (endpoints) of disconnected edge fragments.
   - Uses Principal Component Analysis (PCA) over a small local window to calculate each endpoint’s 
     outward pointing direction (tangent) and curvature.

4. **Intelligent Geometric Gap Bridging**
   - Searches for nearby endpoints within a distance threshold (`MIN_BRIDGE` to `MAX_BRIDGE`).
   - Calculates a combined compatibility cost based on:
     - **Distance** between endpoints
     - **Angle alignment** (whether they point towards each other)
     - **Curvature continuity** (whether the fragments curve in the same direction this is for swirls)
     - **Path prediction error**
   - Adds bridge links between fragments that meet strict compatibility thresholds.

5. **Path Extraction & Export**
   - Extracts the longest, continuous principal backbone paths from the bridged graph.
   - Exports the extracted coordinate arrays to `outputs/swirl_graph_detection.npz`.
   - Generates a 4-panel diagnostic plot (`outputs/swirl_graph_detection.png`):
     - Original Image
     - Preprocessed Image (CLAHE + Median)
     - Detected Edge Fragments
     - Final Traced Paths with Bézier-interpolated gap bridges

6. **Optional Swirl Analysis** *(toggled via `RUN_SWIRL_CENTRES`)*
   - Automatically passes the traced curves to other scripts (to pinpoint the swirl vortex centre and reconstruct spiral arms(future work)).
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from skimage import exposure
from skimage.feature import canny
from skimage.morphology import skeletonize

# -----------------------------------------------------------------------------
# Configuration (RUNNIG ALREADY PROCESSED FILE THAT WE GET FROM THE CONVERSION CODE)
# -----------------------------------------------------------------------------
FITS_FILE = Path(__file__).resolve().parent / 'data' / 'ca8542_224_swirl_pos22.fits'
FRAME_INDEX = 0
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
CLAHE_CLIP, MEDIAN_SIZE = 0.03, 3
CANNY_SIGMA, CANNY_LOW, CANNY_HIGH = 1.2, 0.08, 0.22
SKELETONIZE_EDGES = True
MIN_FRAGMENT_PIXELS, MIN_FINAL_PATH_PIXELS = 6, 20
TANGENT_WINDOW, CURVATURE_CLIP = 8, 0.50

# Hard rejection limits (candidates outside these are discarded).
MIN_BRIDGE, MAX_BRIDGE = 4.0, 12.0
MAX_ANGLE, MAX_CURVATURE_DIFF, MAX_PREDICTION_FRAC = 35.0, 0.12, 0.75

# Fixed reference scales for the cost terms. Kept separate from the hard
# limits above so that sweeping a limit does not rescale every candidate's cost.
REF_BRIDGE, REF_ANGLE = 12.0, 35.0
REF_CURVATURE_DIFF, REF_PREDICTION_FRAC = 0.12, 0.75

W_DISTANCE, W_ANGLE, W_CURVATURE, W_PREDICTION = 0.75, 2.0, 1.0, 2.0
MAX_BRIDGE_COST = 4.0
NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

# -----------------------------------------------------------------------------
# Optional swirl analysis (runs after the tracing in main())
# -----------------------------------------------------------------------------
RUN_SWIRL_CENTRES = False          # swirl_centre_voting.py: finds the center of a swirl based on extension of lines from each pixel of a loop and seeing where the lines converge. 
RUN_ARM_RECONSTRUCTION = False     # swirl_arm_reconstruction.py (needs RUN_SWIRL_CENTRES)
SWIRL_OUTPUT_DIR = OUTPUT_DIR / 'swirl_centres'
# Override any swirl_centre_voting setting by name, e.g.
# {'N_NULL': 10, 'R_MIN': 8.0, 'R_MAX': 50.0, 'MAX_GAP_PX': 20.0}
SWIRL_SETTINGS = {}


def load_frame(filename, frame_index=0):
    data = np.squeeze(np.asarray(fits.getdata(filename), dtype=float))
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[frame_index]
    raise ValueError(f'Expected 2-D image or 3-D cube. Got shape {data.shape}')


def preprocess(image):
    """Replace invalid pixels, robustly scale intensities, then enhance edges."""
    img = np.asarray(image, dtype=float).copy()
    finite = np.isfinite(img)
    if not finite.any():
        raise ValueError('Image contains no finite pixels.')
    img[~finite] = np.nanmedian(img[finite])
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-12)
    return median_filter(exposure.equalize_adapthist(img, clip_limit=CLAHE_CLIP), size=(MEDIAN_SIZE, MEDIAN_SIZE))


def build_graph(edges):
    """Treat each (thinned) edge pixel as a node using mixed adjacency.

    Horizontal/vertical neighbours are always linked (length 1). A diagonal
    neighbour (length sqrt(2)) is linked only when neither of the two pixels
    sharing a side with both is itself an edge pixel. This prevents corner
    triangles, so a curve's last pixel has degree 1 even on a stair step.
    Very small connected fragments are discarded.
    """
    edges = np.asarray(edges, dtype=bool)
    if SKELETONIZE_EDGES:
        edges = skeletonize(edges)
    G = nx.Graph()
    pixels = set(zip(*np.nonzero(edges)))
    G.add_nodes_from(pixels)
    for r, c in pixels:
        for dr, dc in NEIGHBOURS:
            other = (r + dr, c + dc)
            if other not in pixels or (r, c) >= other:
                continue
            if dr and dc and ((r + dr, c) in pixels or (r, c + dc) in pixels):
                continue  # a 4-connected route already joins these pixels
            G.add_edge((r, c), other, weight=np.hypot(dr, dc), bridge=False)
    for nodes in list(nx.connected_components(G)):
        if len(nodes) < MIN_FRAGMENT_PIXELS:
            G.remove_nodes_from(nodes)
    return G


def xy(node):
    return np.array([float(node[1]), float(node[0])])


def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def endpoint_geometry(fragment, endpoint):
    """Estimate an outward tangent (PCA) and signed curvature (quadratic fit).

    The tangent is oriented away from the fragment towards the missing region.
    Positive curvature means the curve, followed outward, turns towards the
    left-hand normal (-t_y, t_x). Pixels nearest the endpoint receive more
    weight in the curvature fit.
    """
    local_nodes = list(nx.single_source_shortest_path_length(fragment, endpoint, cutoff=TANGENT_WINDOW))
    if len(local_nodes) < 4:
        return None
    points = np.array([xy(node) for node in local_nodes])
    p = xy(endpoint)
    centre = points.mean(axis=0)
    vals, vecs = np.linalg.eigh((points - centre).T @ (points - centre))
    tangent = vecs[:, np.argmax(vals)]
    tangent /= np.linalg.norm(tangent) + 1e-12
    if np.dot(tangent, p - centre) < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    xx, yy = (points - p) @ tangent, (points - p) @ normal
    mask = xx <= 1.0
    curvature = 0.0
    if mask.sum() >= 5 and np.ptp(xx[mask]) >= 2:
        try:
            a, b, _ = np.polyfit(xx[mask], yy[mask], 2, w=1 / (1 + np.abs(xx[mask])))
            curvature = float(np.clip(2 * a / (1 + b**2)**1.5, -CURVATURE_CLIP, CURVATURE_CLIP))
        except Exception:
            pass
    return dict(node=endpoint, position=p, tangent=tangent, curvature=curvature)


def get_endpoints(G):
    components = list(nx.connected_components(G))
    endpoints = []
    for i, nodes in enumerate(components):
        fragment = G.subgraph(nodes)
        for node in fragment:
            if fragment.degree(node) == 1:
                info = endpoint_geometry(fragment, node)
                if info is not None:
                    info['component'] = i
                    endpoints.append(info)
    return components, endpoints


def predict_from_curvature(p, tangent, curvature, distance):
    """Extrapolate an endpoint as a line or constant-curvature circular arc."""
    tangent = np.asarray(tangent).copy()
    tangent /= np.linalg.norm(tangent) + 1e-12
    if abs(curvature) < 1e-4:
        return p + distance * tangent
    k, s = curvature, distance
    return p + (np.sin(k * s) / k) * tangent + ((1 - np.cos(k * s)) / k) * np.array([-tangent[1], tangent[0]])


def evaluate_bridge(A, B):
    """Score a candidate bridge using separation, tangents, and curvature.

    Reject bridges outside hard limits; lower cost denotes a better match.

    Curvature: both tangents point outward, i.e. towards each other across
    the gap. Seen from A the arc is traversed one way, seen from B the other,
    so one smooth arc gives kA ~ -kB. |kA + kB| is therefore the mismatch;
    an S-bend (kA ~ +kB, both non-zero) is penalised.
    """
    pA, pB = A['position'], B['position']
    delta = pB - pA
    distance = np.linalg.norm(delta)
    if not MIN_BRIDGE <= distance <= MAX_BRIDGE:
        return None
    direction = delta / (distance + 1e-12)
    angle_A = angle_between(A['tangent'], direction)
    angle_B = angle_between(B['tangent'], -direction)
    if max(angle_A, angle_B) > MAX_ANGLE:
        return None
    kA, kB = A['curvature'], B['curvature']
    kdiff = abs(kA + kB)
    if kdiff > MAX_CURVATURE_DIFF:
        return None
    error_A = np.linalg.norm(predict_from_curvature(pA, A['tangent'], kA, distance) - pB)
    error_B = np.linalg.norm(predict_from_curvature(pB, B['tangent'], kB, distance) - pA)
    prediction_error = 0.5 * (error_A + error_B)
    prediction_fraction = prediction_error / distance
    if prediction_fraction > MAX_PREDICTION_FRAC:
        return None
    cost = (W_DISTANCE * distance / REF_BRIDGE
            + W_ANGLE * 0.5 * (angle_A + angle_B) / REF_ANGLE
            + W_CURVATURE * kdiff / REF_CURVATURE_DIFF
            + W_PREDICTION * prediction_fraction / REF_PREDICTION_FRAC)
    if cost > MAX_BRIDGE_COST:
        return None
    return dict(node_A=A['node'], node_B=B['node'], component_A=A['component'],
                component_B=B['component'], distance=distance, angle_A=angle_A,
                angle_B=angle_B, curvature_A=kA, curvature_B=kB,
                prediction_error=prediction_error, cost=cost,
                tangent_A=A['tangent'], tangent_B=B['tangent'])


class UnionFind:
    """Keep accepted bridges from forming artificial component-level cycles."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def find_bridges(G, verbose=True):
    """Choose low-cost, non-conflicting endpoint pairs within MAX_BRIDGE.

    Each endpoint can be used once, and components cannot be re-joined after
    they have already been connected by an accepted bridge.
    """
    components, endpoints = get_endpoints(G)
    if verbose:
        print(f'Detected {len(components)} edge fragments')
        print(f'Found {len(endpoints)} usable endpoints')
    if len(endpoints) < 2:
        return []
    pairs = cKDTree([e['position'] for e in endpoints]).query_pairs(MAX_BRIDGE)
    candidates = []
    for i, j in pairs:
        if endpoints[i]['component'] == endpoints[j]['component']:
            continue
        bridge = evaluate_bridge(endpoints[i], endpoints[j])
        if bridge is not None:
            candidates.append(bridge)
    candidates.sort(key=lambda b: b['cost'])
    used, groups, accepted = set(), UnionFind(len(components)), []
    for b in candidates:
        A, B = b['node_A'], b['node_B']
        ca, cb = b['component_A'], b['component_B']
        if A in used or B in used or groups.find(ca) == groups.find(cb):
            continue
        accepted.append(b)
        used.update((A, B))
        groups.union(ca, cb)
    return accepted


def extract_paths(G):
    """Extract one representative principal path per connected component.

    Dijkstra distances select the most distant endpoint pair (graph diameter).
    """
    paths = []
    for nodes in nx.connected_components(G):
        if len(nodes) < MIN_FINAL_PATH_PIXELS:
            continue
        fragment = G.subgraph(nodes)
        endpoints = [n for n in fragment if fragment.degree(n) == 1]

        def farthest(source, candidates=None):
            lengths = nx.single_source_dijkstra_path_length(fragment, source, weight='weight')
            return max((n for n in (candidates if candidates is not None else lengths) if n in lengths), key=lengths.get)

        if len(endpoints) >= 2:
            A = farthest(endpoints[0], endpoints)
            B = farthest(A, endpoints)
        else:
            A = farthest(next(iter(fragment)))
            B = farthest(A)
        path = nx.shortest_path(fragment, A, B, weight='weight')
        if len(path) >= MIN_FINAL_PATH_PIXELS:
            paths.append(path)
    return sorted(paths, key=len, reverse=True)


def bezier_bridge(b, n=30):
    """Generate a smooth dashed display curve for an accepted gap only."""
    p0, p3 = xy(b['node_A']), xy(b['node_B'])
    scale = 0.35 * np.linalg.norm(p3 - p0)
    p1 = p0 + scale * b['tangent_A']
    p2 = p3 + scale * b['tangent_B']
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def run_swirl_analysis(raw):
    """Run the swirl-centre voting (and optionally arm reconstruction) on raw.

    swirl_centre_voting imports this tracker by module name. When this file
    is executed as a script it is registered under that name first, so the
    swirl code uses the exact settings in this file.
    """
    if not RUN_SWIRL_CENTRES:
        if RUN_ARM_RECONSTRUCTION:
            print('RUN_ARM_RECONSTRUCTION ignored: it requires RUN_SWIRL_CENTRES = True')
        return None
    sys.modules.setdefault(Path(__file__).stem, sys.modules[__name__])
    try:
        import swirl_centre_voting as scv
    except ModuleNotFoundError as err:
        if err.name == 'swirl_centre_voting':
            raise ModuleNotFoundError('RUN_SWIRL_CENTRES = True requires swirl_centre_voting.py '
                                      'in the same folder as this script') from err
        raise
    for name, value in SWIRL_SETTINGS.items():
        if not hasattr(scv, name):
            raise AttributeError(f'SWIRL_SETTINGS: swirl_centre_voting has no setting {name!r}')
        setattr(scv, name, value)

    print('\n=== Swirl centre detection ===')
    result = scv.run(raw, out_dir=SWIRL_OUTPUT_DIR)
    if not result['results']:
        print('No swirl centre candidates found.')
    print('Saved:', SWIRL_OUTPUT_DIR)

    if RUN_ARM_RECONSTRUCTION and result['results']:
        try:
            import swirl_arm_reconstruction as sar
        except ModuleNotFoundError as err:
            if err.name == 'swirl_arm_reconstruction':
                raise ModuleNotFoundError('RUN_ARM_RECONSTRUCTION = True requires '
                                          'swirl_arm_reconstruction.py in the same folder') from err
            raise
        print('\n=== Conservative arm reconstruction ===')
        sar.reconstruct(result, out_dir=SWIRL_OUTPUT_DIR / 'arm_reconstruction')
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_frame(FITS_FILE, FRAME_INDEX)
    processed = preprocess(raw)
    edges = canny(processed, sigma=CANNY_SIGMA, low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    G = build_graph(edges)
    print(f'\nBefore gap bridging\nGraph pixels: {G.number_of_nodes()}\nFragments: {nx.number_connected_components(G)}')
    bridges = find_bridges(G)
    print(f'\nAccepted bridges: {len(bridges)}')
    for i, b in enumerate(bridges, 1):
        print(f"{i:3d}  distance={b['distance']:.2f} px   "
              f"angles={b['angle_A']:.1f},{b['angle_B']:.1f} deg   "
              f"kappa={b['curvature_A']:.4f},{b['curvature_B']:.4f} px^-1   "
              f"prediction error={b['prediction_error']:.2f} px   cost={b['cost']:.3f}")
    G_final = G.copy()
    for b in bridges:
        G_final.add_edge(b['node_A'], b['node_B'], weight=b['distance'], bridge=True, bridge_cost=b['cost'])
    paths = extract_paths(G_final)
    print(f'\nFinal detected structures: {len(paths)}')
    arrays = [np.array([[c, r] for r, c in path]) for path in paths]
    np.savez_compressed(OUTPUT_DIR / 'swirl_graph_detection.npz', raw=raw, processed=processed,
                        edges=edges, paths=np.array(arrays, dtype=object))

    # Diagnostic plot: keep identical physical image coordinates across panels.
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for ax, img, title in zip(axes.flat, (raw, processed, processed, processed),
                              ('Original image', 'CLAHE + median', 'Detected edge fragments',
                               f'Final graph tracing: {len(paths)} paths, {len(bridges)} gap bridges')):
        ax.imshow(img, cmap='gray', origin='lower')
        ax.set(title=title, xlabel='X [pixel]', ylabel='Y [pixel]')
    rows, cols = np.nonzero(edges)
    axes[1, 0].scatter(cols, rows, s=1)

    for path in paths:
        points = np.array([xy(node) for node in path])
        axes[1, 1].plot(points[:, 0], points[:, 1], linewidth=1.2)
    for b in bridges:
        curve = bezier_bridge(b)
        axes[1, 1].plot(curve[:, 0], curve[:, 1], '--', linewidth=2)
        A, B = xy(b['node_A']), xy(b['node_B'])
        axes[1, 1].plot([A[0], B[0]], [A[1], B[1]], 'o', markersize=3)
    height, width = raw.shape
    for ax in axes.flat:
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal', adjustable='box')

    plt.savefig(OUTPUT_DIR / 'swirl_graph_detection.png', dpi=200, bbox_inches='tight')

    run_swirl_analysis(raw)
    plt.show()


if __name__ == '__main__':
    main()
