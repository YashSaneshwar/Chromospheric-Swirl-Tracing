"""Conservative reconstruction of spiral arms from centre-associated segments.

Place beside swirl_centre_voting.py and swirl_tracer.py,
then run:
    python swirl_arm_reconstruction.py

Overview
--------
This module orders member pixels from swirl_centre_voting.py into monotonic
polar segments, evaluates short geometry-consistent bridges between segment
endpoints, and assembles reconstructed spiral arms without interpolating
synthetic pixels across gaps. It serves as a strict, short-gap counterpart to
the log-spiral linker in swirl_centre_voting.py.

Output: outputs/swirl_centres/arm_reconstruction/centre_XX_arms.png and *_arms.npz.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import median_filter

import swirl_centre_voting as scv

# -----------------------------------------------------------------------------
# Configuration: all distances in pixels; angles in degrees.
# -----------------------------------------------------------------------------
OUT_DIR = scv.OUTPUT_DIR / 'arm_reconstruction'
MAX_GAP = 15.0              # maximum bridged endpoint separation (px)
END_WINDOW = 7              # local end geometry and pitch window (points)
MAX_TANGENT_ERROR = 40.0    # maximum endpoint-to-chord angle (deg)
MAX_PITCH_ERROR = 24.0      # maximum local polar pitch mismatch (deg)
MAX_PREDICTION_ERROR = 5.0  # maximum discrepancy from polar continuation (px)
MIN_SEGMENT = 12
MAX_JOIN_COST = 2.3
MIN_CENTRE_RADIUS = 5.0
MAX_THETA_REVERSAL = 1.0    # deg per step tolerated before a path counts as reversing
USE_SMOOTHED_PIXELS = True


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros(2)


def wrap(a):
    """Wrapped difference between two angles (radians)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def obj_array(items):
    """Build a 1-D object array regardless of item shapes."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def order_segment(G, nodes, index, xy, centre):
    """Extract the principal path through a member component, trimmed to its
    longest stretch that is monotonic in position angle about the centre.
    """
    sub = G.subgraph(nodes)
    ends = [n for n in sub if sub.degree(n) == 1]
    if len(ends) >= 2:
        def farthest(start, choices):
            d = nx.single_source_dijkstra_path_length(sub, start, weight='weight')
            return max(choices, key=lambda k: d.get(k, -1))
        a = farthest(ends[0], ends)
        b = farthest(a, ends)
    else:
        start = next(iter(sub))
        d = nx.single_source_dijkstra_path_length(sub, start, weight='weight')
        a = max(d, key=d.get)
        d = nx.single_source_dijkstra_path_length(sub, a, weight='weight')
        b = max(d, key=d.get)
    seq = nx.shortest_path(sub, a, b, weight='weight')
    ids = np.array([index[n] for n in seq], int)

    v = xy[ids] - centre
    theta = median_filter(np.unwrap(np.arctan2(v[:, 1], v[:, 0])), size=5, mode='nearest')
    sgn = 1.0 if theta[-1] >= theta[0] else -1.0
    i0, i1 = scv._longest_true_run(np.diff(theta) * sgn > -np.radians(MAX_THETA_REVERSAL))
    ids = ids[i0:i1 + 1]
    excluded = len(nodes) - len(ids)
    return (ids, excluded) if len(ids) >= MIN_SEGMENT else (None, excluded)


def get_segments(G, g, labels, centre):
    """Turn disconnected membership masks into ordered pixel sequences."""
    p = g.Ps if USE_SMOOTHED_PIXELS else g.P
    segments = []
    for label in np.unique(labels[labels >= 0]):
        nodes = [g.nodes[i] for i in np.where(labels == label)[0]]
        ids, excluded = order_segment(G, nodes, g.index, p, centre)
        if ids is None:
            continue
        segments.append(dict(id=len(segments), source_label=int(label), ids=ids,
                             points=p[ids], excluded=int(excluded)))
    return segments


def end_info(segment, side, centre):
    """Estimate endpoint tangent and local spiral pitch using nearby arc pixels."""
    pts = segment['points']
    part = pts[:END_WINDOW] if side == 0 else pts[-END_WINDOW:]
    endpoint = pts[0] if side == 0 else pts[-1]
    q = part - part.mean(axis=0)
    _, vecs = np.linalg.eigh(q.T @ q)
    t = unit(vecs[:, -1])
    outward = endpoint - part.mean(axis=0)
    if np.dot(t, outward) < 0:
        t = -t
    radii = np.linalg.norm(part - centre, axis=1)
    theta = np.unwrap(np.arctan2(part[:, 1] - centre[1], part[:, 0] - centre[0]))
    good = np.all(radii > MIN_CENTRE_RADIUS) and np.ptp(theta) > 0.025
    if good:
        slope = np.polyfit(theta, np.log(radii), 1)[0]
        good = np.isfinite(slope) and abs(slope) < 3.0
    else:
        slope = np.nan
    return dict(point=endpoint, tangent=t, slope=slope if good else np.nan,
                radius=np.linalg.norm(endpoint - centre), segment=segment['id'],
                side=side)


def extrapolation_error(A, B, centre):
    """Check how well a local log spiral predicts the opposite endpoint."""
    def predict(src, dst):
        theta_s = np.arctan2(src['point'][1]-centre[1], src['point'][0]-centre[0])
        theta_d = np.arctan2(dst['point'][1]-centre[1], dst['point'][0]-centre[0])
        dt = wrap(theta_d-theta_s)
        if not np.isfinite(src['slope']):
            return np.inf
        rhat = src['radius'] * np.exp(np.clip(src['slope']*dt, -3, 3))
        return abs(rhat - dst['radius'])
    return max(predict(A, B), predict(B, A))


def candidate_bridges(segments, centre):
    """Score end-to-end joins; reject tangent, pitch, and radius discontinuities."""
    ends = [end_info(seg, side, centre) for seg in segments for side in (0, 1)]
    candidates = []
    for i, A in enumerate(ends):
        for B in ends[i+1:]:
            if A['segment'] == B['segment']:
                continue
            vec = B['point']-A['point']
            dist = np.linalg.norm(vec)
            if not (0.5 < dist <= MAX_GAP):
                continue
            d = vec / dist
            angle_a = np.degrees(np.arccos(np.clip(np.dot(A['tangent'], d), -1, 1)))
            angle_b = np.degrees(np.arccos(np.clip(np.dot(B['tangent'], -d), -1, 1)))
            if max(angle_a, angle_b) > MAX_TANGENT_ERROR:
                continue
            if not (np.isfinite(A['slope']) and np.isfinite(B['slope'])):
                continue
            pa, pb = np.degrees(np.arctan(A['slope'])), np.degrees(np.arctan(B['slope']))
            pitch_delta = abs(pa-pb)
            if pitch_delta > MAX_PITCH_ERROR:
                continue
            pred_error = extrapolation_error(A, B, centre)
            if pred_error > MAX_PREDICTION_ERROR:
                continue
            cost = (0.55*dist/MAX_GAP + 0.7*(angle_a+angle_b)/(2*MAX_TANGENT_ERROR)
                    + 0.65*pitch_delta/MAX_PITCH_ERROR
                    + 0.8*pred_error/MAX_PREDICTION_ERROR)
            if cost <= MAX_JOIN_COST:
                candidates.append(dict(A=A, B=B, cost=cost, dist=dist,
                                       pitch_delta=pitch_delta, prediction=pred_error))
    return sorted(candidates, key=lambda c: c['cost'])


def select_bridges(segments, candidates):
    """Greedy low-cost joining: each end joins once; no cycles of segments."""
    parent = list(range(len(segments)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    used, accepted = set(), []
    for item in candidates:
        A, B = item['A'], item['B']
        key_a, key_b = (A['segment'], A['side']), (B['segment'], B['side'])
        ra, rb = root(A['segment']), root(B['segment'])
        if key_a in used or key_b in used or ra == rb:
            continue
        accepted.append(item)
        used.update((key_a, key_b))
        parent[rb] = ra
    return accepted


def assemble_arms(segments, bridges):
    """Walk the endpoint-connection graph, reversing segments where required."""
    conn = {}
    for b in bridges:
        a, z = b['A'], b['B']
        ka, kb = (a['segment'], a['side']), (z['segment'], z['side'])
        conn[ka] = (kb, b)
        conn[kb] = (ka, b)
    visited, arms = set(), []
    for s in range(len(segments)):
        if s in visited:
            continue
        current = s
        seen = set()
        while True:
            seen.add(current)
            links = [(side, conn[(current, side)][0]) for side in (0, 1)
                     if (current, side) in conn]
            available = [(side, nb) for side, nb in links if nb[0] not in seen]
            if not available:
                break
            side, nb = available[0]
            current = nb[0]
        start = current
        free = [side for side in (0, 1) if (start, side) not in conn]
        entrance = free[0] if free else 0
        pieces, gaps, members = [], [], []
        while current not in visited:
            visited.add(current)
            members.append(current)
            piece = segments[current]['points']
            pieces.append(piece if entrance == 0 else piece[::-1])
            exit_side = 1-entrance
            edge = conn.get((current, exit_side))
            if edge is None:
                break
            (next_seg, next_side), bridge = edge
            gaps.append(bridge)
            current, entrance = next_seg, next_side
        arms.append(dict(pieces=pieces, gaps=gaps, segments=len(pieces), members=members,
                         observed_pixels=sum(len(p) for p in pieces)))
    return sorted(arms, key=lambda a: a['observed_pixels'], reverse=True)


def plot_diagnostics(processed, centre, segments, candidates, accepted, arms, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    colors = plt.get_cmap('tab20')
    titles = ['Original swirl-member segments', 'All admissible endpoint joins',
              'Accepted joins', 'Reconstructed spiral arms']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.imshow(processed, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.plot(*centre, '+', color='red', ms=13, mew=2)
        ax.set(xlim=(-0.5, processed.shape[1]-0.5),
               ylim=(-0.5, processed.shape[0]-0.5), aspect='equal',
               xlabel='X [pixel]', ylabel='Y [pixel]')
    for idx, seg in enumerate(segments):
        p = seg['points']
        for ax in axes.flat[:3]:
            ax.plot(p[:, 0], p[:, 1], color=colors(idx % 20), lw=1.7)
    for ax, links, col, opacity in [(axes[0, 1], candidates, 'yellow', 0.4),
                                    (axes[1, 0], accepted, 'lime', 0.9)]:
        for b in links:
            a, z = b['A']['point'], b['B']['point']
            ax.plot([a[0], z[0]], [a[1], z[1]], '--', color=col, alpha=opacity, lw=1)
    for k, arm in enumerate(arms):
        col = colors(k % 20)
        multi = arm['segments'] > 1
        label = f"arm {k}: {arm['segments']} segs, {arm['observed_pixels']} px" if multi else None
        for n, piece in enumerate(arm['pieces']):
            axes[1, 1].plot(piece[:, 0], piece[:, 1], color=col, lw=2.2 if multi else 1.2,
                            label=label if n == 0 else None)
        for b in arm['gaps']:
            a, z = b['A']['point'], b['B']['point']
            axes[1, 1].plot([a[0], z[0]], [a[1], z[1]], '--', color=col, lw=2)
    if any(a['segments'] > 1 for a in arms):
        axes[1, 1].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Centre ({centre[0]:.1f}, {centre[1]:.1f}) px | '
                 f'{len(segments)} segments, {len(candidates)} possible joins, '
                 f'{len(accepted)} accepted, {len(arms)} arms', fontsize=15)
    fig.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close(fig)


def reconstruct(result, out_dir=OUT_DIR, verbose=True):
    """Run the conservative reconstruction on an scv.run() result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g, G, processed = result['geometry'], result['G'], result['processed']
    if set(G.nodes) != set(g.nodes):
        raise RuntimeError('Graph and per-pixel geometry do not match.')
    summaries = []
    for k, centre_result in enumerate(result['results']):
        centre = centre_result['centre']
        segments = get_segments(G, g, centre_result['labels'], centre)
        if not segments:
            if verbose:
                print(f'Centre {k}: no usable member segments')
            continue
        candidates = candidate_bridges(segments, centre)
        selected = select_bridges(segments, candidates)
        arms = assemble_arms(segments, selected)
        prefix = out_dir / f'centre_{k:02d}'
        plot_diagnostics(processed, centre, segments, candidates, selected, arms,
                         prefix.with_name(prefix.name + '_arms.png'))
        np.savez_compressed(
            prefix.with_name(prefix.name + '_arms.npz'),
            centre=centre,
            segments=obj_array([s['points'] for s in segments]),
            segment_source_labels=np.array([s['source_label'] for s in segments], int),
            segment_excluded_pixels=np.array([s['excluded'] for s in segments], int),
            arm_pieces=obj_array([obj_array(a['pieces']) for a in arms]),
            arm_members=obj_array([np.array(a['members'], int) for a in arms]),
            bridges=np.array([[*b['A']['point'], *b['B']['point'], b['cost']]
                              for b in selected]).reshape(-1, 5),
            summary=np.array([[a['segments'], a['observed_pixels']] for a in arms], int),
            summary_columns=np.array(['n_segments', 'observed_pixels']))
        if verbose:
            print(f'Centre {k}: {len(segments)} segments, {len(candidates)} candidate joins, '
                  f'{len(selected)} accepted -> {len(arms)} arms')
            for a in arms:
                if a['segments'] > 1:
                    print(f"   arm: {a['segments']} segments, {a['observed_pixels']} px")
        summaries.append(dict(centre=k, segments=segments, arms=arms,
                              candidates=candidates, selected=selected))
    if verbose:
        print('Saved:', out_dir)
    return summaries


def main():
    raw = scv.tracker.load_frame(scv.FITS_FILE, scv.FRAME_INDEX)
    result = scv.run(raw)
    reconstruct(result)


if __name__ == '__main__':
    main()