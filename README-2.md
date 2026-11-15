# Chromospheric Swirl & Fibril Tracing

Graph-based tools for tracing chromospheric fibrils and loops, and for locating spiral (swirl) structures, in high-resolution solar images such as SST/CRISP Ca II 8542 Å.

## Overview

Chromospheric loops, fibrils and swirls appear as fragmented dark features, which makes them difficult to trace. This repository provides a three-stage pipeline:

1. **Graph tracing:** enhances local contrast (CLAHE), extracts skeletonised Canny edges with mixed adjacency ($m$-adjacency), and bridges geometrically compatible gaps using endpoint-tangent and signed-curvature matching.
2. **Centre voting:** estimates swirl rotation centres and global pitch angles by casting curvature-weighted normal votes from traced edge pixels, refines them by weighted least squares, and tests each detection against randomised-fragment null models.
3. **Arm reconstruction (experimental):** links centre-associated segments into spiral arms, using either a permissive logarithmic-spiral model (`swirl_centre_voting.py`) or a strict local-geometry linker that does not extrapolate across wide gaps (`swirl_arm_reconstruction.py`).

---

## Installation

Requires Python 3.10 or later.

```bash
git clone <repository-url>
cd <repository-folder>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick start (demo)

A small example cube, `data/ca8542_224.000.sav`, is included so you can run the pipeline straight away.

**Step 1: convert the IDL `.sav` cube to FITS.** Skip this step if you already have a FITS file.

```bash
python sav_to_fits.py
```

With `PLOT_MODE = True` the script opens an interactive viewer so you can choose a clean spectral position. Set `PLOT_MODE = False` and run it again to write the cropped FITS cube and NPZ archive to `data/`.

**Step 2: trace fibrils, loops and swirls.**

```bash
python swirl_tracer.py
```

This runs the graph tracer and saves diagnostic plots and path arrays to `outputs/`. The default parameters are tuned for the example data; to choose parameters for your own data, run the parameter sweep first (see *Tuning parameters* below).

**Step 3: locate swirl centres.** In `swirl_tracer.py`, set:

```python
RUN_SWIRL_CENTRES = True
RUN_ARM_RECONSTRUCTION = False   # experimental; requires RUN_SWIRL_CENTRES = True
```

and run `swirl_tracer.py` again. Arm reconstruction is off by default because it is still experimental.

Each stage can also be run on its own:

```bash
python parameter_sweep.py            # 4x4 sweeps of Canny thresholds, bridge limits and cost weights
python swirl_centre_voting.py        # swirl centres, bootstrap and null tests, log-spiral arm linking
python swirl_arm_reconstruction.py   # conservative short-gap arm reconstruction (experimental)
```

An illustrated step-by-step guide is in [`docs/swirl_detection_guide.pdf`](docs/swirl_detection_guide.pdf).

---

## Working with your own data

### Option A: an IDL `.sav` spectral cube

1. In `sav_to_fits.py`, set `SAV_FILE` to your file, and set `SWIRL_X`, `SWIRL_Y` and `CROP_HALF` to define the field of view.
2. Keep `PLOT_MODE = True` and run the script to scan through spectral positions (`PLOT_STYLE = "slider"` or `"grid"`) and check for frame dropouts.
3. Once you have chosen a clean spectral position (`SPECTRAL_POS`), set `PLOT_MODE = False` and run it again. This writes a cropped 3-D FITS cube and a single-frame 2-D FITS file to `data/`.
4. Copy the printed `FITS_FILE` and `FRAME_INDEX` values into `swirl_tracer.py`.

### Option B: an existing 2-D or 3-D FITS file

Set `FITS_FILE` (and `FRAME_INDEX` for a 3-D cube) at the top of `swirl_tracer.py`. The other scripts (`parameter_sweep.py`, `swirl_centre_voting.py` and `swirl_arm_reconstruction.py`) read `FITS_FILE` and `FRAME_INDEX` from `swirl_tracer.py` automatically.

### Tuning parameters

The defaults are tuned for SST/CRISP chromospheric data. Check them when moving to a different instrument or spatial resolution:

- **Edge scale and contrast:** run `parameter_sweep.py`, then inspect `outputs/parameter_sweep/01_canny_mosaic.png` and `02_bridge_mosaic.png` to choose `CANNY_SIGMA`, `CANNY_HIGH`, `MAX_BRIDGE` and `MAX_ANGLE`. Avoid choosing parameters only by maximising path length, because over-bridging merges unrelated fibrils.
- **Swirl size and run time:** in `swirl_centre_voting.py`, set `R_MIN` and `R_MAX` (in pixels) to the size of your swirl. Bootstrapping (`N_BOOT = 300`) and null-model permutations (`N_NULL = 30`) take a minute or two; lower `N_NULL` for quick exploratory runs.

---

## Repository structure and outputs

| Script | Description | Main outputs |
| :--- | :--- | :--- |
| `sav_to_fits.py` | Interactive spectral viewer and `.sav` to cropped FITS/NPZ converter | `data/ca8542_*.fits`, `.npz` |
| `swirl_tracer.py` | Canny + $m$-adjacency graph tracer with geometric gap bridging | `outputs/swirl_graph_detection.png`, `.npz` |
| `parameter_sweep.py` | 4×4 visual mosaics across Canny, bridging and cost-weight parameters | `outputs/parameter_sweep/01..04_*.png`, `sweep_metrics.csv` |
| `swirl_centre_voting.py` | Normal-line accumulator voting, least-squares centre refinement, null tests and log-spiral arm linking | `outputs/swirl_centres/overview.png`, `centre_X.png`, `centres.csv`, `arms.csv` |
| `swirl_arm_reconstruction.py` | Conservative segment linker across short gaps, without model interpolation (experimental) | `outputs/swirl_centres/arm_reconstruction/centre_XX_arms.png`, `.npz` |

Generated results go to `outputs/`, which is not tracked by git.

---

## Data acknowledgement

The example data were obtained with the Swedish 1-m Solar Telescope (SST), operated on the island of La Palma by the Institute for Solar Physics of Stockholm University in the Spanish Observatorio del Roque de los Muchachos of the Instituto de Astrofísica de Canarias.

---

## Citation and licence

If you use or adapt this code, please cite the software using the details in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button generates the citation). A paper describing the code is in preparation.

Released under the [MIT License](LICENSE).
