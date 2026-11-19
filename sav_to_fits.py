"""Convert an IDL .sav spectral cube into a cropped FITS cube and NPZ archive.

Designed for SST/CRISP Ca II 8542 spectral scans (IDL dimensions [x, y, n_lambda]).

Modes (controlled via PLOT_MODE in the Configuration block):
    PLOT_MODE = True  -> Interactive inspection only; no files are written.
                         - PLOT_STYLE = "slider": scan spectral positions with a
                           slider or Left/Right arrow keys.
                         - PLOT_STYLE = "grid": display selected spectral planes
                           side by side (PLOT_POSITIONS).
    PLOT_MODE = False -> Export cropped 3-D FITS cube, single-frame 2-D FITS,
                         and compressed NPZ to OUTPUT_DIR.

                         
After scipy.io.readsav, the Python array axis order is data[spectral, y, x].
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from astropy.io import fits
from scipy.io import readsav

#hello :)
# =============================================================================
# Configuration
# =============================================================================

# Input IDL .sav file and variable name
SAV_FILE = Path(__file__).resolve().parent / 'data' / 'ca8542_224.000.sav'
SAV_VARIABLE = "data"

# Output directory for FITS / NPZ files (used when PLOT_MODE = False)
OUTPUT_DIR = Path(__file__).resolve().parent / 'data'

# Swirl centre in full-frame pixel coordinates (IDL x, y)
SWIRL_X = 724
SWIRL_Y = 524

# Half-width of the square crop in pixels (80 -> 160 x 160 px crop)
CROP_HALF = 80

# Spectral index to export as a 2-D frame (and initial slider position)
SPECTRAL_POS = 22

# Convert all-zero dropout rows to NaN so downstream preprocessing fills them
MASK_DROPOUT_ROWS = True

# -----------------------------------------------------------------------------
# Plotting options
# -----------------------------------------------------------------------------
PLOT_MODE = True
PLOT_STYLE = "slider"  # "slider" or "grid"
PLOT_POSITIONS = [19, 20, 21, 23, 25, 26, 27, 28, 29]
GRID_COLUMNS = 4
SHOW_CENTRE = True
DROPOUT_COLOUR = "red"
CLIP_PERCENTILES = (1.0, 99.5)


# =============================================================================
# Plotting helpers
# =============================================================================

def _clim(frame):
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return 0.0, 1.0
    return tuple(np.percentile(finite, CLIP_PERCENTILES))


def _cmap():
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(DROPOUT_COLOUR)
    return cmap


def print_dropout_table(crop):
    print("\nDropout inside the crop, per spectral position")
    print("(positions with 0.0% are clean across the cropped FOV)\n")
    for p in range(crop.shape[0]):
        frac = 100 * np.isnan(crop[p]).mean()
        flag = "" if frac == 0 else "   <-- missing data"
        print(f"  pos {p:2d}: {frac:5.1f}%{flag}")


def plot_slider(crop, x0, x1, y0, y1):
    nlam = crop.shape[0]
    extent = [x0, x1, y0, y1]
    cmap = _cmap()
    start = int(np.clip(SPECTRAL_POS, 0, nlam - 1))

    fig, ax = plt.subplots(figsize=(8, 8.5))
    plt.subplots_adjust(bottom=0.14)

    frame = np.ma.masked_invalid(crop[start])
    vmin, vmax = _clim(crop[start])

    im = ax.imshow(
        frame,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    if SHOW_CENTRE:
        ax.plot(SWIRL_X, SWIRL_Y, "c+", markersize=14, markeredgewidth=1.5)

    ax.set_xlabel("x [pixel, full frame]")
    ax.set_ylabel("y [pixel, full frame]")

    def title(p):
        frac = 100 * np.isnan(crop[p]).mean()
        return f"Spectral position {p}   |   dropout {frac:.1f}%"

    ax.set_title(title(start))

    slider_ax = fig.add_axes([0.15, 0.04, 0.70, 0.03])
    slider = Slider(
        slider_ax,
        "position",
        0,
        nlam - 1,
        valinit=start,
        valstep=1
    )

    def update(val):
        p = int(slider.val)
        im.set_data(np.ma.masked_invalid(crop[p]))
        lo, hi = _clim(crop[p])
        im.set_clim(lo, hi)
        ax.set_title(title(p))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(slider.val + 1, nlam - 1))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._slider = slider
    plt.show()


def plot_grid(crop, x0, x1, y0, y1):
    nlam = crop.shape[0]
    extent = [x0, x1, y0, y1]
    cmap = _cmap()
    positions = [p for p in PLOT_POSITIONS if 0 <= p < nlam]

    ncols = min(GRID_COLUMNS, len(positions))
    nrows = int(np.ceil(len(positions) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.4 * nrows),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, p in zip(axes.flat, positions):
        ax.axis("on")
        vmin, vmax = _clim(crop[p])
        ax.imshow(
            np.ma.masked_invalid(crop[p]),
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        if SHOW_CENTRE:
            ax.plot(SWIRL_X, SWIRL_Y, "c+", markersize=12, markeredgewidth=1.3)
        frac = 100 * np.isnan(crop[p]).mean()
        ax.set_title(f"pos {p}  (dropout {frac:.0f}%)", fontsize=10)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    cube = readsav(str(SAV_FILE))[SAV_VARIABLE]
    cube = np.asarray(cube).astype(np.float32)

    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D cube, got {cube.shape}")

    nlam, ny, nx = cube.shape
    print(f"Cube [spectral, y, x] = {cube.shape}")

    if MASK_DROPOUT_ROWS:
        for p in range(nlam):
            bad_rows = (cube[p] == 0).all(axis=1)
            cube[p, bad_rows, :] = np.nan

    x0 = max(0, SWIRL_X - CROP_HALF)
    x1 = min(nx, SWIRL_X + CROP_HALF)
    y0 = max(0, SWIRL_Y - CROP_HALF)
    y1 = min(ny, SWIRL_Y + CROP_HALF)

    crop = cube[:, y0:y1, x0:x1]

    print(f"Crop x[{x0}:{x1}] y[{y0}:{y1}] -> {crop.shape}")
    print(f"Swirl centre inside crop: x={SWIRL_X - x0}, y={SWIRL_Y - y0}")

    frac_nan = np.isnan(crop[SPECTRAL_POS]).mean()
    print(f"Spectral position {SPECTRAL_POS}: "
          f"{100 * frac_nan:.1f}% of the crop is dropout")

    if frac_nan > 0.05:
        good = [p for p in range(nlam) if np.isnan(crop[p]).mean() == 0]
        near = sorted(good, key=lambda p: abs(p - SPECTRAL_POS))[:6]
        print(f"  WARNING: position {SPECTRAL_POS} has missing data in this crop.")
        print("  Nearest gap-free positions in this crop:", near)

    if PLOT_MODE:
        print_dropout_table(crop)
        print(f"\nPLOT_MODE is on ({PLOT_STYLE}): no files are written.")
        if PLOT_STYLE == "slider":
            plot_slider(crop, x0, x1, y0, y1)
        elif PLOT_STYLE == "grid":
            plot_grid(crop, x0, x1, y0, y1)
        else:
            raise ValueError("PLOT_STYLE must be 'slider' or 'grid'")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    hdr["CROPX0"] = (x0, "crop origin x in full frame")
    hdr["CROPY0"] = (y0, "crop origin y in full frame")
    hdr["SWIRLX"] = (SWIRL_X - x0, "swirl centre x inside crop")
    hdr["SWIRLY"] = (SWIRL_Y - y0, "swirl centre y inside crop")
    hdr["AXIS0"] = "spectral position (0-based)"

    cube_name = OUTPUT_DIR / "ca8542_224_swirl_cube.fits"
    frame_name = OUTPUT_DIR / f"ca8542_224_swirl_pos{SPECTRAL_POS:02d}.fits"
    npz_name = OUTPUT_DIR / "ca8542_224_swirl_cube.npz"

    fits.PrimaryHDU(crop, header=hdr).writeto(cube_name, overwrite=True)
    fits.PrimaryHDU(crop[SPECTRAL_POS], header=hdr).writeto(frame_name, overwrite=True)

    np.savez_compressed(
        npz_name,
        cube=crop,
        x0=x0,
        y0=y0,
        swirl_x=SWIRL_X - x0,
        swirl_y=SWIRL_Y - y0,
    )

    print("\nWrote:")
    for f in (cube_name, frame_name, npz_name):
        print("  ", f)

    print("\nIn swirl_tracer.py set:")
    print(f'  FITS_FILE   = Path("{cube_name.resolve()}")')
    print(f"  FRAME_INDEX = {SPECTRAL_POS}")


if __name__ == "__main__":
    main()
