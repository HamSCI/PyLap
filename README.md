# PyLap

A Python 3 wrapper for the PHaRLAP ionospheric raytracer.

> **Warning:** PyLap is based on the scientific software PHaRLAP and is not meant for operational use as stated on PHaRLAP's website. Any and all risk falls to you if you are using this software.

## About this fork

This is a patched fork of [HamSCI/PyLap](https://github.com/HamSCI/PyLap) maintained for use with [hf-timestd](https://github.com/mijahauan/hf-timestd) and other HF-propagation projects. The upstream version will not build or run correctly against current PHaRLAP (4.7.4) without these patches:

- **PHaRLAP 4.7.4 support** — upstream targets 4.5.x; this fork's `setup.py` links against 4.7.4 static libraries and gracefully skips legacy IRI modules (`iri2007`, `iri2012`) when their `.a` files are absent.
- **Cross-platform builds** — adds macOS arm64 and macOS x86_64 alongside Linux x86_64 (upstream is Linux-only).
- **Unified gfortran build** — Intel Fortran redistributable is no longer required; Linux and macOS both build against gfortran, eliminating a significant install-time hurdle.
- **GCC 14+ compatibility** — adds `-Wno-error=incompatible-pointer-types` so builds succeed on Debian 13/trixie and other distros that ship GCC 14 (which promotes this warning to an error).
- **Multi-hop stride fix in `raytrace_2d.c`** — the `ray_data` C-array hop stride was `num_rays × 9`; corrected to `num_rays × 24` so fields past hop 0 are read from the right offsets under PHaRLAP 4.7.4.
- **`iri2016` module now calls IRI-2020** — PHaRLAP 4.7.4 replaced IRI-2016 internals with IRI-2020; the fork's wrapper was updated to match.
- **Fortran SAVE-variable segfault workaround (caller-side note)** — repeated `raytrace_2d` calls can crash due to persistent Fortran state; make a single call with `nhops=max_hops` and iterate hops in Python.
- **Ne unit convention (caller-side note)** — IRI-2020 returns electron density in m⁻³ but `raytrace_2d` expects cm⁻³; scale by 10⁻⁶ before passing the grid.

## Requirements

- **OS:** Ubuntu/Debian Linux (x86_64), WSL2 on Windows 10/11, macOS (arm64 or x86_64)
- **Python:** 3.8+
- **PHaRLAP:** 4.7.4 required for raytracing (request access from DST Australia — see below)
- **gfortran:** Required to build PyLap and the IRI-2020 Python package — `apt install gfortran` on Linux, `brew install gcc` on macOS. Intel Fortran is **not** required (removed in this fork).

## Quick Start

### 1. Obtain PHaRLAP

PHaRLAP is **not** redistributed with this repo — you must obtain it separately.

**PHaRLAP 4.7.4** (required for raytracing):
- Request access at https://www.dst.defence.gov.au/our-technologies/pharlap-provision-high-frequency-raytracing-laboratory-propagation-studies
- DST will email you a download link after reviewing your request (typically 1–3 business days)
- License restricts use to research; see PHaRLAP's own license terms
- After extracting, verify `lib/` contains `linux/`, `maca/`, `maci/` subdirectories (these hold the static libraries this fork links against)

Suggested layout:

```
~/pylap_project/
├── PyLap/            # This repository
└── pharlap_4.7.4/    # PHaRLAP toolbox (from DST)
```

### 2. Run Setup Script

```bash
cd ~/pylap_project/PyLap
. ./setup.sh
# Enter: ~/pylap_project
```

The setup script will:
- Create a Python virtual environment at `PyLap/venv/`
- Install system dependencies (requires sudo)
- Install Python packages from `requirements.txt`
- Build and install PyLap
- Add a `pylap-activate` alias to your `.bashrc`

### 3. Activate and Run

```bash
# Activate the virtual environment
source venv/bin/activate
# Or use the alias:
pylap-activate

# Test IRI-2020 (works without PHaRLAP)
python3 Examples/test_iri2020.py

# Test raytracing (requires PHaRLAP)
python3 Examples/ray_test1.py
```

> **Note:** The raytracing examples (`ray_test1.py`, etc.) require PHaRLAP to be installed. Use `test_iri2020.py` to verify your installation before PHaRLAP is available.

### 4. Verify the Build

Quick smoke test that confirms PHaRLAP linked correctly and IRI is usable:

```bash
python3 -c "
import importlib, math
iri = importlib.import_module('pylap.iri2016')
_, oarr = iri.iri2016(40.0, -90.0, 100.0, [2026, 1, 15, 18, 0], 100.0, 10.0, 50, {})
foF2 = 8.98 * math.sqrt(max(oarr[0], 0)) / 1e6
print(f'IRI OK — foF2={foF2:.1f} MHz, hmF2={oarr[1]:.0f} km')
"
```

Expected: `IRI OK — foF2=X.X MHz, hmF2=XXX km` (values depend on solar conditions at the queried date/time). If you see `ImportError` or a Fortran/link error, `PHARLAP_HOME` and `DIR_MODELS_REF_DAT` are probably unset or pointing at the wrong directory.

## IRI-2020 Ionosphere Model

PyLap now supports IRI-2020 via the `iri2020` Python package. This works **independently of PHaRLAP** for ionosphere generation.

### Supported Profile Types

| Profile Type | Description |
|--------------|-------------|
| `'iri'` or `'iri2016'` | IRI-2016 model (default, requires PHaRLAP) |
| `'iri2020'` | IRI-2020 model (standalone, no PHaRLAP needed) |
| `'iri2012'` | IRI-2012 model (requires PHaRLAP) |
| `'iri2007'` | IRI-2007 model (requires PHaRLAP) |
| `'firi'` | IRI-2016 with FIRI D-region (requires PHaRLAP) |

### Example Usage

```python
from Ionosphere import gen_iono_grid_2d as gen_iono

# Generate ionosphere using IRI-2020 (no PHaRLAP required)
iono_pf_grid, iono_pf_grid_5, collision_freq, irreg, iono_te_grid = \
    gen_iono.gen_iono_grid_2d(
        origin_lat, origin_long, R12, UT, ray_bear,
        max_range, num_range, range_inc, start_height,
        height_inc, num_heights, kp, doppler_flag, 
        'iri2020'  # Use IRI-2020
    )
```

## Manual Setup

If you prefer not to use the setup script:

```bash
# 1. Set environment variables
export PHARLAP_HOME="/path/to/pharlap_4.7.4"
export PYTHONPATH="/path/to/PyLap"
export DIR_MODELS_REF_DAT="${PHARLAP_HOME}/dat"

# 2. Install system dependencies (Linux)
sudo apt-get install python3-tk python3-pil python3-pil.imagetk \
    libqt5gui5 python3-pyqt5 python3-venv gfortran
sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev \
    libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

# macOS equivalent
# brew install gcc python-tk

# 3. Create virtual environment
cd /path/to/PyLap
python3 -m venv venv
source venv/bin/activate

# 4. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 5. Build PyLap
python3 setup.py install
```

## Re-running Setup

To redo the setup:

1. Edit `~/.bashrc` and remove PyLap environment variables
2. Delete the venv: `rm -rf PyLap/venv`
3. Run `setup.sh` again

## Contact

For questions or help troubleshooting, email: Devin.diehl@scranton.edu
