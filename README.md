# uspexkit — USPEX Crystal Structure Prediction Post-Processing Toolkit

`uspexkit` is a post-processing toolkit for USPEX molecular crystal structure prediction (calculation type 310), providing trajectory conversion, feature extraction, Gaussian process prediction, high-throughput DFT screening, broken molecule repair, and more.

## Installation

```bash
git clone https://github.com/FengGo/uspexkit.git
cd uspexkit
pip install .
```

Dependencies: `numpy>=1.20`, `scikit-learn>=1.0`, `ase>=3.22`, and the internal library `irff` (providing GULP / LAMMPS / SIESTA interfaces).

---

## Command Overview

| Command | Function |
|---------|----------|
| `traj` | Convert `gatheredPOSCARS` to ASE trajectory file |
| `calcdata` | Batch compute crystal feature vectors from trajectories |
| `gp` | Gaussian process prediction of crystal density (called within USPEX pipeline) |
| `pred` | Predict density and energy using GP + Random Forest |
| `calc` | High-throughput DFT (SIESTA) calculation + structure matching deduplication |
| `fixbroken` | Repair broken molecules |
| `add` | Add a single structure to the feature database |
| `addall` | Batch add all structures from a trajectory to the feature database |
| `zmat` | Convert structure coordinates to USPEX Z-matrix internal coordinates |
| `fdf` | Generate SIESTA input files |
| `sample` | Sample structures by index and output to a trajectory |
| `supercell` | Build supercells |
| `fingerprint` | Compute USPEX molecular structure fingerprints (Cython-accelerated) |

---

## 1. `traj` — Trajectory Conversion

Convert USPEX output file `gatheredPOSCARS` to ASE `.traj` trajectory file, while parsing energy information from the `Individuals` file.

```bash
uspexkit traj [--fposcar FILE]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fposcar` | `gatheredPOSCARS` | Input POSCAR file path |

### Output

- `Individuals.traj` — ASE trajectory file containing all structures and energies

### How It Works

Parse each structure in `gatheredPOSCARS` (delimited by `EA` lines), write to `POSCAR` and read with ASE, then match corresponding enthalpy values from `Individuals` and write them into `SinglePointCalculator`.

---

## 2. `calcdata` — Compute Feature Vectors

Read structures from ASE trajectories, relax them via MLP (neural network potential) or MTP (moment tensor potential), then compute energy decomposition features using GULP and output to a CSV database.

```bash
uspexkit calcdata [--t TRAJ] [--n NCPU] [--c CALC] [--step STEPS]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t` | `structures.traj` | Input trajectory file |
| `--n` | `1` | Number of parallel CPU cores |
| `--c` | `nn` | Calculator type: `nn` (neural network reactive potential) or `mtp` (MTP potential) |
| `--step` | `1000` | MLP relaxation steps |

### Output

| File | Content |
|------|---------|
| `feature_mlp.csv` | 10-dimensional features (GULP total energy + energy decomposition + density) |
| `feature.csv` | 10-dimensional features (DFT total energy + GULP energy decomposition + density) |
| `structures_mlp.traj` | Structure trajectory after MLP relaxation |

### Feature Vector (10 dimensions)

| Dimension | Meaning |
|-----------|---------|
| 0 | Total energy (etot) |
| 1 | Bond energy (ebond) |
| 2 | Angle energy (eang) |
| 3 | Torsion energy (etor) |
| 4 | Van der Waals energy (evdw) |
| 5 | Hydrogen bond energy C-H-O (ehb_cho) |
| 6 | Hydrogen bond energy C-H-N (ehb_chn) |
| 7 | Hydrogen bond energy C-H-C (ehb_chc) |
| 8 | Coulomb energy (ecoul) |
| 9 | Density (density) |

---

## 3. `gp` — Gaussian Process Prediction (USPEX Pipeline)

Called internally within USPEX evolutionary search: GULP gradient relaxation of the current structure → feature extraction → GP + RF prediction of density/energy → write predictions back to USPEX output format.

```bash
uspexkit gp [--n NCPU] [--t TOL] [--step STEPS] [--b BROKEN] [--u UNCERT] [--f FEAT] [--data DIR] [--resf DIR]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n` | `1` | Number of parallel CPU cores |
| `--t` | `0.005` | Structure matching tolerance |
| `--step` | `1000` | MLP relaxation steps |
| `--b` | `1.5` | Broken molecule threshold: current energy deviating from mean by more than this value is considered broken |
| `--u` | `0.2` | GP uncertainty threshold |
| `--f` | `1` | Feature flag: `1` = 10-dimensional (with hydrogen bonds), other = 7-dimensional |
| `--data` | `data` | Training data directory name |
| `--resf` | `results1` | Results output directory name |

### Output

| File | Content |
|------|---------|
| `output` | USPEX-format energy output |
| `optimized.structure` | USPEX-format optimized structure |
| `gpr_density.pkl` / `gpr_energy.pkl` | Trained GP models |
| `rfr_density.pkl` | Trained Random Forest model |
| `gp.csv` (under `results1/`) | Prediction log |

### How It Works

1. GULP gradient relaxation of the current structure
2. Compute 10-dimensional features (including C-H-O / C-H-N / C-H-C hydrogen bonds)
3. Load `feature_mlp.csv` / `feature.csv` / `structures.traj` from the training data directory
4. Train GP density model + GP energy model + RF density model (cached as `.pkl` after first training)
5. Predict density and energy for new structures, output in USPEX format
6. If the residual with nearest neighbor > 10 and RF prediction deviation is large, fall back to scaling correction

---

## 4. `pred` — Predict Density/Energy

Predict density and energy for specified structures (by index or POSCAR file) using trained GP + RF models.

```bash
uspexkit pred [--t TRAJ] [--g GEO] [--f FEAT] [--den DENSITY] [--ids IDS] [--x INDEX] [--c CALC] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t` | `Individuals.traj` | Trajectory file |
| `--g` | `None` | Geometry file (e.g., `POSCAR`); if specified, predict directly from this file |
| `--f` | `1` | Feature flag: `1` = 10-dimensional (with hydrogen bonds), other = 7-dimensional |
| `--den` | `1.88` | Density threshold: only predict structures with density above this value |
| `--ids` | `None` | Crystal indices (space-separated), e.g., `"214 215"` |
| `--x` | `-1` | Single structure index (`-1` means the last one) |
| `--c` | `nn` | Calculator: `nn` (neural network potential) or `mtp` (MTP potential) |
| `--step` | `300` | MLP relaxation steps |
| `--ncpu` | `8` | Number of parallel CPU cores |
| `--dat` | `data` | Training data directory name |
| `--tolerance` | `0.001` | Structure matching tolerance |

### Output

- `density_predict.log` — Prediction log, each line contains: structure ID, residual, density (MLP/RF/GP), GP uncertainty, energy prediction

### Usage Examples

```bash
# Predict structures with specified indices
uspexkit pred --ids="214 215" --n=24 --dat=data11_44

# Predict from a POSCAR file
uspexkit pred --g=POSCAR --n=24 --dat=data11_44
```

---

## 5. `calc` — High-Throughput DFT Calculation

Perform SIESTA DFT calculations on qualifying structures, with structure matching deduplication (already-calculated structures are automatically skipped).

```bash
uspexkit calc [--t TRAJ] [--den DENSITY] [--ids IDS] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t` | `Individuals.traj` | Trajectory file |
| `--den` | `1.88` | Density threshold: only calculate structures with density above this value |
| `--ids` | `None` | Crystal indices (space-separated) |
| `--step` | `300` | GULP relaxation steps |
| `--ncpu` | `8` | Number of parallel CPU cores |
| `--dat` | `data` | Training data directory name |
| `--tolerance` | `0.01` | Structure matching tolerance |

### Output

- `density.log` — Calculation log
- `{id}/` — Independent working directory for each structure, containing DFT input/output
- `{id}/POSCAR.{id}` — Original structure
- `{id}/POSCAR.{id}_opt` — DFT-optimized structure
- `{id}/id_{id}.traj` — DFT optimization trajectory

### How It Works

1. Read structure list from `Individuals`, filter structures with `density > den` and `fitness < 0`
2. For each structure: GULP relaxation → compute 10-dimensional features → match against database
3. If a known structure is matched (residual < tolerance): directly reuse existing DFT results
4. If unmatched: execute SIESTA DFT optimization (GGA-PBE), and append results to the database

---

## 6. `fixbroken` — Repair Broken Molecules

Detect whether the current structure is broken (energy deviation from mean exceeds threshold), and if so, attempt repair by expanding the unit cell and re-relaxing.

```bash
uspexkit fixbroken [--n NCPU] [--data DIR] [--s SCALE] [--b BROKEN]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n` | `1` | Number of parallel CPU cores |
| `--data` | `data` | Training data directory name |
| `--s` | `1.2` | Unit cell scaling factor (multiplied each iteration) |
| `--b` | `1.5` | Broken threshold (eV): current energy deviating from training data mean by more than this value is considered broken |

### Output

- `output` — USPEX-format energy output
- `optimized.structure` — Optimized structure

### How It Works

1. GULP gradient relaxation of the current structure
2. If `E_mean_train - E_current > broken`: read molecular fragments → gradually expand the unit cell (×1.2, up to 15 iterations) → re-relax until energy returns to normal
3. If not broken: identify and cache molecular fragments (`molecule.pkl`) on first run

---

## 7. `add` — Add a Single Structure

Append a single DFT-optimized structure to the feature database.

```bash
uspexkit add [--t TRAJ] [--i INDEX] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t` | `structures.traj` | Trajectory file |
| `--i` | `-1` | Structure index (`-1` means the last one) |
| `--s` | `1000` | MLP relaxation steps |
| `--tolerance` | `0.005` | Structure matching tolerance (deduplication) |
| `--n` | `1` | Number of parallel CPU cores |

### Output

Updates `feature_mlp.csv`, `feature.csv`, `structures_mlp.traj`, `structures.traj`.

---

## 8. `addall` — Batch Add Structures

Batch append all structures from a trajectory to the feature database.

```bash
uspexkit addall [--t TRAJ] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--t` | `structures.traj` | Input trajectory file |
| `--s` | `1000` | MLP relaxation steps |
| `--tolerance` | `0.005` | Structure matching tolerance (deduplication) |
| `--n` | `1` | Number of parallel CPU cores |

---

## 9. `zmat` — Z-matrix Internal Coordinates

Convert Cartesian coordinate structures to USPEX-format Z-matrix internal coordinate files (`MOL_*`).

```bash
uspexkit zmat [--geo GEO] [--i INDEX]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--geo` | `POSCAR` | Input geometry file |
| `--i` | `-1` | Frame index (`-1` = last frame) |

### Output

- USPEX-format `MOL_*` internal coordinate files

### Usage Example

```bash
uspexkit zmat --g=POSCAR
```

---

## 10. `fdf` — Generate SIESTA Input

Generate SIESTA DFT `.fdf` input files from structure files.

```bash
uspexkit fdf [--gen GEN] [--xcf XCF] [--i INDEX]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gen` | `poscar.gen` | Input `.gen` format structure file |
| `--xcf` | `gga` | Exchange-correlation functional: `gga` (GGA-PBE) or `vdw` (VDW-DRSLL) |
| `--i` | `-1` | Frame index |

---

## 11. `sample` — Sample Structures

Extract specified structures by index from a trajectory or DFT results directory.

```bash
uspexkit sample [--ind INDICES] [--t TRAJ]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ind` | `""` | Structure indices (space-separated), e.g., `"0 5 12"` |
| `--t` | `None` | Trajectory file path. If specified, extract from the trajectory; otherwise read from `{i}/POSCAR.{i}_opt` |

### Output

- `samples.traj` — Sampled structure trajectory

---

## 12. `supercell` — Build Supercells

Build supercells from structures or trajectories.

```bash
uspexkit supercell [--x NX] [--y NY] [--z NZ] [--t TRAJ] [--g GEO]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--x` | `1` | X-direction multiplier |
| `--y` | `1` | Y-direction multiplier |
| `--z` | `1` | Z-direction multiplier |
| `--t` | `None` | Trajectory file (takes the last frame) |
| `--g` | `None` | Geometry file (e.g., `POSCAR`) |

### Output

- If `--g` is specified: outputs `POSCAR.supercell_{x}_{y}_{z}`
- If `--t` is specified: outputs `{traj_name}_{x}{y}{z}.traj`, energy scaled by volume ratio

---

## 13. `fingerprint` — Molecular Structure Fingerprint

Compute USPEX molecular structure fingerprints using Cython-accelerated computation (`makeMatrices` + `fingerprint_calc`), outputting three fingerprint arrays: `order`, `fing`, and `atom_fing`.

```bash
uspexkit fingerprint [--g GEO] [--traj TRAJ] [--i I] [--rmax RMAX] [--sigma SIGMA] [--delta DELTA] [--dimension DIM] [--output OUTPUT]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--g` | `None` | Geometry file (e.g., `POSCAR`, `gulp.cif`), read with ASE |
| `--traj` | `None` | Trajectory file (mutually exclusive with `--g`) |
| `--i` | `-1` | Trajectory frame index (`-1` = last frame) |
| `--rmax` | `12.0` | Neighbor search cutoff radius Rmax (Å) |
| `--sigma` | `0.05` | Gaussian broadening σ |
| `--delta` | `0.08` | Distance bin width δ (Å) |
| `--dimension` | `3` | Dimensionality: `3` = 3D crystal, `0` = cluster, `2` = 2D |
| `--output` | `None` | Output `.npz` file path (optional; if not specified, only prints summary) |

### Output

- Terminal prints fingerprint statistics (shape, min/max, mean, timing)
- If `--output` is specified: saves a `.npz` file containing `order`, `fing`, `atom_fing`, `V`, `n_pairs`, and other arrays

### Usage Examples

```bash
# Compute fingerprint from POSCAR
uspexkit fingerprint --g=POSCAR

# Custom parameters and save results
uspexkit fingerprint --g=POSCAR --rmax=10.0 --sigma=0.07 --output=fp.npz

# Compute from trajectory file
uspexkit fingerprint --traj=Individuals.traj --i=-1
```

### How It Works

1. Read structure file with ASE, extracting lattice, fractional coordinates, element types, and atom counts
2. Call the Cython-accelerated module `uspex_fast_core.compute_all`:
   - `build_distance_matrix`: build neighbor distance matrix (replaces Octave `makeMatrices.m`)
   - `fingerprint_calc`: compute erf-broadened distance histogram fingerprint (replaces Octave `fingerprint_calc.m`)
3. Output three fingerprint arrays:
   - `order` (N,): structural order parameter for each atom (√(Σ weight·δ·atom_fing²/V^(1/3)))
   - `fing` (S², numBins): global fingerprint matrix (S = number of element types)
   - `atom_fing` (N, S, numBins): atom-level fingerprint

---

## Data Format Reference

### Fingerprint Output (`.npz`)

The `.npz` file generated by `fingerprint --output` contains the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `order` | (N,) | Structural order parameter for each atom |
| `fing` | (S², numBins) | Global fingerprint matrix (S = number of element types) |
| `atom_fing` | (N, S, numBins) | Atom-level fingerprint |
| `V` | scalar | Unit cell volume |
| `n_pairs` | scalar | Number of neighbor atom pairs |
| `numIons` | (S,) | Atom count per element type |
| `atomType` | (S,) | Atomic number per element type |
| `rmax` / `sigma` / `delta` / `dimension` | scalar | Computation parameters |

### `feature_mlp.csv` (GULP Energy Features)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -123.45, -56.78, ...
```

The first column is the structure index, followed by the 10-dimensional feature vector.

### `feature.csv` (DFT Energy + GULP Decomposition)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -234.56, -56.78, ...
```

The first column is the DFT total energy (from `SinglePointCalculator`), followed by GULP decomposed energies.

### GP Model Files

- `gpr_density.pkl` — Gaussian process density model
- `gpr_energy.pkl` — Gaussian process energy model
- `rfr_density.pkl` — Random Forest density model

Kernel: `0.00581² · DotProduct(σ₀=0.412) + 0.35² · Matern(ν=2.5, ARD) + WhiteKernel`

---

## Typical Workflows

### 1. Build Training Database

```bash
# Generate feature database from DFT results
uspexkit calcdata --t=structures.traj --n=24

# Or manually add structures
uspexkit add --t=structures.traj --i=-1 --n=24
```

### 2. Predict During Evolutionary Search

Configure in USPEX's `command`:

```bash
uspexkit gp --n=24 --data=data11_44 --resf=results1
```

### 3. High-Throughput DFT Screening

```bash
uspexkit calc --n=24 --den=1.88 --dat=data11_44
```

### 4. Broken Molecule Repair

```bash
uspexkit fixbroken --n=24 --data=data11_44 --b=1.5
```

