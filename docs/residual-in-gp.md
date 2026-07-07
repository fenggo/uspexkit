# Understanding the Residual in `gp.csv`

## Definition

The residual column in `gp.csv` is the **squared Euclidean distance** between the current structure's 10-D feature vector and its nearest neighbor in the training set.

From `core.py`:

```python
res    = np.sum(np.square(d - feature), axis=1)   # squared Euclidean distance to all training points
imin   = np.argmin(res)                            # index of the nearest neighbor
```

Mathematically:

\[
\text{residual} = \min_{j} \sum_{i=1}^{10} \left( f_i - f_i^{(j)} \right)^2
\]

---

## What It Means

### 1. A Proxy for Extrapolation Risk

GP predictions are fundamentally interpolation within the feature space. The residual measures how far the query point lies from known territory:

| Residual | Interpretation | GP Reliability |
|----------|---------------|----------------|
| < 0.1 | Nearly identical to a known structure | High (essentially interpolation) |
| 0.1 – 1 | Close to known structures | Medium-high |
| 1 – 10 | Deviating from the training distribution | Medium-low |
| > 10 | Unexplored region of feature space | Low (triggers the fallback logic) |

The `res[imin] > 10` threshold in the code embodies this design: when the residual is too large, GP extrapolation is unreliable, so the code falls back to the RF prediction or density scaling.

### 2. What "Distance" Means in This 10-D Space

The ten dimensions are:

```
etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
```

They form a joint **energy–density descriptor space**. Two structures with small residual share similar:

- Bonding energetics (bond, angle, torsion)
- Intermolecular interactions (vdW, H-bond, Coulomb)
- Packing density

This similarity is **chemically meaningful** — it captures not just geometry but the energy fingerprint of the crystal.

### 3. Complementarity with GP Uncertainty

```
Residual     → local metric  (nearest neighbor only)
GP std       → global metric (conditioned on all training points + kernel)
```

They are not redundant:

- **Small residual, large uncertainty** → the region is sparsely sampled but happens to have one close neighbor
- **Large residual, small uncertainty** → theoretically unlikely (distant points usually mean high uncertainty), but the smoothness of the Matérn ν=2.5 kernel can produce this artifact

The code outputs both the residual and `1.96 * std` (95% confidence interval half-width) so users can assess prediction quality from two angles.

### 4. The Fallback Logic at Residual > 10

When `res[imin] > 10`, the code executes:

```python
if density_rf[0] / density > 1.5 or density / density_rf[0] > 1.5:
    density_ = density * d_scaler    # fall back to scaled mean density
else:
    density_ = density_rf[0]         # use random forest prediction
```

This is a sanity check: if even the random forest disagrees with the MLP-relaxed density by more than 50%, neither model can be trusted, and falling back to the training-set mean density (via `density * d_scaler`) is the safer bet.

---

## Practical Guidelines

When inspecting `gp.csv`:

- **residual < 1 and uncertainty < 0.1**: prediction is trustworthy, use directly
- **residual 1–10**: prediction has reference value; cross-check with the RF prediction
- **residual > 10**: the structure likely represents a genuinely new packing motif. The GP prediction is only a rough guide — prioritize DFT validation
