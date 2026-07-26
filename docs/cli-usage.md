# uspexkit CLI 用法

## fingerprint — 结构指纹计算

### 1. 仅 RDF 指纹（USPEX 原版算法，Cython 加速）

```bash
uspexkit fingerprint --g POSCAR --rmax 10 --sigma 0.03 --delta 0.08
```

### 2. RDF + Intra_map 过滤（分子晶体全原子指纹）

```bash
uspexkit fingerprint --g POSCAR \
    --rmax 10 --sigma 0.03 --delta 0.08 \
    --intra-map intra_map.npy \
    --output fp.npz
```

`--intra-map` 支持 `.npy` 和 `.npz` 格式。仅过滤零位移（基本晶胞内）
的分子内原子对，周期镜像对始终保留——与 Octave `ReadJobs_310.m` 行为一致。

### 3. RDF + SOAP 混合指纹

```bash
uspexkit fingerprint --g POSCAR \
    --soap --soap-r-cut 6 --soap-n-max 8 --soap-l-max 6 \
    --output fp.npz
```

### 4. Intra_map + SOAP（完整模式）

```bash
uspexkit fingerprint --g POSCAR \
    --rmax 10 --sigma 0.03 --delta 0.08 \
    --intra-map intra_map.npy \
    --soap --soap-r-cut 6 --soap-n-max 8 --soap-l-max 6 \
    --output fp.npz
```

### 参数说明

| 参数             | 类型  | 默认值 | 说明                           |
|------------------|-------|--------|--------------------------------|
| `--g`            | str   | -      | 结构文件（POSCAR、cif 等）     |
| `--traj`         | str   | -      | ASE 轨迹文件（与 --g 二选一）  |
| `--i`            | int   | -1     | 轨迹帧索引                     |
| `--rmax`         | float | 12.0   | RDF 截断半径 (Å)               |
| `--sigma`        | float | 0.05   | RDF 高斯展宽                   |
| `--delta`        | float | 0.08   | RDF 距离 bin 宽度 (Å)          |
| `--dimension`    | int   | 3      | 维度：3=3D, 0=cluster, 2=2D   |
| `--output`       | str   | -      | 输出 .npz 文件                 |
| `--intra-map`    | str   | -      | 分子内距离标记矩阵 (.npy/.npz) |
| `--soap`         | flag  | False  | 同时计算 SOAP 指纹             |
| `--soap-r-cut`   | float | 6.0    | SOAP 局部环境截断半径 (Å)      |
| `--soap-n-max`   | int   | 8      | SOAP 径向基函数数量            |
| `--soap-l-max`   | int   | 6      | SOAP 最大角动量                |

### Python API 用法

```python
from uspexkit.core import fingerprint

# 仅 RDF
result = fingerprint(gen='POSCAR', rmax=10.0, sigma=0.03, delta=0.08)

# RDF + Intra_map
import numpy as np
intra_map = np.load('intra_map.npy')
result = fingerprint(gen='POSCAR', rmax=10.0, sigma=0.03, delta=0.08,
                     intra_map=intra_map)

# RDF + SOAP
result = fingerprint(gen='POSCAR', soap=True,
                     soap_r_cut=6.0, soap_n_max=8, soap_l_max=6)

# 完整模式
result = fingerprint(gen='POSCAR', rmax=10.0, sigma=0.03, delta=0.08,
                     intra_map=intra_map, soap=True)
```

### SOAP 模块独立使用

```python
from uspexkit.soap import (soap_fingerprint, soap_fingerprint_per_molecule,
                            soap_distance, soap_distance_matrix,
                            hybrid_distance, hybrid_distance_matrix)
from ase.io import read

# 单结构 SOAP
atoms = read('POSCAR')
fp = soap_fingerprint(atoms, r_cut=6.0, n_max=8, l_max=6)

# 按分子聚合 SOAP（捕获分子取向）
molecules = [[0,1,2,...], [7,8,9,...], ...]  # 原子索引列表
fp_mol = soap_fingerprint_per_molecule(atoms, molecules, r_cut=10.0)

# 两结构比较
fp1 = soap_fingerprint(read('POSCAR_1'))
fp2 = soap_fingerprint(read('POSCAR_2'))
dist = soap_distance(fp1, fp2)

# 混合 RDF + SOAP 距离
hybrid_dist = hybrid_distance(rdf_fp1, rdf_fp2, soap_fp1, soap_fp2, alpha=0.9)

# 批量距离矩阵
soap_fps = [soap_fingerprint(read(f)) for f in files]
dist_matrix = soap_distance_matrix(soap_fps)
```

### 输出文件格式 (.npz)

```python
import numpy as np
data = np.load('fp.npz')
print(list(data.keys()))
# ['order', 'fing', 'atom_fing', 'V', 'n_pairs', 'numIons', 'atomType',
#  'rmax', 'sigma', 'delta', 'dimension', 'soap_fp']
```

| 字段          | 形状              | 说明                          |
|---------------|-------------------|-------------------------------|
| `order`       | (N,)              | 每原子有序度                  |
| `fing`        | (n_species², n_bins) | 全局 RDF 指纹矩阵          |
| `atom_fing`   | (N, n_species, n_bins) | 每原子 RDF 指纹           |
| `V`           | scalar            | 晶胞体积 (ų)                  |
| `n_pairs`     | scalar            | 原子对数                      |
| `soap_fp`     | (n_features,)     | SOAP 指纹（仅 --soap 时存在） |
