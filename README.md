# uspexkit — USPEX 晶体结构预测后处理工具包

`uspexkit` 是 USPEX 分子晶体结构预测（计算类型 310）的后处理工具包，提供轨迹转换、特征提取、高斯过程预测、高通量 DFT 筛选、破损分子修复等功能。

## 安装

```bash
git clone https://github.com/FengGo/uspexkit.git
cd uspexkit
pip install .
```

依赖：`numpy>=1.20`, `scikit-learn>=1.0`, `ase>=3.22`，以及内部库 `irff`（提供 GULP / LAMMPS / SIESTA 接口）。

---

## 命令总览

| 命令 | 功能 |
|------|------|
| `traj` | 将 `gatheredPOSCARS` 转换为 ASE 轨迹文件 |
| `calcdata` | 从轨迹批量计算晶体特征向量 |
| `gp` | 高斯过程预测晶体密度（USPEX 流水线内调用） |
| `pred` | 高斯过程 + 随机森林预测密度和能量 |
| `calc` | 高通量 DFT（SIESTA）计算 + 结构匹配去重 |
| `fixbroken` | 修复破损分子 |
| `add` | 将单个结构添加到特征数据库 |
| `addall` | 将轨迹中所有结构批量添加到特征数据库 |
| `zmat` | 结构坐标转 USPEX Z-matrix 内坐标 |
| `fdf` | 生成 SIESTA 输入文件 |
| `sample` | 按索引采样结构输出到轨迹 |
| `supercell` | 构建超胞 |

---

## 1. `traj` — 轨迹转换

将 USPEX 输出文件 `gatheredPOSCARS` 转换为 ASE `.traj` 轨迹文件，同时从 `Individuals` 文件中解析能量信息。

```bash
uspexkit traj [--fposcar FILE]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fposcar` | `gatheredPOSCARS` | 输入 POSCAR 文件路径 |

### 输出

- `Individuals.traj` — ASE 轨迹文件，包含所有结构及能量

### 工作原理

解析 `gatheredPOSCARS` 中的每个结构（以 `EA` 行分隔），写入 `POSCAR` 后用 ASE 读取，再从 `Individuals` 中匹配对应焓值写入 `SinglePointCalculator`。

---

## 2. `calcdata` — 计算特征向量

从 ASE 轨迹中读取结构，通过 MLP（神经网络势）或 MTP（矩张量势）弛豫后，用 GULP 计算能量分解特征，输出到 CSV 数据库。

```bash
uspexkit calcdata [--t TRAJ] [--n NCPU] [--c CALC] [--step STEPS]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 输入轨迹文件 |
| `--n` | `1` | 并行 CPU 数 |
| `--c` | `nn` | 计算器类型：`nn`（神经网络反应势）或 `mtp`（MTP 势） |
| `--step` | `1000` | MLP 弛豫步数 |

### 输出

| 文件 | 内容 |
|------|------|
| `feature_mlp.csv` | 10 维特征（GULP 总能 + 能量分解 + 密度） |
| `feature.csv` | 10 维特征（DFT 总能 + GULP 能量分解 + 密度） |
| `structures_mlp.traj` | MLP 弛豫后的结构轨迹 |

### 特征向量 (10 维)

| 维度 | 含义 |
|------|------|
| 0 | 总能 (etot) |
| 1 | 键能 (ebond) |
| 2 | 角能 (eang) |
| 3 | 扭转能 (etor) |
| 4 | 范德华能 (evdw) |
| 5 | 氢键能 C-H-O (ehb_cho) |
| 6 | 氢键能 C-H-N (ehb_chn) |
| 7 | 氢键能 C-H-C (ehb_chc) |
| 8 | 库仑能 (ecoul) |
| 9 | 密度 (density) |

---

## 3. `gp` — 高斯过程预测（USPEX 流水线）

在 USPEX 进化搜索内部调用：对当前结构做 GULP 梯度弛豫 → 提取特征 → GP + RF 预测密度/能量 → 将预测值写回 USPEX 输出格式。

```bash
uspexkit gp [--n NCPU] [--t TOL] [--step STEPS] [--b BROKEN] [--u UNCERT] [--f FEAT] [--data DIR] [--resf DIR]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | `1` | 并行 CPU 数 |
| `--t` | `0.005` | 结构匹配容差 |
| `--step` | `1000` | MLP 弛豫步数 |
| `--b` | `1.5` | 破损判断阈值：当前能量偏离均值超过此值视为破损 |
| `--u` | `0.2` | GP 不确定性阈值 |
| `--f` | `1` | 特征标志：`1` = 10 维（含氢键），其他 = 7 维 |
| `--data` | `data` | 训练数据目录名 |
| `--resf` | `results1` | 结果输出目录名 |

### 输出

| 文件 | 内容 |
|------|------|
| `output` | USPEX 格式的能量输出 |
| `optimized.structure` | USPEX 格式的优化结构 |
| `gpr_density.pkl` / `gpr_energy.pkl` | 训练好的 GP 模型 |
| `rfr_density.pkl` | 训练好的随机森林模型 |
| `gp.csv` (在 `results1/` 下) | 预测日志 |

### 工作原理

1. GULP 梯度弛豫当前结构
2. 计算 10 维特征（含 C-H-O / C-H-N / C-H-C 氢键）
3. 从训练数据目录加载 `feature_mlp.csv` / `feature.csv` / `structures.traj`
4. 训练 GP 密度模型 + GP 能量模型 + RF 密度模型（首次训练后缓存为 `.pkl`）
5. 对新结构预测密度和能量，输出到 USPEX 格式
6. 若与最近邻残差 > 10 且 RF 预测偏差大，则回退到缩放修正

---

## 4. `pred` — 预测密度/能量

对指定结构（按索引或 POSCAR 文件）预测密度和能量，使用已训练的 GP + RF 模型。

```bash
uspexkit pred [--t TRAJ] [--g GEO] [--f FEAT] [--den DENSITY] [--ids IDS] [--x INDEX] [--c CALC] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `Individuals.traj` | 轨迹文件 |
| `--g` | `None` | 几何结构文件（如 `POSCAR`），指定后直接预测该文件 |
| `--f` | `1` | 特征标志：`1` = 10 维（含氢键），其他 = 7 维 |
| `--den` | `1.88` | 密度阈值：只预测密度高于此值的结构 |
| `--ids` | `None` | 晶体索引（空格分隔），如 `"214 215"` |
| `--x` | `-1` | 单个结构索引（`-1` 表示最后一个） |
| `--c` | `nn` | 计算器：`nn`（神经网络势）或 `mtp`（MTP 势） |
| `--step` | `300` | MLP 弛豫步数 |
| `--ncpu` | `8` | 并行 CPU 数 |
| `--dat` | `data` | 训练数据目录名 |
| `--tolerance` | `0.001` | 结构匹配容差 |

### 输出

- `density_predict.log` — 预测日志，每行包含：结构ID、残差、密度（MLP/RF/GP）、GP 不确定性、能量预测

### 使用示例

```bash
# 预测指定索引的结构
uspexkit pred --ids="214 215" --n=24 --dat=data11_44

# 从 POSCAR 文件预测
uspexkit pred --g=POSCAR --n=24 --dat=data11_44
```

---

## 5. `calc` — 高通量 DFT 计算

对符合条件的结构执行 SIESTA DFT 计算，支持结构匹配去重（已计算过的结构自动跳过）。

```bash
uspexkit calc [--t TRAJ] [--den DENSITY] [--ids IDS] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `Individuals.traj` | 轨迹文件 |
| `--den` | `1.88` | 密度阈值：只计算密度高于此值的结构 |
| `--ids` | `None` | 晶体索引（空格分隔） |
| `--step` | `300` | GULP 弛豫步数 |
| `--ncpu` | `8` | 并行 CPU 数 |
| `--dat` | `data` | 训练数据目录名 |
| `--tolerance` | `0.01` | 结构匹配容差 |

### 输出

- `density.log` — 计算日志
- `{id}/` — 每个结构的独立工作目录，包含 DFT 输入输出
- `{id}/POSCAR.{id}` — 原始结构
- `{id}/POSCAR.{id}_opt` — DFT 优化后结构
- `{id}/id_{id}.traj` — DFT 优化轨迹

### 工作原理

1. 从 `Individuals` 读取结构列表，筛选 `density > den` 且 `fitness < 0` 的结构
2. 对每个结构：GULP 弛豫 → 计算 10 维特征 → 与数据库匹配
3. 若匹配到已知结构（残差 < tolerance）：直接复用已有 DFT 结果
4. 若未匹配：执行 SIESTA DFT 优化（GGA-PBE），并将结果追加到数据库

---

## 6. `fixbroken` — 修复破损分子

检测当前结构是否破损（能量偏离均值超过阈值），若破损则通过扩大晶胞并重新弛豫尝试修复。

```bash
uspexkit fixbroken [--n NCPU] [--data DIR] [--s SCALE] [--b BROKEN]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | `1` | 并行 CPU 数 |
| `--data` | `data` | 训练数据目录名 |
| `--s` | `1.2` | 晶胞放大因子（每次迭代乘以该值） |
| `--b` | `1.5` | 破损判断阈值（eV）：当前能量偏离训练数据均值超过此值视为破损 |

### 输出

- `output` — USPEX 格式能量输出
- `optimized.structure` — 优化后结构

### 工作原理

1. GULP 梯度弛豫当前结构
2. 若 `E_mean_train - E_current > broken`：读取分子片段 → 逐步放大晶胞（×1.2，最多 15 次）→ 重新弛豫直到能量恢复正常
3. 若未破损：首次运行时会识别并缓存分子片段（`molecule.pkl`）

---

## 7. `add` — 添加单个结构

将单个结构（DFT 优化后的）追加到特征数据库。

```bash
uspexkit add [--t TRAJ] [--i INDEX] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 轨迹文件 |
| `--i` | `-1` | 结构索引（`-1` 表示最后一个） |
| `--s` | `1000` | MLP 弛豫步数 |
| `--tolerance` | `0.005` | 结构匹配容差（去重） |
| `--n` | `1` | 并行 CPU 数 |

### 输出

更新 `feature_mlp.csv`、`feature.csv`、`structures_mlp.traj`、`structures.traj`。

---

## 8. `addall` — 批量添加结构

将轨迹中所有结构批量追加到特征数据库。

```bash
uspexkit addall [--t TRAJ] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 输入轨迹文件 |
| `--s` | `1000` | MLP 弛豫步数 |
| `--tolerance` | `0.005` | 结构匹配容差（去重） |
| `--n` | `1` | 并行 CPU 数 |

---

## 9. `zmat` — Z-matrix 内坐标

将笛卡尔坐标结构转换为 USPEX 格式的 Z-matrix 内坐标文件（`MOL_*`）。

```bash
uspexkit zmat [--geo GEO] [--i INDEX]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--geo` | `POSCAR` | 输入几何文件 |
| `--i` | `-1` | 帧索引（`-1` = 最后一帧） |

### 输出

- USPEX 格式的 `MOL_*` 内坐标文件

### 使用示例

```bash
uspexkit zmat --g=POSCAR
```

---

## 10. `fdf` — 生成 SIESTA 输入

从结构文件生成 SIESTA DFT 的 `.fdf` 输入文件。

```bash
uspexkit fdf [--gen GEN] [--xcf XCF] [--i INDEX]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gen` | `poscar.gen` | 输入 `.gen` 格式结构文件 |
| `--xcf` | `gga` | 交换关联泛函：`gga`（GGA-PBE）或 `vdw`（VDW-DRSLL） |
| `--i` | `-1` | 帧索引 |

---

## 11. `sample` — 采样结构

按索引从轨迹或 DFT 结果目录中提取指定结构。

```bash
uspexkit sample [--ind INDICES] [--t TRAJ]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ind` | `""` | 结构索引（空格分隔），如 `"0 5 12"` |
| `--t` | `None` | 轨迹文件路径。若指定则从轨迹中提取；若不指定则从 `{i}/POSCAR.{i}_opt` 中读取 |

### 输出

- `samples.traj` — 采样的结构轨迹

---

## 12. `supercell` — 构建超胞

从结构或轨迹构建超胞。

```bash
uspexkit supercell [--x NX] [--y NY] [--z NZ] [--t TRAJ] [--g GEO]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--x` | `1` | X 方向倍数 |
| `--y` | `1` | Y 方向倍数 |
| `--z` | `1` | Z 方向倍数 |
| `--t` | `None` | 轨迹文件（取最后一帧） |
| `--g` | `None` | 几何文件（如 `POSCAR`） |

### 输出

- 若指定 `--g`：输出 `POSCAR.supercell_{x}_{y}_{z}`
- 若指定 `--t`：输出 `{traj_name}_{x}{y}{z}.traj`，能量按体积比例缩放

---

## 数据格式说明

### `feature_mlp.csv` (GULP 能量特征)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -123.45, -56.78, ...
```

第一列为结构索引，后续为 10 维特征向量。

### `feature.csv` (DFT 能量 + GULP 分解)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -234.56, -56.78, ...
```

第一列为 DFT 总能（来自 `SinglePointCalculator`），后续为 GULP 分解能量。

### GP 模型文件

- `gpr_density.pkl` — 高斯过程密度模型
- `gpr_energy.pkl` — 高斯过程能量模型
- `rfr_density.pkl` — 随机森林密度模型

核函数：`0.00581² · DotProduct(σ₀=0.412) + 0.35² · Matern(ν=2.5, ARD) + WhiteKernel`

---

## 典型工作流

### 1. 构建训练数据库

```bash
# 从 DFT 结果生成特征数据库
uspexkit calcdata --t=structures.traj --n=24

# 或手动添加结构
uspexkit add --t=structures.traj --i=-1 --n=24
```

### 2. 进化搜索中预测

在 USPEX 的 `command` 中配置：
```bash
uspexkit gp --n=24 --data=data11_44 --resf=results1
```

### 3. 高通量 DFT 筛选

```bash
uspexkit calc --n=24 --den=1.88 --dat=data11_44
```

### 4. 破损分子修复

```bash
uspexkit fixbroken --n=24 --data=data11_44 --b=1.5
```
