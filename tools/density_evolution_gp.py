#!/usr/bin/env python
"""
密度演化图 + 高斯过程预测对比图（含置信区间）

GP 预测通过外部调用 uspexkit pred 即时计算，不依赖预存日志。
自动选取每代最高 ML 密度结构进行 GP 预测。

用法:
  # 仅密度演化图（不运行 GP 预测）
  python density_evolution_gp.py

  # 自动对每代最高密度结构做 GP 预测（调用 uspexkit pred）
  python density_evolution_gp.py --gp --n=8 --dat=data11_22

  # 直接从 density_predict.log 读取已有 GP 预测（不重新计算）
  python density_evolution_gp.py --read-log

  # 指定输出文件
  python density_evolution_gp.py --gp --n=8 --dat=data11_22 --out=density.png

GP 模型: sklearn GaussianProcessRegressor
  DotProduct² + Matern(ν=1.5) + WhiteKernel
  置信区间: ±1.96σ (95% CI)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from os import getcwd
from os.path import exists, dirname, abspath, join as pjoin


class Stack():
    def __init__(self, entry=[]):
        self.entry = entry
    def push(self, x): self.entry.append(x)
    def pop(self): return self.entry.pop()
    def close(self): self.entry = None


# ══════════════════════════════════════════════════════════════════
#  解析 density_predict.log（公用）
# ══════════════════════════════════════════════════════════════════

def parse_gp_log(logfile, ids=None, skip_lines=0):
    """
    解析 density_predict.log，返回 GP 预测结果。

    参数
    ----
    logfile : str
        日志文件路径
    ids : set or None
        若提供，只返回这些 ID 的预测；若 None，返回所有 ID
    skip_lines : int
        跳过前 N 行（用于增量解析）

    返回: {id: (density_pred, std_den, energy_pred, std_eng)}
    """
    gp_results = {}
    if not exists(logfile):
        print(f"[WARN] {logfile} not found.")
        return gp_results

    with open(logfile) as f:
        all_lines = f.readlines()

    target_lines = all_lines[skip_lines:]

    for line in target_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            sid = int(parts[0])
            if ids is not None and sid not in ids:
                continue
            if len(parts) >= 8:
                # 8 列格式: ID Residual Density_mlp Density_rf Density_gp std_den Energy std_eng
                density_pred = float(parts[4])  # Density_gp
                std_den      = float(parts[5])  # std_den
                energy_pred  = float(parts[6])  # Energy
                std_eng      = float(parts[7])  # std_eng
            else:
                # 5 列格式 (uspexkit pred 输出): ID energy_pred density_pred std_den std_eng
                density_pred = float(parts[2])
                std_den      = float(parts[3])
                energy_pred  = float(parts[1])
                std_eng      = float(parts[4])
            gp_results[sid] = (density_pred, std_den, energy_pred, std_eng)
        except (ValueError, IndexError):
            continue

    return gp_results


# ══════════════════════════════════════════════════════════════════
#  调用 uspexkit pred 并解析结果
# ══════════════════════════════════════════════════════════════════

def run_uspexkit_pred(ids, dat='data11_22', ncpu=8, traj='Individuals.traj',
                      python_exe=None):
    """
    调用 uspexkit pred 命令对指定 ID 列表进行 GP 预测。

    uspexkit 包已安装在 Anaconda Python 环境中。

    返回: {id: (density_pred, std_den, energy_pred, std_eng)}
    """
    if not ids:
        return {}

    # ── 定位 Python ──
    if python_exe is None:
        for py in ['/home/feng/.local/anaconda/bin/python3',
                   '/home/feng/.local/anaconda/bin/python',
                   '/usr/bin/python3', '/usr/bin/python3.12']:
            if exists(py):
                python_exe = py
                break

    if not python_exe:
        print("[ERROR] No Python interpreter found.")
        return {}

    print(f"[INFO] Python: {python_exe}")

    ids_str = ' '.join(str(i) for i in sorted(ids))
    cmd = (f'uspexkit pred '
           f'--t={traj} --dat={dat} --ncpu={ncpu} '
           f'--ids="{ids_str}"')

    print(f"[INFO] Calling: {cmd}")
    print(f"[INFO] Predicting {len(ids)} structures (GULP optimization, "
          f"may take several minutes per structure)...")

    # ── 记录日志当前行数，只解析新写入的行 ──
    logfile = 'density_predict.log'
    pre_lines = 0
    if exists(logfile):
        with open(logfile) as f:
            pre_lines = sum(1 for _ in f)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            timeout=7200)

    if result.returncode != 0:
        print(f"[ERROR] uspexkit pred failed (exit {result.returncode}):")
        print(result.stderr[-2000:])
        return {}

    print(result.stdout)

    gp_results = parse_gp_log(logfile, ids=ids, skip_lines=pre_lines)
    print(f"[INFO] Got {len(gp_results)}/{len(ids)} GP predictions from "
          f"{logfile}")
    return gp_results


# ══════════════════════════════════════════════════════════════════
#  数据解析
# ══════════════════════════════════════════════════════════════════

def parse(findi='Individuals', penalty=-1144.05,
          do_gp=False, read_log=False, logfile='density_predict.log',
          dat='data11_22', ncpu=8, traj_file='Individuals.traj',
          python_exe=None):
    """
    解析 Individuals 文件，返回每代 ML密度。

    do_gp=True:  自动选取每代最高 ML 密度结构，调用 uspexkit pred 即时计算
    read_log=True: 直接从 density_predict.log 读取已有 GP 预测（不重新计算）

    Returns
    -------
    x_ticks, ml_best, gene_ml, id_to_info, gp_predictions, top_ids_per_gen
    """
    gene_ml = {}
    id_to_info = {}
    # 追踪每代最高 ML 密度对应的 ID
    gene_max_id = {}   # {gen_str: sid}

    # ── 解析 Individuals ──
    with open(findi) as f:
        for line in f.readlines():
            st = Stack([])
            for x in line:
                if x != ']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ != '[':
                        x_ = st.pop()
            line = ''.join(st.entry)
            l = line.split()
            if len(l) >= 10 and l[0] != 'Gen':
                if abs(float(l[3]) - penalty) < 0.001:
                    continue
                gen = l[0]
                sid = int(l[1])
                ml_d = float(l[5])
                gene_ml.setdefault(gen, []).append(ml_d)
                id_to_info[sid] = (gen, ml_d)
                # 追踪最高密度
                if gen not in gene_max_id:
                    gene_max_id[gen] = (sid, ml_d)
                else:
                    if ml_d > gene_max_id[gen][1]:
                        gene_max_id[gen] = (sid, ml_d)
        st.close()

    # ── 提取每代最高密度 ID ──
    gen_keys = sorted(gene_ml.keys(), key=int)
    x_ticks = [int(g) for g in gen_keys]
    ml_best = [max(gene_ml[g]) for g in gen_keys]

    top_ids_per_gen = {}  # {gen: sid}
    for g in gen_keys:
        top_ids_per_gen[int(g)] = gene_max_id[g][0]

    # ── GP 预测 ──
    gp_predictions = {}
    if do_gp:
        pred_ids = set(gene_max_id[g][0] for g in gen_keys)
        print(f"\n[GP] Auto-selected {len(pred_ids)} structures "
              f"(top ML density per generation, {len(gen_keys)} generations)")
        print(f"[GP] IDs: {sorted(pred_ids)}\n")
        gp_predictions = run_uspexkit_pred(
            ids=pred_ids, dat=dat, ncpu=ncpu, traj=traj_file,
            python_exe=python_exe)
    elif read_log:
        pred_ids = set(gene_max_id[g][0] for g in gen_keys)
        print(f"\n[GP] Reading {len(pred_ids)} predictions from {logfile}...")
        gp_predictions = parse_gp_log(logfile, ids=pred_ids)
        print(f"[GP] Found {len(gp_predictions)}/{len(pred_ids)} matching "
              f"predictions in {logfile}")
        if len(gp_predictions) < len(pred_ids):
            missing = pred_ids - set(gp_predictions.keys())
            print(f"[WARN] {len(missing)} IDs not found in log: "
                  f"{sorted(missing)}")

    return x_ticks, ml_best, gene_ml, id_to_info, gp_predictions, top_ids_per_gen


# ══════════════════════════════════════════════════════════════════
#  绑图
# ══════════════════════════════════════════════════════════════════

def plot(x_ticks, ml_best, gene_ml,
         id_to_info, gp_predictions, top_ids_per_gen,
         outfile='density_evolution.svg'):
    """单图：ML 密度演化 + GP 预测点（含置信区间）"""

    fig, ax = plt.subplots(figsize=(16, 8))

    # ═══════════════════════════════════════════════════════════
    # ML 密度演化（背景散点 + 每代最优线）
    # ═══════════════════════════════════════════════════════════
    all_x, all_y_ml = [], []
    for g in x_ticks:
        for d in gene_ml[str(g)]:
            all_x.append(g); all_y_ml.append(d)

    ax.scatter(all_x, all_y_ml, alpha=0.10, color='#3498db', s=8, zorder=1,
               label=r'$Density_{ReaxFF-nn}\ (all valid structures)$')

    ax.scatter(x_ticks, ml_best, color='#2980b9', marker='o', s=80,
               zorder=3, edgecolors='white', linewidth=1.0,
               label=r'$Density_{ReaxFF-nn}\ (highest)$')
    ax.plot(x_ticks, ml_best, color='#2980b9', alpha=0.6, linewidth=2.0,
            zorder=2)

    # ── 标注每代最高密度的 ID ──
    for g in x_ticks:
        if g in top_ids_per_gen:
            sid = top_ids_per_gen[g]
            idx = x_ticks.index(g)
            ax.annotate(str(sid), (g, ml_best[idx]),
                        textcoords="offset points", xytext=(0, -14),
                        fontsize=6, color='#2980b9', ha='center', alpha=0.5)

    idx_ml = np.argmax(ml_best)
    ax.annotate(f'MLP(ReaxFF-nn) max: {ml_best[idx_ml]:.3f} @ Gen {x_ticks[idx_ml]}',
                xy=(x_ticks[idx_ml], ml_best[idx_ml]),
                xytext=(x_ticks[idx_ml] - 2.5, ml_best[idx_ml] + 0.015),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5),
                fontsize=10, color='#2980b9', fontweight='bold')

    # ═══════════════════════════════════════════════════════════
    # GP 预测点（来自 density_predict.log）
    # ═══════════════════════════════════════════════════════════
    gp_gen_x = []
    gp_gen_y_pred = []
    gp_gen_y_std = []
    gp_gen_y_ml = []
    gp_gen_ids = []  # 直接记录 ID，避免密度值相同时匹配错误

    if gp_predictions:
        for sid, (den_pred, std_den, eng_pred, std_eng) in \
                gp_predictions.items():
            if sid in id_to_info:
                gen_str, ml_d = id_to_info[sid]
                gp_gen_x.append(int(gen_str))
                gp_gen_y_pred.append(den_pred)
                gp_gen_y_std.append(std_den)
                gp_gen_y_ml.append(ml_d)
                gp_gen_ids.append(sid)

    if len(gp_gen_x) > 0:
        gp_gen_x = np.array(gp_gen_x)
        gp_gen_y_pred = np.array(gp_gen_y_pred)
        gp_gen_y_std = np.array(gp_gen_y_std)
        gp_gen_y_ml = np.array(gp_gen_y_ml)
        gp_gen_ids = np.array(gp_gen_ids)

        sort_idx = np.argsort(gp_gen_x)
        gp_gen_x = gp_gen_x[sort_idx]
        gp_gen_y_pred = gp_gen_y_pred[sort_idx]
        gp_gen_y_std = gp_gen_y_std[sort_idx]
        gp_gen_y_ml = gp_gen_y_ml[sort_idx]
        gp_gen_ids = gp_gen_ids[sort_idx]

        ci_upper = gp_gen_y_pred + 1.96 * gp_gen_y_std
        ci_lower = gp_gen_y_pred - 1.96 * gp_gen_y_std

        ax.fill_between(gp_gen_x, ci_lower, ci_upper,
                        alpha=0.15, color='#27ae60',
                        label='GP 95% confidence interval')

        ax.plot(gp_gen_x, gp_gen_y_pred, color='#27ae60', linewidth=2.5,
                marker='D', markersize=8, zorder=4,
                label='GP Predicted Density (sklearn GPR)')

        ax.errorbar(gp_gen_x, gp_gen_y_pred,
                    yerr=1.96 * gp_gen_y_std,
                    fmt='none', ecolor='#27ae60', alpha=0.4,
                    capsize=3, linewidth=0.8, zorder=3)

        # ── GP 预测点标注 ID ──
        for i in range(len(gp_gen_x)):
            sid = gp_gen_ids[i]
            ax.annotate(f'ID:{sid}', (gp_gen_x[i], gp_gen_y_pred[i]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, color='#27ae60', ha='center',
                        alpha=0.8)

        # ── GP 最高密度标注 ──
        idx_gp = np.argmax(gp_gen_y_pred)
        sid_gp = gp_gen_ids[idx_gp]
        ax.annotate(f'GP max: {gp_gen_y_pred[idx_gp]:.3f} @ Gen {gp_gen_x[idx_gp]} (ID:{sid_gp})',
                    xy=(gp_gen_x[idx_gp], gp_gen_y_pred[idx_gp]),
                    xytext=(gp_gen_x[idx_gp] - 2.5, gp_gen_y_pred[idx_gp] + 0.035),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                    fontsize=10, color='#27ae60', fontweight='bold')

        # ── 统计输出 ──
        residuals = gp_gen_y_pred - gp_gen_y_ml
        print(f"\n{'='*60}")
        print(f"  GP 预测统计 ({len(gp_gen_x)} 个结构，每代最高密度)")
        print(f"{'='*60}")
        print(f"  残差 (GP_pred − ML):")
        print(f"    min={residuals.min():.4f}  max={residuals.max():.4f}  "
              f"mean={residuals.mean():.4f}  std={residuals.std():.4f}")
        print(f"  GP 预测密度:  min={gp_gen_y_pred.min():.4f}  "
              f"max={gp_gen_y_pred.max():.4f}  mean={gp_gen_y_pred.mean():.4f}")
        print(f"  平均不确定性 σ:    {gp_gen_y_std.mean():.4f} g/cm³")
        print(f"  平均 95% CI 半宽:  {1.96 * gp_gen_y_std.mean():.4f} g/cm³")
        print(f"  {'ID':>6} {'Gen':>4} {'ML':>8} {'GP-pred':>8} "
              f"{'±1.96σ':>8} {'Δ(ML−pred)':>12}")
        print(f"  {'-'*52}")
        for i in range(len(gp_gen_x)):
            sid = gp_gen_ids[i]
            print(f"  {sid!s:>6} {gp_gen_x[i]:4d} {gp_gen_y_ml[i]:8.3f} "
                  f"{gp_gen_y_pred[i]:8.3f} "
                  f"{1.96*gp_gen_y_std[i]:8.3f} "
                  f"{residuals[i]:12.4f}")

        # ── Y轴范围 ──
        all_vals = list(all_y_ml) + list(gp_gen_y_pred) + list(ci_upper) + list(ci_lower)
        ax.set_ylim(1.65, max(all_vals) + 0.04)
    else:
        ax.set_ylim(1.65, max(all_y_ml) + 0.04)

    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel(r'Density (g/cm$^3$)', fontsize=13)
    ax.set_title(r'USPEX Density Evolution — ML + GP Prediction  |  '
                 r'TNT$_4$·CL20$_4$', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(x_ticks)

    plt.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Saved → {outfile}")

    # ── 总体统计 ──
    ml_arr = [d for v in gene_ml.values() for d in v]
    print(f"\n{'='*60}")
    print(f"  总体统计 ({len(ml_arr)} 个结构)")
    print(f"{'='*60}")
    print(f"  ML density:   min={min(ml_arr):.4f}  max={max(ml_arr):.4f}  "
          f"mean={np.mean(ml_arr):.4f}")

    print(f"\n  {'Gen':>4}  {'ML #1':>8} {'ML_mean':>8}  {'TopID':>6}")
    print(f"  {'-'*38}")
    for i, g in enumerate(x_ticks):
        ml_vals = gene_ml[str(g)]
        top_sid = top_ids_per_gen.get(g, '?')
        print(f"  {g:4d}  {ml_best[i]:8.3f} {np.mean(ml_vals):8.3f}  "
              f"{top_sid!s:>6}")


# ══════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='密度演化图 + 高斯预测对比（自动选每代最高密度，含置信区间）')
    parser.add_argument('--individuals', default='Individuals',
                        help='Individuals 文件路径')
    parser.add_argument('--gp', action='store_true', default=False,
                        help='启用 GP 预测（自动选每代最高 ML 密度结构）')
    parser.add_argument('--read-log', action='store_true', default=False,
                        help='直接从 density_predict.log 读取已有 GP 预测（不重新计算）')
    parser.add_argument('--n', type=int, default=8,
                        help='GULP 并行核数 (default: 8)')
    parser.add_argument('--dat', default='data11_22',
                        help='数据目录名（相对于 USPEX 根目录）')
    parser.add_argument('--out', default='density_evolution.svg',
                        help='输出图片文件名')
    parser.add_argument('--penalty', type=float, default=-1144.05,
                        help='Enthalpy penalty 值')
    parser.add_argument('--traj', default='Individuals.traj',
                        help='结构轨迹文件')
    parser.add_argument('--python', default=None,
                        help='uspexkit 所在的 Python 解释器 '
                             '(default: 自动检测 Anaconda Python)')

    args = parser.parse_args()

    # ── 解析 + GP 预测 ──
    x_ticks, ml_best, gene_ml, id_to_info, \
        gp_predictions, top_ids_per_gen = \
        parse(findi=args.individuals,
              penalty=args.penalty,
              do_gp=args.gp,
              read_log=args.read_log,
              logfile='density_predict.log',
              dat=args.dat,
              ncpu=args.n,
              traj_file=args.traj,
              python_exe=args.python)

    # ── 绑图 ──
    plot(x_ticks, ml_best, gene_ml,
         id_to_info, gp_predictions, top_ids_per_gen,
         outfile=args.out)