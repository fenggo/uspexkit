"""Utility classes and functions for USPEX data processing."""
import re

class Stack:
    def __init__(self, entry=None):
        self.entry = entry if entry is not None else []

    def push(self, x):
        self.entry.append(x)

    def pop(self):
        return self.entry.pop()

    def close(self):
        self.entry = None


def read_individuals(g=None):
    """Parse USPEX Individuals file, return list of (index, enthalpy, density, fitness)."""
    gene = {}
    with open("Individuals") as f:
        for line in f.readlines():
            st = Stack([])
            for x in line:
                if x != "]":
                    st.push(x)
                else:
                    x_ = " "
                    while x_ != "[":
                        x_ = st.pop()
            line = "".join(st.entry)
            l = line.split()

            if len(l) >= 10 and l[0] != "Gen":
                g_ = int(l[0])
                i = int(l[1])
                e = float(l[3])
                d = float(l[5])
                f_ = 99999 if l[6] == "N/A" else float(l[6])
                if g_ in gene:
                    gene[g_].append((i, e, d, f_))
                else:
                    gene[g_] = [(i, e, d, f_)]
        st.close()

    k_ = max(gene.keys()) if g is None else g
    return gene[k_]


def search_structure(feature, D, tolerance=0.01):
    """Search for matching structure in feature database."""
    import numpy as np
    if D.ndim == 2:
        res = np.sum(np.square(D - feature), axis=1)
        ind = np.where(res < tolerance)
        imin = np.argmin(res)
        res_ = res[imin]
    else:
        res = np.sum(np.square(D - feature))
        ind = [[0]] if res < tolerance else [[]]
        imin = 0
        res_ = res
    return ind, imin, res_


def generate_hbond_lib(elements, src='reaxff_nn.lib', dst=None,
                       hbond_energy=-10.0):
    '''从 src 复制生成 dst，只保留 elements 指定的氢键项，能量替换为 hbond_energy。
    
    elements: 如 'H core C core O core'，自动推导 dst 文件名后缀。
    若 elements 在原始氢键段中不存在，则以 'H core C core O core' 为模板替换元素名。
    '''
    if dst is None:
        elems = [p.lower() for p in elements.split()
                 if len(p) == 1 and p.isalpha()]
        suffix = 'ch' + elems[2]  # C-H-X 命名约定
        dst = f'reaxff_nn_{suffix}.lib'
    if exists(dst):
        return dst

    template = 'H core C core O core'

    with open(src, 'r') as f:
        lines = f.readlines()

    # 检查 elements 是否在氢键段中存在
    in_hbond = False
    found = False
    for line in lines:
        if line.startswith('reaxff3_hbond'):
            in_hbond = True
            continue
        if in_hbond and (line.startswith('#') or line.startswith('reaxff4')):
            break
        if in_hbond and line.startswith(elements):
            found = True
            break

    in_hbond = False
    hbond_done = False
    with open(dst, 'w') as f:
        for line in lines:
            if line.startswith('reaxff3_hbond'):
                in_hbond = True
                f.write(line)
                continue
            if in_hbond and not hbond_done:
                if found:
                    if line.startswith(elements):
                        # 替换该行第二个数值（能量项），保留原始空格
                        pat = r'^(' + re.escape(elements) + r'\s+[\d.-]+\s+)[\d.-]+'
                        line = re.sub(pat, r'\g<1>' + str(hbond_energy), line)
                        f.write(line)
                        f.write('\n')
                        hbond_done = True
                else:
                    if line.startswith(template):
                        # 用模板行替换元素名 + 能量
                        pat = r'^(' + re.escape(template) + r'\s+[\d.-]+\s+)[\d.-]+'
                        line = re.sub(pat, r'\g<1>' + str(hbond_energy), line)
                        line = line.replace(template, elements, 1)
                        f.write(line)
                        f.write('\n')
                        hbond_done = True
                continue
            if in_hbond and hbond_done:
                if line.startswith('#') or line.startswith('reaxff4'):
                    in_hbond = False
                    f.write(line)
                continue
            if not in_hbond:
                f.write(line)
    return dst

