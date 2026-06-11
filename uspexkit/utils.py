"""Utility classes and functions for USPEX data processing."""


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
