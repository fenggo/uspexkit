"""Core commands: pred, calc, traj, zmat, fdf, sample."""

import os
import subprocess
import pickle
import numpy as np
from os import getcwd, chdir, mkdir
from os.path import exists

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct, WhiteKernel, Matern,
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from ase.io import read
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator

from uspexkit.utils import read_individuals, search_structure


# ──────────────────────────────────────────────
#  GULP energy helper
# ──────────────────────────────────────────────

def get_gulp_energy(atoms, ncpu=8):
    from irff.md.gulp import opt as gulp_opt, get_reax_energy, write_gulp_in
    atoms_opt = gulp_opt(atoms=atoms, step=1000, l=1, t=0.000001, n=ncpu, lib="reaxff_nn")
    write_gulp_in(atoms_opt, runword="gradient nosymmetry conv qite verb", lib="reaxff_nn")
    if ncpu == 1:
        subprocess.call("gulp<inp-gulp>out", shell=True)
    else:
        subprocess.call(f"mpirun -n {ncpu:d} gulp<inp-gulp>out", shell=True)
    e = get_reax_energy(fo="out")
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses / volume / 0.602214129
    return atoms_opt, e, density


# ──────────────────────────────────────────────
#  ML model helpers
# ──────────────────────────────────────────────

def load_gaussian_process(X, y, y_eng):
    if X.shape[1] == 8:
    length_scale = [0.1 for i in range(X.shape[1])]

    if not exists("gpr_density.pkl"):
        kernel = (
            0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
            + 0.35**2 * Matern(length_scale=length_scale, nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
        )
        gpr_density = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr_density.fit(X, y)
        with open("gpr_density.pkl", "wb") as f:
            pickle.dump(gpr_density, f)
    else:
        with open("gpr_density.pkl", "rb") as f:
            gpr_density = pickle.load(f)

    if not exists("gpr_energy.pkl"):
        kernel = (
            0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
            + 0.35**2 * Matern(length_scale=length_scale, nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
        )
        gpr_energy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr_energy.fit(X, y_eng)
        with open("gpr_energy.pkl", "wb") as f:
            pickle.dump(gpr_energy, f)
    else:
        with open("gpr_energy.pkl", "rb") as f:
            gpr_energy = pickle.load(f)

    with open("gpcsp.log", "w") as fl:
        print(gpr_density.kernel_, file=fl)
        print(gpr_density.log_marginal_likelihood(), file=fl)
        print(gpr_energy.kernel_, file=fl)
        print(gpr_energy.log_marginal_likelihood(), file=fl)

    return gpr_energy, gpr_density


def load_mlp(X, y):
    mlp = MLPRegressor((16, 8), max_iter=20000)
    mlp.fit(X, y)
    return mlp


def load_rfr(X, y):
    rfr = RandomForestRegressor(random_state=37, n_estimators=300,
                                min_weight_fraction_leaf=0.0, oob_score=True)
    rfr.fit(X, y)
    return rfr


# ──────────────────────────────────────────────
#  pred — 高斯过程预测
# ──────────────────────────────────────────────

def pred(t="Individuals.traj", g=None, f=1, den=1.88, ids=None,
         step=300, ncpu=8, dat="data", tolerance=0.001):
    """
    Predict density and energy using Gaussian Process + MLP + RandomForest.

    Args:
        t: trajectory file name
        g: generation number (None = latest)
        f: feature flag (1 = 8D feature, else 7D)
        den: density threshold
        ids: comma/space separated crystal indices
        step: optimization steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
    """
    images = Trajectory(t)
    if not ids:
        ids_list = []
        res = read_individuals(g)
        for i, e, d, _f in res:
            if d > den and _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density_predict.log"):
        with open("density_predict.log", "w") as fd:
            print("# Crystal_id Density_mlp Density_rf Density_gp Energy std_den std_eng", file=fd)

    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        atoms = images[s - 1]

        chdir(data_dir)
        atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
        e_cho = get_hbond_energy(atoms,ncpu=ncpu)
        if f == 1:
            feature = np.array([e[0], e[1], e[5], e[8], e[10], e[11], e[12], density])
        else:
            feature = np.array([e[0], e[5], e[8], e[10], e[11], e[12], density])

        assert exists("structures.traj"), "Error, datafile not found in data directory!"

        data = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
        data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)

        D = data[:, 1:]
        D_ = data_[:, 1:]
        ind, imin, res_ = search_structure(feature, D, tolerance=tolerance)

        X_raw = data[:, 1:]
        y = data_[:, -1]
        y_eng = data_[:, 1]
        scaler = preprocessing.StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)

        gpr_energy, gpr_density = load_gaussian_process(X, y, y_eng)
        rfr = load_rfr(X, y)
        mlp = load_mlp(X, y)

        X_ = scaler.transform(np.expand_dims(feature, axis=0))
        density_pred, std_den_pred = gpr_density.predict(X_, return_std=True)
        energy_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        energy_pred = energy_pred[0]
        density_pred = density_pred[0]
        std_den_pred = std_den_pred[0]
        std_eng_pred = std_eng_pred[0]
        density_rf = rfr.predict(X_)[0]
        density_mlp = mlp.predict(X_)[0]

        if f == 1:
            print(f"{s:5d} rf: {density_rf:9.4f} mlp: {density_mlp:9.4f} "
                  f"{feature[4]:9.4f} {feature[5]:9.4f} {feature[6]:9.4f} {feature[7]:9.4f} "
                  f"gp: {density_pred:7.4f} uncert: {std_den_pred:7.4f}")
        else:
            print(f"{s:5d} rf: {density_rf:9.4f} mlp: {density_mlp:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} {feature[6]:9.4f} "
                  f"gp: {density_pred:7.4f} uncert: {std_den_pred:7.4f}")

        chdir(root_dir)
        with open("density_predict.log", "a") as fd:
            print(f"{s:5d} {density_mlp:9.6f} {density_rf:9.6f} "
                  f"{density_pred:9.6f} {energy_pred:10.6f} "
                  f"{std_den_pred:9.6f} {std_eng_pred:9.6f}", file=fd)


# ──────────────────────────────────────────────
#  calc — DFT 高通量计算
# ──────────────────────────────────────────────

def calc(t="Individuals.traj", den=1.88, ids=None, step=300,
         ncpu=8, dat="data", tolerance=0.01):
    """
    High-throughput DFT calculation with structure matching.

    Args:
        t: trajectory file name
        den: density threshold
        ids: comma/space separated crystal indices
        step: MD steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
    """
    images = Trajectory(t)
    if not ids:
        ids_list = []
        res = read_individuals()
        for i, e, d, _f in res:
            if d > den and _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density.log"):
        with open("density.log", "w") as fd:
            print("# Crystal_id Density Energy", file=fd)

    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        work_dir = os.path.join(root_dir, str(s))
        atoms = images[s - 1]

        if exists(str(s)):
            continue
        else:
            mkdir(str(s))

        chdir(data_dir)
        atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
        feature = np.array([e[0], e[1], e[5], e[8], e[10], e[11], e[12], density])

        if exists("structures.traj"):
            data = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
            data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
            struc = Trajectory("structures.traj")
            try:
                D = data[:, 1:]
                D_ = data_[:, 1:]
            except IndexError:
                D = data[1:]
                D_ = data_[1:]
            ind, imin, res_ = search_structure(feature, D, tolerance=tolerance)
        else:
            ind = [[]]
            with open("feature_mlp.csv", "w") as fd:
                print(", etot, ebond, eang, etor, evdw, ehb, ecoul, density", file=fd)
            with open("feature.csv", "w") as fd_:
                print(", etot, ebond, eang, etor, evdw, ehb, ecoul, density", file=fd_)
            masses = np.sum(atoms.get_masses())
            volume = atoms.get_volume()
            density = masses / volume / 0.602214129
            res_ = 0.0

        chdir(work_dir)
        if len(ind[0]) > 0:
            atoms.write(f"POSCAR.{s}")
            struc[imin].write(f"POSCAR.{s}_opt")
            if D.ndim == 2:
                energy = D_[imin, 0]
                density = D_[imin, 7]
            else:
                energy = D_[0]
                density = D_[7]
            print(f"{s:5d} mt {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")
            traj_w = TrajectoryWriter(f"id_{s}.traj", mode="w")
            traj_w.write(atoms=struc[imin])
            traj_w.close()
        else:
            from irff.dft.siesta import siesta_opt
            subprocess.call(f"cp {rootdir}/Specific/*.psf ./", shell=True)
            img = siesta_opt(atoms, ncpu=ncpu, us="F", VariableCell="true", tstep=step,
                             xcf="GGA", xca="PBE", basistype="split")
            subprocess.call(f"mv siesta.out siesta-{s}.out", shell=True)
            subprocess.call(f"mv siesta.MDE siesta-{s}.MDE", shell=True)
            subprocess.call(f"mv siesta.MD_CAR siesta-{s}.MD_CAR", shell=True)
            subprocess.call(f"mv siesta.traj id_{s}.traj", shell=True)
            subprocess.call("rm siesta.* ", shell=True)
            subprocess.call("rm *.xml ", shell=True)
            subprocess.call("rm INPUT_TMP.* ", shell=True)
            subprocess.call("rm fdf-* ", shell=True)
            img[0].write(f"POSCAR.{s}")
            atoms_opt = img[-1]
            atoms_opt.write(f"POSCAR.{s}_opt")
            masses = np.sum(atoms_opt.get_masses())
            volume = atoms_opt.get_volume()
            density = masses / volume / 0.602214129
            energy = atoms_opt.get_potential_energy()

            print(f"{s:5d} cl {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")

            chdir(data_dir)
            with open("feature_mlp.csv", "a") as fd:
                print(f"0,{feature[0]},{feature[1]},{feature[2]},{feature[3]},"
                      f"{feature[4]},{feature[5]},{feature[6]},{feature[7]}", file=fd)
            with open("feature.csv", "a") as fd:
                print(f"0,{energy},{feature[1]},{feature[2]},"
                      f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{density}", file=fd)

            atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
            with TrajectoryWriter("structures_mlp.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_mlp)
            with TrajectoryWriter("structures.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_opt)

        chdir(root_dir)
        with open("density.log", "a") as fd:
            print(f"{s:5d} {density:10.6f} {energy:10.8f}", file=fd)


# ──────────────────────────────────────────────
#  traj — POSCAR 转 trajectory
# ──────────────────────────────────────────────

def traj(fposcar="gatheredPOSCARS"):
    """Convert gatheredPOSCARS to ASE trajectory file."""
    from uspexkit.utils import Stack

    with open(fposcar) as fbp:
        lines = fbp.readlines()

    traj_w = TrajectoryWriter("Individuals.traj", mode="w")
    k = 0
    s = 0
    energies = []

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
                energies.append(float(l[3]))
        st.close()

    for line in lines:
        if "EA" in line:
            if k > 0:
                fpos.close()
                atoms = read("POSCAR")
                atoms.calc = SinglePointCalculator(atoms, energy=energies[s])
                traj_w.write(atoms=atoms)
                s += 1
            fpos = open("POSCAR", "w")
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)

    fpos.close()
    atoms = read("POSCAR")
    atoms.calc = SinglePointCalculator(atoms, energy=energies[s])
    traj_w.write(atoms=atoms)
    traj_w.close()


# ──────────────────────────────────────────────
#  zmat — 内坐标
# ──────────────────────────────────────────────

def zmat(geo="POSCAR", i=-1):
    """Convert structure to USPEX Z-matrix format."""
    from irff.AtomDance import AtomDance
    atoms = read(geo, index=i)
    ad = AtomDance(atoms=atoms, rcut={"H-O": 2.7, "O-H": 2.7})
    zmat_data = ad.InitZmat
    ad.write_zmat(zmat_data, uspex=True)
    ad.close()


# ──────────────────────────────────────────────
#  fdf — 写 SIESTA 输入
# ──────────────────────────────────────────────

def fdf(gen="poscar.gen", xcf="gga", i=-1):
    """Generate SIESTA input files."""
    from irff.dft.siesta import write_siesta_in
    A = read(gen, index=i)
    print("\n-  writing siesta input ...")
    if xcf == "gga":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="GGA", xca="PBE", basistype="split")
    elif xcf == "vdw":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="VDW", xca="DRSLL", basistype="split")
    else:
        print("Not supported yet!")


# ──────────────────────────────────────────────
#  sample — 采样结构
# ──────────────────────────────────────────────

def sample(ind="", t=None):
    """Sample structures by index to samples.traj."""
    traj_w = TrajectoryWriter("samples.traj", mode="w")

    if ind:
        ids = [int(i) for i in ind.split()]
        if t is not None:
            images = Trajectory(t)
            for i in ids:
                traj_w.write(atoms=images[i])
        else:
            for i in ids:
                atoms = read(f"{i}/POSCAR.{i}_opt")
                atoms.calc = SinglePointCalculator(atoms, energy=0.0)
                traj_w.write(atoms=atoms)

    traj_w.close()
