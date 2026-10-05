"""Command-line interface for uspexkit."""

import argparse
import sys
import numpy as np
from uspexkit.core import pred, calc, traj, zmat, fdf, sample,calcdata,gp,fixbroken,add,addall,supercell,update,info,fingerprint,lib,ffield,molinfo,pack
from uspexkit.md2pdf import md2pdf
from uspexkit.denevo import denevo
from uspexkit.smd import smd

COMMANDS = {
    "pred": (pred, "Predict density/energy using Gaussian Process regression"),
    "calc": (calc, "High-throughput DFT calculation with structure matching"),
    "traj": (traj, "Convert gatheredPOSCARS to ASE trajectory file"),
    "zmat": (zmat, "Convert structure to USPEX Z-matrix format"),
    "fdf":  (fdf,  "Generate SIESTA input files"),
    "sample": (sample, "Sample structures by index to trajectory"),
    "calcdata": (calcdata, "calculate the feature vector of crystal structures"),
    "gp": (gp, "Gaussian process to predict the crystal density"),
    "fixbroken": (fixbroken, "fix broken molecule"),
    "add": (add, "add a structure to data"),
    "addall": (addall, "add a structure to data"),
    "supercell": (supercell, "build a supercell"),
    "update": (update, "update a structure to data"),
    "info": (info, "Print energy and lattice information of a structure"),
    "fingerprint": (fingerprint, "Compute USPEX structural fingerprint (Cython accelerated)"),
    "lib": (lib, "Convert ffield.json to reaxff_nn.lib"),
    "ffield": (ffield, "Convert ffield.json to ReaxFF ffield"),
    "molinfo": (molinfo, "Print molecule atom indices for LAMMPS/COLVARS"),
    "pack": (pack, "Pack POSCAR.* files into a USPEX gatheredPOSCARS file"),
    "md2pdf": (md2pdf, "Convert Markdown to PDF"),
    "denevo": (denevo, "Plot per-generation density evolution (GP/EI/DFT)"),
    "smd": (smd, "SIESTA MD/optimization workflow (use one action flag: --opt/--fdf/--traj/--md/...)"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="uspexkit",
        description="USPeX Kit — USPEX crystal structure prediction post-processing toolkit",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # ── pred ──
    p_pred = sub.add_parser("pred", help=COMMANDS["pred"][1])
    p_pred.add_argument("--t", default="Individuals.traj", help="Trajectory file")
    p_pred.add_argument("--g", type=str, default=None, help="geometry structure")
    p_pred.add_argument("--gen", type=int, default=None, help="Generation number")
    p_pred.add_argument("--f", type=int, default=1, help="Feature flag (1=8D)")
    p_pred.add_argument("--x", type=int, default=-1, help="index")
    p_pred.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_pred.add_argument("--ids", default=None, help="Crystal indices (space-separated)")
    p_pred.add_argument("--step", type=int, default=300, help="Optimization steps")
    p_pred.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_pred.add_argument("--data", "--dat", dest="data", default="data", help="Data directory name")
    p_pred.add_argument("--tolerance", type=float, default=0.001, help="Structure matching tolerance")
    p_pred.add_argument("--c",type=str, default='nn', help="the calculator to be used, aviliable: nn, mtp")

    # ── calc ──
    p_calc = sub.add_parser("calc", help=COMMANDS["calc"][1])
    p_calc.add_argument("--t", default="Individuals.traj", help="Trajectory file")
    p_calc.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_calc.add_argument("--ids", default=None, help="Crystal indices")
    p_calc.add_argument("--step", type=int, default=300, help="MD steps")
    p_calc.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_calc.add_argument("--data", "--dat", dest="data", default="data", help="Data directory name")
    p_calc.add_argument("--tolerance", type=float, default=0.01, help="Structure matching tolerance")
    p_calc.add_argument("--gen", type=int, default=None, help="Generation number")

    # ── traj ──
    p_traj = sub.add_parser("traj", help=COMMANDS["traj"][1])
    p_traj.add_argument("--fposcar", default="gatheredPOSCARS", help="Input POSCAR file")

    # ── zmat ──
    p_zmat = sub.add_parser("zmat", help=COMMANDS["zmat"][1])
    p_zmat.add_argument("--geo", default="POSCAR", help="Input geometry file")
    p_zmat.add_argument("--i", type=int, default=-1, help="Frame index")

    # ── fdf ──
    p_fdf = sub.add_parser("fdf", help=COMMANDS["fdf"][1])
    p_fdf.add_argument("--gen", default="poscar.gen", help="Input .gen file")
    p_fdf.add_argument("--xcf", default="gga", choices=["gga", "vdw"], help="XC functional")
    p_fdf.add_argument("--i", type=int, default=-1, help="Frame index")

    # ── sample ──
    p_sample = sub.add_parser("sample", help=COMMANDS["sample"][1])
    p_sample.add_argument("--ind", default="", help="Indices (space-separated)")
    p_sample.add_argument("--t", default=None, help="Trajectory file")

    # ── calcdata ──
    p_calcdata = sub.add_parser("calcdata", help=COMMANDS["calcdata"][1])
    p_calcdata.add_argument("--n",type=int, default=1, help="number cpu tobe used")
    p_calcdata.add_argument("--t", default='structures.traj', help="Trajectory file")
    p_calcdata.add_argument("--step",type=int,  default=1000, help="number of step to used to optimize by MLP")
    p_calcdata.add_argument("--c",type=str, default='nn', help="the calculator to be used, aviliable: nn, mtp")

   # ── gp ──  
    p_gp = sub.add_parser("gp", help=COMMANDS["gp"][1])
    p_gp.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_gp.add_argument("--t",type=float, default=0.005, help="structure match tolerance")
    p_gp.add_argument("--step",type=int, default=1000, help="number of step to used to optimize by MLP")
    p_gp.add_argument("--b",type=float, default=1.5, help="energy devate the mean tolerance that the structure is broken")
    p_gp.add_argument("--u",type=float, default=0.03, help="uncertainty of Gaussian Process")
    p_gp.add_argument("--f", type=int,default=1, help="which feature factor to be used")
    p_gp.add_argument("--dft", type=int,default=0, help="whether using active learning and calling DFT")
    p_gp.add_argument("--den", type=float,default=1.82, help="density  criteria to use active learning and calling DFT")
    p_gp.add_argument("--pop", type=int,default=100, help="the population size")
    p_gp.add_argument("--data", "--dat", dest="data", default='data',
                      help="which data to be used (alias: --dat)")
    p_gp.add_argument("--ref", default='results1', help="results file directory")
    p_gp.add_argument("--k", type=int, default=1,
                      help="Top-K crystals for EI active learning (default: 1)")
    p_gp.add_argument("--mode", default="ei",
                      help="Selection mode for active learning: ei, random, uncertainty (default: ei)")
    p_gp.add_argument("--optype", type=int, default=1,
                      help="USPEX optType passed from submitJob")
    p_gp.add_argument("--id", type=int, default=None,
                      help="crystal global ID (bodyCount+1) passed by USPEX submitJob")

 # ── fixbroken ── 
    p_fixbroken = sub.add_parser("fixbroken", help=COMMANDS["fixbroken"][1])
    p_fixbroken.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_fixbroken.add_argument("--data", "--dat", dest="data", default='data', help="which data to be used")
    p_fixbroken.add_argument("--s", type=float,default=1.2, help="scale factor")
    p_fixbroken.add_argument("--b", type=float,default=1.5, help="energy devate the mean tolerance that the structure is broken")

 # ── add ── 
    p_add = sub.add_parser("add", help=COMMANDS["add"][1])
    p_add.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_add.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_add.add_argument("--i", type=int, default=-1, help="the index of the Atoms object in trajectory")
    p_add.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_add.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
 # ── update ── 
    p_update = sub.add_parser("update", help=COMMANDS["update"][1])
    p_update.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_update.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_update.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_update.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
    p_update.add_argument("--i", default=None, help="Crystal indices (space-separated)")
 # ── addall ── 
    p_addall = sub.add_parser("addall", help=COMMANDS["addall"][1])
    p_addall.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_addall.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_addall.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_addall.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
 # ── supercell ── 
    p_supercell = sub.add_parser("supercell", help=COMMANDS["supercell"][1])
    p_supercell.add_argument("--x", type=int, default=1, help="X")
    p_supercell.add_argument("--y", type=int, default=1, help="Y")
    p_supercell.add_argument("--z", type=int, default=1, help="Z")
    p_supercell.add_argument("--t", type=str,default=None, help="trajector file name")
    p_supercell.add_argument("--g", type=str,default=None, help="geometry file name")

 # ── info ──
    p_info = sub.add_parser("info", help=COMMANDS["info"][1])
    p_info.add_argument("--gen", default=None, help="Geometry file (e.g. POSCAR, gulp.cif)")
    p_info.add_argument("--traj", default=None, help="Trajectory file name")
    p_info.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_info.add_argument("--symmetry", action="store_true", default=True,
                        help="Perform pymatgen symmetry analysis (default: on)")
    p_info.add_argument("--no-symmetry", action="store_false", dest="symmetry",
                        help="Disable symmetry analysis")
    p_info.add_argument("--symprec", type=float, default=0.1,
                        help="Symmetry tolerance for pymatgen (default: 0.1)")

 # ── fingerprint ──
    p_fp = sub.add_parser("fingerprint", help=COMMANDS["fingerprint"][1])
    p_fp.add_argument("--g", default=None, help="Geometry structure file (e.g. POSCAR)")
    p_fp.add_argument("--traj", default=None, help="Trajectory file name")
    p_fp.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_fp.add_argument("--rmax", type=float, default=12.0, help="Cutoff radius Rmax (Å)")
    p_fp.add_argument("--sigma", type=float, default=0.05, help="Gaussian broadening sigma")
    p_fp.add_argument("--delta", type=float, default=0.08, help="Bin width delta (Å)")
    p_fp.add_argument("--dimension", type=int, default=3, help="Dimension: 3=3D, 0=cluster, 2=2D")
    p_fp.add_argument("--output", default=None, help="Output .npz file (optional)")
    p_fp.add_argument("--intra-map", default=None,
                      help="Intra-molecular distance map (.npy/.npz) for filtering "
                           "intra-molecular pairs. Only zero-shift (basic-cell) pairs "
                           "are filtered; periodic-image pairs are always kept.")
    p_fp.add_argument("--soap", action="store_true",
                      help="Also compute SOAP fingerprint (dscribe). "
                           "Captures local angular environment for better "
                           "discrimination of molecular crystal polymorphs.")
    p_fp.add_argument("--soap-r-cut", type=float, default=6.0,
                      help="SOAP local-environment cutoff radius (Å)")
    p_fp.add_argument("--soap-n-max", type=int, default=8,
                      help="SOAP number of radial basis functions")
    p_fp.add_argument("--soap-l-max", type=int, default=6,
                      help="SOAP maximum angular momentum")

    # ── lib ──
    p_lib = sub.add_parser("lib", help=COMMANDS["lib"][1])
    p_lib.add_argument("--json", default="ffield.json", help="Path to ffield.json")
    p_lib.add_argument("--lib", default="reaxff_nn.lib", help="Output lib file name")

    # ── ffield ──
    p_ffield = sub.add_parser("ffield", help=COMMANDS["ffield"][1])
    p_ffield.add_argument("--json", default="ffield.json", help="Path to ffield.json")
    p_ffield.add_argument("--ffield", default="ffield", help="Output ffield file name")

    # ── molinfo ──
    p_molinfo = sub.add_parser("molinfo", help=COMMANDS["molinfo"][1])
    p_molinfo.add_argument("--g", "--gen", dest="gen", default="data.traj",
                           help="Geometry file (ASE-readable: traj, POSCAR, lammps-data, etc.)")
    p_molinfo.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_molinfo.add_argument("--json", dest="jsonfile", default=None,
                           help="Path to ffield.json for bond cutoffs (optional)")
    p_molinfo.add_argument("--quiet", action="store_true", default=False,
                           help="Suppress verbose output (print only atom indices)")

    # ── md2pdf ──
    p_md2pdf = sub.add_parser("md2pdf", help=COMMANDS["md2pdf"][1])
    p_md2pdf.add_argument("--i", dest="input", required=True,
                          help="Input file (with or without .md extension)")

    # ── pack ──
    p_pack = sub.add_parser("pack", help=COMMANDS["pack"][1])
    p_pack.add_argument("--o", default="POSCARS", help="Output gathered POSCARS file")
    p_pack.add_argument("--traj", default=None, help="ASE trajectory file (default: pack POSCAR.*)")
    p_pack.add_argument("--range", default=None,
                        help="Frame range for --traj: index (5), list (0,2,4) or slice (0:10 / 0:10:2)")

    # ── denevo ──
    p_denevo = sub.add_parser("denevo", help=COMMANDS["denevo"][1])
    p_denevo.add_argument("--k", type=int, default=2,
                          help="EI 选择个数 (默认 2)")
    p_denevo.add_argument("--top", type=int, default=5,
                          help="每代 GP 高密度 top-N (默认 5)")
    p_denevo.add_argument("--last", type=int, default=None,
                          help="只绘制最后 N 代")
    p_denevo.add_argument("--gen-range", default=None, metavar="LO-HI",
                          help="只绘制 LO~HI 代 (含端点), 例如 20-25")
    p_denevo.add_argument("--out", default=None,
                          help="输出前缀 (默认 ./gp_all_gens)")
    p_denevo.add_argument("--gp-csv", default=None,
                          help="GP 数据源 gp.csv (默认自动定位 ../CalcFold1/gp.csv, "
                               "回退 density_predict.log)")

    # ── smd ──
    p_smd = sub.add_parser("smd", help=COMMANDS["smd"][1])
    # action flags (exactly one must be given)
    p_smd.add_argument("--opt", dest="opt", action="store_true", help="structure optimization")
    p_smd.add_argument("--fdf", dest="fdf", action="store_true", help="write SIESTA fdf input only")
    p_smd.add_argument("--traj", dest="traj", action="store_true", help="convert SIESTA run to ASE trajectory")
    p_smd.add_argument("--md", dest="md", action="store_true", help="NVT molecular dynamics")
    p_smd.add_argument("--npt", dest="npt", action="store_true", help="NPT molecular dynamics")
    p_smd.add_argument("--pm", dest="pm", action="store_true", help="press molecule")
    p_smd.add_argument("--mde", dest="mde", action="store_true", help="mean T/P from siesta.MDE")
    p_smd.add_argument("--xv", dest="xv", action="store_true", help="convert siesta.XV to geo.gen")
    p_smd.add_argument("--w", dest="w", action="store_true", help="write SIESTA input (VDW/DZP)")
    # common options
    p_smd.add_argument("--ncpu", "--n", dest="ncpu", type=int, default=20, help="CPU cores")
    p_smd.add_argument("--T", type=float, default=300, help="temperature (K)")
    p_smd.add_argument("--P", type=float, default=10.0, help="pressure (GPa, --npt)")
    p_smd.add_argument("--us", default="F", help="unrestricted spin: F/U")
    p_smd.add_argument("--tstep", type=int, default=50, help="MD number of steps")
    p_smd.add_argument("--dt", type=float, default=1.0, help="MD timestep (fs)")
    p_smd.add_argument("--gen", "--g", dest="gen", default="poscar.gen", help="geometry file")
    p_smd.add_argument("--i", type=int, default=-1, help="frame index")
    p_smd.add_argument("--l", type=int, default=0, help="opt: 1=variable cell, 0=fixed cell")
    p_smd.add_argument("--step", type=int, default=200, help="opt max steps")
    p_smd.add_argument("--kgrid", default=None, help="k-grid cutoff")
    p_smd.add_argument("--equil", type=int, default=250, help="equilibration frames skipped (--mde)")
    p_smd.add_argument("--xvfile", default="siesta.XV", help="XV file (--xv)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_func = COMMANDS[args.command][0]

    # Map args to function kwargs
    if args.command == "pred":
        cmd_func(t=args.t, g=args.g, gen=args.gen, f=args.f, den=args.den, ids=args.ids,x=args.x,
                 c=args.c,step=args.step, ncpu=args.ncpu, dat=args.data, tolerance=args.tolerance)
    elif args.command == "calc":
        cmd_func(t=args.t, den=args.den, ids=args.ids, step=args.step,
                 ncpu=args.ncpu, dat=args.data, tolerance=args.tolerance, gen=args.gen)
    elif args.command == "traj":
        cmd_func(fposcar=args.fposcar)
    elif args.command == "zmat":
        cmd_func(geo=args.geo, i=args.i)
    elif args.command == "fdf":
        cmd_func(gen=args.gen, xcf=args.xcf, i=args.i)
    elif args.command == "sample":
        cmd_func(ind=args.ind, t=args.t)
    elif args.command == "calcdata":
        cmd_func(traj=args.t, step=args.step,n=args.n,c=args.c)
    elif args.command == "gp":
        cmd_func(tolerance=args.t,step=args.step,n=args.n,b=args.b,u=args.u,f=args.f,
                 dft=args.dft,pop=args.pop,
                 dat=args.data,ref=args.ref,id_=args.id,k=args.k,mode=args.mode,optype=args.optype)
    elif args.command == "fixbroken":
        cmd_func(broken=args.b,dat=args.data,scale=args.s,ncpu=args.n)
    elif args.command == "add":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,i=args.i,ncpu=args.n)
    elif args.command == "addall":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,ncpu=args.n)
    elif args.command == "supercell":
        cmd_func(traj=args.t,gen=args.g,x=args.x,y=args.y,z=args.z)
    elif args.command == "update":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,inde=args.i,ncpu=args.n)
    elif args.command == "info":
        cmd_func(gen=args.gen, traj=args.traj, i=args.i,
                 symmetry=args.symmetry, symprec=args.symprec)
    elif args.command == "fingerprint":
        intra_map = None
        if args.intra_map:
            if args.intra_map.endswith(".npz"):
                tmp = np.load(args.intra_map)
                intra_map = tmp[list(tmp.keys())[0]] if len(tmp.files) == 1 else tmp["intra_map"]
            else:
                intra_map = np.load(args.intra_map)
        cmd_func(gen=args.g, traj=args.traj, i=args.i,
                 rmax=args.rmax, sigma=args.sigma, delta=args.delta,
                 dimension=args.dimension, output=args.output,
                 intra_map=intra_map, soap=args.soap,
                 soap_r_cut=args.soap_r_cut, soap_n_max=args.soap_n_max,
                 soap_l_max=args.soap_l_max)
    elif args.command == "lib":
        cmd_func(jsonfile=args.json, libfile=args.lib)
    elif args.command == "ffield":
        cmd_func(jsonfile=args.json, ffieldfile=args.ffield)
    elif args.command == "molinfo":
        cmd_func(gen=args.gen, i=args.i, jsonfile=args.jsonfile,
                 verbose=not args.quiet)
    elif args.command == "md2pdf":
        cmd_func(input=args.input)
    elif args.command == "pack":
        cmd_func(output=args.o, traj=args.traj, range_=args.range)
    elif args.command == "denevo":
        cmd_func(k=args.k, top=args.top, last=args.last,
                 gen_range=args.gen_range, out=args.out, gp_csv=args.gp_csv)
    elif args.command == "smd":
        cmd_func(opt=args.opt, fdf=args.fdf, traj=args.traj, md=args.md,
                 npt=args.npt, pm=args.pm, mde=args.mde, xv=args.xv, w=args.w,
                 ncpu=args.ncpu, T=args.T, P=args.P, us=args.us,
                 tstep=args.tstep, dt=args.dt, gen=args.gen, i=args.i,
                 l=args.l, step=args.step, kgrid=args.kgrid,
                 equil=args.equil, xvfile=args.xvfile)

