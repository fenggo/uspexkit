"""Command-line interface for uspexkit."""

import argparse
import sys
from uspexkit.core import pred, calc, traj, zmat, fdf, sample,calcdata,gp,fixbroken,add,addall

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
    p_pred.add_argument("--g", type=int, default=None, help="Generation number")
    p_pred.add_argument("--f", type=int, default=1, help="Feature flag (1=8D)")
    p_pred.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_pred.add_argument("--ids", default=None, help="Crystal indices (space-separated)")
    p_pred.add_argument("--step", type=int, default=300, help="Optimization steps")
    p_pred.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_pred.add_argument("--dat", default="data", help="Data directory name")
    p_pred.add_argument("--tolerance", type=float, default=0.001, help="Structure matching tolerance")

    # ── calc ──
    p_calc = sub.add_parser("calc", help=COMMANDS["calc"][1])
    p_calc.add_argument("--t", default="Individuals.traj", help="Trajectory file")
    p_calc.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_calc.add_argument("--ids", default=None, help="Crystal indices")
    p_calc.add_argument("--step", type=int, default=300, help="MD steps")
    p_calc.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_calc.add_argument("--dat", default="data", help="Data directory name")
    p_calc.add_argument("--tolerance", type=float, default=0.01, help="Structure matching tolerance")

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
    p_calcdata.add_argument("--n", default=1, help="number cpu tobe used")
    p_calcdata.add_argument("--t", default='structures.traj', help="Trajectory file")
    p_calcdata.add_argument("--step", default=1000, help="number of step to used to optimize by MLP")

   # ── gp ──  
    p_gp = sub.add_parser("gp", help=COMMANDS["gp"][1])
    p_gp.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_gp.add_argument("--t", default=0.005, help="structure match tolerance")
    p_gp.add_argument("--step", default=1000, help="number of step to used to optimize by MLP")
    p_gp.add_argument("--b", default=1.5, help="energy devate the mean tolerance that the structure is broken")
    p_gp.add_argument("--u", default=0.2, help="uncertainty of Gaussian Process")
    p_gp.add_argument("--f", default=1, help="which feature factor to be used")
    p_gp.add_argument("--data", default='data', help="which data to be used")
    p_gp.add_argument("--resf", default='results1', help="results file directory")

 # ── fixbroken ── 
    p_fixbroken = sub.add_parser("fixbroken", help=COMMANDS["fixbroken"][1])
    p_fixbroken.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_fixbroken.add_argument("--data", default='data', help="which data to be used")
    p_fixbroken.add_argument("--s", type=float,default=1.2, help="scale factor")
    p_fixbroken.add_argument("--b", type=float,default=1.5, help="energy devate the mean tolerance that the structure is broken")

 # ── add ── 
    p_add = sub.add_parser("add", help=COMMANDS["add"][1])
    p_add.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_add.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_add.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_add.add_argument("--t", type=str,default='structures.traj', help="trajector file name")

 # ── addall ── 
    p_addall = sub.add_parser("addall", help=COMMANDS["addall"][1])
    p_addall.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_addall.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_addall.add_argument("--t", type=str,default='structures.traj', help="trajector file name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_func = COMMANDS[args.command][0]

    # Map args to function kwargs
    if args.command == "pred":
        cmd_func(t=args.t, g=args.g, f=args.f, den=args.den, ids=args.ids,
                 step=args.step, ncpu=args.ncpu, dat=args.dat, tolerance=args.tolerance)
    elif args.command == "calc":
        cmd_func(t=args.t, den=args.den, ids=args.ids, step=args.step,
                 ncpu=args.ncpu, dat=args.dat, tolerance=args.tolerance)
    elif args.command == "traj":
        cmd_func(fposcar=args.fposcar)
    elif args.command == "zmat":
        cmd_func(geo=args.geo, i=args.i)
    elif args.command == "fdf":
        cmd_func(gen=args.gen, xcf=args.xcf, i=args.i)
    elif args.command == "sample":
        cmd_func(ind=args.ind, t=args.t)
    elif args.command == "calcdata":
        cmd_func(traj=args.t, step=args.step,n=args.n)
    elif args.command == "gp":
        cmd_func(tolerance=args.t,step=args.step,n=args.n,b=args.b,u=args.u,f=args.f,dat=args.data,resf=args.resf)
    elif args.command == "fixbroken":
        cmd_func(broken=args.b,dat=args.data,scale=args.s,ncpu=args.n)
    elif args.command == "add":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,ncpu=args.n)
    elif args.command == "addall":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,ncpu=args.n)