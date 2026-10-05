"""SIESTA MD/optimization wrapper (integrated from I-ReaxFF tools/dft/smd.py).

Usage::

    uspexkit smd --opt --gen=poscar.gen --ncpu=8 --l=0
    uspexkit smd --fdf --gen=poscar.gen
    uspexkit smd --traj
    uspexkit smd --md --T=300 --tstep=50
"""
from os import system, getcwd
from os.path import exists
import numpy as np
from ase.io import read
from ase.data import chemical_symbols
from irff.dft.siesta import siesta_md, siesta_opt, write_siesta_in
from irff.molecule import press_mol
from irff.data.mdtodata import MDtoData


def _run_md(ncpu=20, T=300, us='F', tstep=50, dt=1.0, gen='poscar.gen', index=-1,
            P=None):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
        system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen, index=index)
    if P is None:
        print('\n-  running siesta md ...')
        siesta_md(A, ncpu=ncpu, T=T, dt=dt, tstep=tstep, us=us,
                  xcf='GGA', xca='PBE', basistype='split')
    else:
        print('\n-  running siesta npt ...')
        siesta_md(A, ncpu=ncpu, P=P, T=T, dt=dt, tstep=tstep, us=us,
                  opt='NoseParrinelloRahman')


def _run_opt(ncpu=8, T=2500, us='F', gen='poscar.gen', l=0, i=-1, step=200,
             kgrid=None):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
        system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen, index=i)
    print('\n-  running siesta opt ...')
    vc = 'true' if l else 'false'
    if kgrid is None:
        siesta_opt(A, ncpu=ncpu, us=us, VariableCell=vc, tstep=step,
                   xcf='GGA', xca='PBE', basistype='split')
    else:
        siesta_opt(A, ncpu=ncpu, us=us, VariableCell=vc, tstep=step,
                   KgridCutoff=kgrid,
                   xcf='GGA', xca='PBE', basistype='split')

    s = gen.split('.')[0]
    if s == 'POSCAR':
        s = gen.split('.')[-1]

    system('mv siesta.out siesta-{:s}.out'.format(s))
    system('mv siesta.traj id_{:s}.traj'.format(s))
    system('rm siesta.* ')
    system('rm fdf-* ')
    system('rm INPUT_TMP.*')


def _run_traj():
    cwd = getcwd()
    d = MDtoData(structure='siesta', dft='siesta', direc=cwd, batch=10000)
    d.get_traj()
    d.close()


def _run_pm(gen='siesta.traj', index=-1):
    """pressMol"""
    A = read(gen, index=index)
    print(A.get_cell())
    A = press_mol(A)
    A.write('poscar.gen')
    del A


def _run_mde(equil=250):
    t = []
    p = []
    with open('siesta.MDE', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i > equil:
                l = line.split()
                if len(l) > 0:
                    t.append(float(l[1]))
                    p.append(float(l[5]))

    print(' * Mean Temperature: %12.6f K' % np.mean(t))
    print(' * Mean Pressure: %12.6f GPa' % (np.mean(p) * 0.1))


def _run_xv(f='siesta.XV'):
    """XV to gen"""
    cell = []
    atoms = []
    element = {}
    natom = 0
    with open(f, 'r') as fv:
        for i, line in enumerate(fv.readlines()):
            if i <= 2:
                cell.append(line)
            elif i == 3:
                natom = int(line)
            else:
                l = line.split()
                atoms.append(line)
                element[int(l[0])] = int(l[1])

    lk = list(element.keys())
    lk.sort()

    with open('geo.gen', 'w') as fg:
        print(natom, 'S', file=fg)
        for k in lk:
            print(chemical_symbols[element[k]], end=' ', file=fg)
        print(' ', file=fg)
        for i, atom in enumerate(atoms):
            a = atom.split()
            print(i + 1, a[0], a[2], a[3], a[4], file=fg)
        print('0.0 0.0 0.0', file=fg)
        for c_ in cell:
            c = c_.split()
            print(c[0], c[1], c[2], file=fg)


def _run_write(gen='poscar.gen'):
    A = read(gen, index=-1)
    print('\n-  writing siesta input ...')
    write_siesta_in(A, coord='cart', md=False, opt='CG',
                    VariableCell='true', xcf='VDW', xca='DRSLL',
                    basistype='DZP')


def smd(opt=False, fdf=False, traj=False, md=False, npt=False, pm=False,
        mde=False, xv=False, w=False,
        ncpu=20, T=300, P=10.0, us='F', tstep=50, dt=1.0,
        gen='poscar.gen', i=-1, l=0, step=200, kgrid=None,
        equil=250, xvfile='siesta.XV'):
    """Dispatch SIESTA workflow by activation flag.

    Exactly one of --opt/--fdf/--traj/--md/--npt/--pm/--mde/--xv/--w must be set.
    """
    actions = [bool(v) for v in (opt, fdf, traj, md, npt, pm, mde, xv, w)]
    if sum(actions) == 0:
        raise SystemExit('smd: no action selected; use one of '
                         '--opt, --fdf, --traj, --md, --npt, --pm, --mde, --xv, --w')
    if sum(actions) > 1:
        raise SystemExit('smd: only one action flag may be active')

    if md:
        _run_md(ncpu=ncpu, T=T, us=us, tstep=tstep, dt=dt, gen=gen, index=i)
    elif npt:
        _run_md(ncpu=ncpu, T=T, us=us, tstep=tstep, dt=dt, gen=gen, index=i, P=P)
    elif opt:
        _run_opt(ncpu=ncpu, T=T, us=us, gen=gen, l=l, i=i, step=step,
                 kgrid=kgrid)
    elif fdf:
        _run_write(gen=gen)
    elif traj:
        _run_traj()
    elif pm:
        _run_pm(gen=gen, index=i)
    elif mde:
        _run_mde(equil=equil)
    elif xv:
        _run_xv(f=xvfile)
    elif w:
        _run_write(gen=gen)
