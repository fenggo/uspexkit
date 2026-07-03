"""Core commands: pred, calc, traj, zmat, fdf, sample, add, addall, gp."""
import os
import subprocess
import pickle
import numpy as np
from os import getcwd, chdir, mkdir
from os.path import exists

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
# from ase import build
from ase.io import read
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from uspexkit.utils import (read_individuals, search_structure,generate_hbond_lib,
                            write_input,run_gulp, # add_structure,
                            lammps_opt_mtp,
                            write_output,write_geometry)
# from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy ,opt
# from irff.dft.dftb import dftb_opt
from irff.dft.siesta import siesta_opt
from irff.molecule import Molecules,enlarge, SuperCell # moltoatoms


''' A work flow in combination with USPEX 
    High-Throughput Evolutionary Crystal Structure Prediction Method
'''

def supercell(gen=None,traj=None,x=1,y=1,z=1):
    if traj is None:
        A = read(gen)
        # build.make_supercell(A,[2,2,2])
        _,atoms = SuperCell(A,fac=1.0,supercell=[args.x,args.y,args.z])
        write(f'POSCAR.supercell_{x}_{y}_{z}',atoms)
    else:
        images = Trajectory(traj)
        A = images[-1]
        # build.make_supercell(A,[2,2,2])
        _,atoms = SuperCell(A,fac=1.0,supercell=[x,y,z])
        atoms.calc = SinglePointCalculator(atoms,energy=A.get_potential_energy()*x*y*z)
        his    = TrajectoryWriter(f'{traj.split(".")[0]}_{x}{y}{z}.traj',mode='w')
        his.write(atoms=atoms)
        his.close()

def addall(traj='structures.traj',step=1000,tolerance=0.005,ncpu=1):
    images = Trajectory(traj)
    for atoms_dft in images:
        add(atoms_dft,step=step,tolerance=tolerance,ncpu=ncpu)
 

def add(atoms_dft=None,traj='structures.traj',step=1000,tolerance=0.005,ncpu=1):
    if atoms_dft is None:
       atoms_dft = read(traj,-1)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    e     = get_feature(atoms,n=ncpu,lib='reaxff_nn')
    e_cho = get_hbond_feature(atoms,n=ncpu,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=ncpu,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=ncpu,elements='H core C core C core')

    volume   = atoms.get_volume()
    density_ = masses/volume/0.602214129
    
    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    d    = data[:,1:]         # 去掉索引
    i    = int(data[-1][0])   # 获取索引
    # print(cry)
    feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density_])
    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    
    if len(ind[0])>0:
       print(f'Structure already in database with index {ind[0]}!') 
       print(f'energy: {d[ind[0],0]}')
    else:
       with open('feature_mlp.csv','a') as fd:
            print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                     feature[5],',',feature[6],',',feature[7],',',
                     feature[8],',',feature[9],
                     file=fd) 
       with open('feature.csv','a') as fd:
            print(i,',',energy,',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                  feature[5],',',feature[6],',',feature[7],',',
                  feature[8],',',density,file=fd)  
    
       atoms.calc = SinglePointCalculator(atoms,energy=e[0])
       with TrajectoryWriter('structures_mlp.traj',mode='a') as traj:
            traj.write(atoms=atoms)
       with TrajectoryWriter('structures.traj',mode='a') as traj:
            traj.write(atoms=atoms_dft)


def get_feature(atoms,n=1,lib='reaxff_nn'):
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib=lib)
    if n==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
    e = get_reax_energy(fo='out')
    return e


def get_hbond_feature(atoms,n=1,elements='H core C core O core'):
    lib = generate_hbond_lib(elements)
    e = get_feature(atoms,n=n,lib=lib)
    return e


def gp(tolerance=0.005,step=1000,n=1,b=1.5,u=0.2,f=1,dat='data',resf='results1'):
    write_input(inp='inp-grad',keyword='grad conv qiterative verb')
    run_gulp(n=n,inp='inp-grad')
    e = get_reax_energy(fo='output')
    write_output(e=e[0])

    atoms  = read('gulp.cif')
    # atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])

    e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

    if f==1:
       # feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
       feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
    else:
       feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
 
    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    images = Trajectory('../{:s}/structures.traj'.format(dat))
    d      = data[:,1:]    # 去掉索引

    # Train a Gaussian Process 
    res    = np.sum(np.square(d - feature),axis=1)
    ind    = np.where(res<tolerance)
    imin   = np.argmin(res)

    ### prepare data 
    X_raw  = data[:,1:]
    y      = data_[:,-1]
    y_eng  = data_[:,1]

    d_scaler= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scaler= e_mean - np.mean(y_eng)

    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
 
    if not exists('gpr_density.pkl'):
        length_scale = [0.1 for _ in feature]
            
        kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                    0.35**2 * Matern(length_scale=length_scale, nu=2.5) +         # 局部耦合
                    WhiteKernel(noise_level=0.031,noise_level_bounds=(1e-8, 1e-1))    )                                   # 噪声补偿
        gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=1e-10,normalize_y=True)
        gpr_density.fit(X,y)
            
        kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                    0.35**2 * Matern(length_scale=length_scale, nu=2.5) +         # 局部耦合
                    WhiteKernel(noise_level=0.031,noise_level_bounds=(1e-8, 1e-1))    )     # 噪声补偿
        gpr_energy = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=1e-10,normalize_y=True)
        gpr_energy.fit(X,y_eng)
        # score  =  gaussian_process.score(X, y)
        with open('gpr_density.pkl', 'wb') as f:
             pickle.dump(gpr_density, f)
        with open('gpr_energy.pkl', 'wb') as f:
             pickle.dump(gpr_energy, f)
        with open('../{:s}/gpcsp.log'.format(resf),'w') as fl:
            print(gpr_density.kernel_,file=fl)
            print(gpr_density.log_marginal_likelihood(),file=fl)
            print(gpr_energy.kernel_,file=fl)
            print(gpr_energy.log_marginal_likelihood(),file=fl)
            # for hyperparameter in kernel.hyperparameters:
                  # print(kernel.kernel_,file=fl)
                  # print(hyperparameter,file=fl)
    else:
        with open('gpr_density.pkl', 'rb') as f:
             gpr_density = pickle.load(f)
        with open('gpr_energy.pkl', 'rb') as f:
             gpr_energy = pickle.load(f)
       
    if not exists('rfr_density.pkl'):
       rfr_density = RandomForestRegressor(random_state=37, n_estimators=300,
                                       min_weight_fraction_leaf=0.0,
                                       oob_score=True)
       rfr_density.fit(X, y)  # train
       feature_importances = rfr_density.feature_importances_
       with open('rfr_density.pkl', 'wb') as f:
            pickle.dump(rfr_density, f)
    else:
       with open('rfr_density.pkl', 'rb') as f:
            rfr_density = pickle.load(f)

    if not exists('../{:s}/gp.csv'.format(resf)):
        with open('../{:s}/gp.csv'.format(resf),'w') as fd:
             print(',   index,          residual,        density_min,         density_rf,   density_gp,'
                '          uncertainty,           energy_min,       eng_pred,        uncertainty_eng',file=fd)

    # X_ = np.concatenate((X,np.expand_dims(feature,axis=0)))  #X_train.extend(feature)
    X_ = scaler.transform(np.expand_dims(feature,axis=0))
    mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
    mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
    density_rf = rfr_density.predict(X_)
    # print('95% confidence interval: \n', 1.96 * std_prediction)
         
    with open('../{:s}/gp.csv'.format(resf),'a') as fd:
        # id_ = fd.tell()
        print(0,',',imin,',',res[imin],',',data_[imin][-1],',',
            density_rf[0],',',mean_prediction[0],',',
            1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
            file=fd)
    
    density_= mean_prediction[0] # data_[ind[0][im],-1]
    # if ((density_>np.max(y)*1.1 and (density_/density>1.5 or  density/density_>1.5)) or 
    #     (density_>np.max(y) and res[imin]>10) ):
    if res[imin]>10:
       if density_rf[0]/density>1.5 or  density/density_rf[0]>1.5:
          density_ = density*d_scaler
       else:
          density_ = density_rf[0]

    energy  = -density_ # mean_eng_pred[0]
    write_output(e=energy)
    write_geometry(atoms=atoms)
    

def calcdata(traj='structures.traj',n=8,c='nn',step=1000):
    ''' c: calculator, which mathine learning potential to be used '''
    images      = Trajectory(traj)
    traj_       = TrajectoryWriter('structures_mlp.traj',mode='w')

    with open('feature_mlp.csv','w') as fd:
        print(', etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density',file=fd)
    with open('feature.csv','w') as fd_:
        print(', etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn,ehb_chc, ecoul, density',file=fd_)

    for i,atoms in enumerate(images):
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density_ = masses/volume/0.602214129
        energy = atoms.get_potential_energy()
        if c=='nn':
           atoms = opt(atoms=atoms,step=step,l=1,t=0.000001,n=n, lib='reaxff_nn')
        elif c=='mtp':
           atoms = lammps_opt_mtp(atoms=atoms,step=step,n=n,lib='pot.almtp')
        else:
           raise RuntimeError("Caluclator not supported!") 
        e     = get_feature(atoms,n=n,lib='reaxff_nn')
        e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

        # atoms = read('gulp.cif')
        atoms.calc = SinglePointCalculator(atoms,energy=e[0])
        traj_.write(atoms=atoms)
        # e,ebond,elp,eover,eunder,eang,epen,tconj,etor,fconj,evdw,ehb,ecl,esl
        
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        #  print(e)
        print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} '
            'ehb: {:8.4f}  {:8.4f} {:8.4f} {:8.4f} ' 
            'ecoul: {:8.4f} density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],
                                        e[11],e_cho[11],e_chn[11],e_chc[11],
                                        e[12],density))
        with open('feature_mlp.csv','a') as fd:
            print(i,',',e[0],',',e[1],',',e[5],',',e[8],',',e[10],',',e_cho[11],',',e_chn[11],',',e_chc[11],',',
                e[12],',',density,file=fd) 
        with open('feature.csv','a') as fd_:
            print(i,',',energy,',',e[1],',',e[5],',',e[8],',',e[10],',',e_cho[11],',',e_chn[11],',',e_chc[11],',',
                e[12],',',density_,file=fd_) 
            
    traj_.close()

# ──────────────────────────────────────────────
#  fix broken molecule
# ──────────────────────────────────────────────
def fixbroken(broken=1.5,dat='data',scale=1.2,ncpu=1):
    write_input(inp='inp-grad',keyword='grad conv qiterative verb')
    run_gulp(n=ncpu,inp='inp-grad')
    e = get_reax_energy(fo='output')
    write_output(e=e[0])

    atoms  = read('gulp.cif')
    # atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])
    # feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])

    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    images = Trajectory('../{:s}/structures.traj'.format(dat))
    d      = data[:,1:]    # 去掉索引

    ### prepare data 
    X_raw  = data[:,1:]
    y      = data_[:,-1]
    y_eng  = data_[:,1]
    d_scale= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scale= e_mean - np.mean(y_eng)

    if e_mean-e[0]>broken:
       if exists("molecule.pkl"):
          with open("molecule.pkl", "rb") as f:
               m_ = pickle.load(f)
          for m in m_:
              for i,na in enumerate(m.mol_index):
                  m.mol_x[i] = atoms.positions[na]

          for m in m_:
              m.center       = np.sum(m.mol_x,axis=0)/m.natom
        
          nmol    = len(m_)
          cell    = atoms.get_cell()
          irun = 0
          fac  = 1.0
          while e_mean-e[0]>broken and irun < 15:
                fac = fac*scale
                _,atoms = enlarge(m_,cell=cell,fac=fac,supercell=[1,1,1])
                atoms,e,density = get_gulp_energy(atoms, ncpu=ncpu,o=False)
                irun += 1
    else:
       if not exists("molecule.pkl"):
          m_  = Molecules(atoms,rcut={"H-H":1.0,"H-O":1.02,"O-O":1.4,"H-N":1.22,"H-C":1.35,
                                "others": 1.75},check=True)
          with open("molecule.pkl", "wb") as f:
               pickle.dump(m_, f)
    
    write_output(e=e[0])
    write_geometry(atoms=atoms)

# ──────────────────────────────────────────────
#  GULP energy helper
# ──────────────────────────────────────────────

def get_gulp_energy(atoms, ncpu=8,o=True):
    if o:
       atoms_opt = opt(atoms=atoms, step=1000, l=1, t=0.000001, n=ncpu, lib="reaxff_nn")
    else:
       atoms_opt = atoms
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

# def load_mlp(X, y):
#     mlp = MLPRegressor((16, 8), max_iter=20000)
#     mlp.fit(X, y)
#     return mlp

def load_rfr(X, y):
    rfr = RandomForestRegressor(random_state=37, n_estimators=300,
                                min_weight_fraction_leaf=0.0, oob_score=True)
    rfr.fit(X, y)
    return rfr

# ──────────────────────────────────────────────
#  pred — 高斯过程预测
# ──────────────────────────────────────────────

def pred(t="Individuals.traj", g=None, f=1, den=1.88, ids=None,
         c='nn',step=300, ncpu=8, dat="data", tolerance=0.001):
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
    if g is not None:
       atoms  = read(g)
       images = [atoms]
    else:
       images = Trajectory(t)

    if g is None:
       ids_list = [1]
    elif not ids:
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

    masses = np.sum(images[0].get_masses())
    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        atoms = images[s - 1]

        chdir(data_dir)
        if c=='nn':
           # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
           atoms_mlp = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
        elif c=='mtp':
           # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
           atoms_mlp = lammps_opt_mtp(atoms=atoms,step=step,n=ncpu,lib='pot.almtp')
        else:
           raise RuntimeError("Caluclator not supported!") 

        volume  = atoms_mlp.get_volume()
        density = masses / volume / 0.602214129
        e       = get_feature(atoms,n=ncpu,lib='reaxff_nn')
        e_cho   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
        e_chn   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
        e_chc   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')
        
        # if f == 1:
        feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
        # else:
        #    feature = np.array([e[0],e[1],e[5], e[8], e[10], e[11], e[12], density])

        assert exists("structures.traj"), "Error, datafile not found in data directory!"
        data  = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
        data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
        struc = Trajectory('structures.traj')

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
        # mlp = load_mlp(X, y)

        X_ = scaler.transform(np.expand_dims(feature, axis=0))
        density_pred, std_den_pred = gpr_density.predict(X_, return_std=True)
        energy_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        energy_pred  = energy_pred[0]
        density_pred = density_pred[0]
        std_den_pred = std_den_pred[0]
        std_eng_pred = std_eng_pred[0]
        density_rf   = rfr.predict(X_)[0]
        # density_mlp  = mlp.predict(X_)[0]

        # if f == 1:
        print(f"{s:5d} res: {res_[imin]}"
              f"rf: {density_rf:7.4f} "
              f"gp(den): {density_pred:7.4f} uncert: {std_den_pred:7.4f} "
              f"gp(eng): {energy_pred:7.4f} uncert: {std_eng_pred:7.4f}" )
        # else:
        #     print(f"{s:5d} rf: {density_rf:9.4f} "
        #           f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} {feature[6]:9.4f} "
        #           f"gp: {density_pred:7.4f} uncert: {std_den_pred:7.4f}")

        chdir(root_dir)
        with open("density_predict.log", "a") as fd:
             print(f"{s:5d} {res_[imin]} "
                   f"{density_rf:7.4f} "
                   f"{density_pred:7.4f} {std_den_pred:7.4f}"
                   f"{energy_pred:7.4f} {std_eng_pred:7.4f} " ,file=fd)

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
        e_cho = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')

        feature = np.array([e[0], e[1], e[5], e[8], e[10], e_cho[11], e_chn[11], e_chc[11],e[12], density])

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
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd)
            with open("feature.csv", "w") as fd_:
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd_)
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
                      f"{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{feature[9]}", file=fd)
            with open("feature.csv", "a") as fd:
                print(f"0,{energy},{feature[1]},{feature[2]},"
                      f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{density}", file=fd)

            atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
            with TrajectoryWriter("structures_mlp.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_mlp)
            with TrajectoryWriter("structures.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_opt)

        chdir(root_dir)
        with open("density.log", "a") as fd:
            print(f"{s:5d} {density:10.6f} {energy:10.8f}", file=fd)

# ──────────────────────────────────────────────
#  traj — POSCAR to trajectory
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

