#!/usr/bin/env python
import subprocess
import argh
import argparse
import numpy as np
from os import getcwd,chdir,mkdir,system
from os.path import exists
import pickle
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from ase.io import read
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.dft.siesta import siesta_opt, write_siesta_in
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from irff.AtomDance import AtomDance


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None

def read_individuals(g=None):
    enthalpy  = []
    gene      = {}
    with open('Individuals') as f:
         for line in f.readlines():
             st = Stack([])
             for x in line:
                if x!=']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ !='[':
                        x_ = st.pop()
             line = ''.join(st.entry)
             l = line.split()
             
             if len(l)>=10:
                if l[0] != 'Gen':
                   g = int(l[0])
                   i = int(l[1])
                   e = float(l[3])
                   d = float(l[5])
                   if l[6]=='N/A':
                     f = 99999
                   else:
                     f = float(l[6])
                   if g in gene:  
                      gene[g].append((i,e,d,f))
                   else:
                      gene[g] = [(i,e,d,f)]
                   # enthalpy.append(float(l[3]))
         st.close()

    k = gene.keys()
    k_ = max(k) if g is None else g
    return gene[k_]

def get_gulp_energy(atoms,ncpu=8):
    atoms = opt(atoms=atoms,step=1000,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')   ## compute feature
    if ncpu==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(ncpu),shell=True)         ## compute feature
    e = get_reax_energy(fo='out')
    masses  = np.sum(atoms.get_masses())
    volume  = atoms.get_volume()
    density = masses/volume/0.602214129
    return atoms,e,density

def search_structure(feature,D,tolerance=0.01):
    # print(D.ndim)
    if D.ndim==2:
       res  = np.sum(np.square(D - feature),axis=1)
       ind  = np.where(res<tolerance)
       imin = np.argmin(res)
       res_ = res[imin]
    else:
       res  = np.sum(np.square(D - feature))
       ind  = [[0]] if res<tolerance else [[]]
       imin = 0
       res_ = res
    return ind,imin,res_

def load_gaussian_process(X,y,y_eng):
    if X.shape[1]==8:
       length_scale = [0.0526, 0.0525, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1]
    else:
       length_scale = [0.0526, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1]
    if not exists('gpr_density.pkl'):
       kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
              0.35**2 * Matern(length_scale=length_scale, nu=2.5) +       # 局部耦合
              WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-8, 1e-1))    )                                      # 噪声补偿
       gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
       gpr_density.fit(X,y)
       with open('gpr_density.pkl', 'wb') as f:
            pickle.dump(gpr_density, f)
    else:
       with open('gpr_density.pkl', 'rb') as f:
            gpr_density = pickle.load(f)
           
    if not exists('gpr_energy.pkl'):
       kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50)) +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                0.35**2 * Matern(length_scale=length_scale, nu=2.5) +       # 局部耦合
                WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-8, 1e-1))    )                                      # 噪声补偿
       gpr_energy = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
       gpr_energy.fit(X,y_eng)
       with open('gpr_energy.pkl', 'wb') as f:
            pickle.dump(gpr_energy, f)
    else:
       with open('gpr_energy.pkl', 'rb') as f:
            gpr_energy = pickle.load(f)

    with open('gpcsp.log','w') as fl:
        print(gpr_density.kernel_,file=fl)
        print(gpr_density.log_marginal_likelihood(),file=fl)
        print(gpr_energy.kernel_,file=fl)
        print(gpr_energy.log_marginal_likelihood(),file=fl)
    return gpr_energy,gpr_density    

def load_mlp(X,y):
    ''' 多层感知机机器学习模型 '''
    mlp = MLPRegressor((16, 8), max_iter=20000)
    mlp.fit(X,y)
    # score = mlp.score(train_inputs, train_labels)
    return mlp  

def load_rfr(X,y):
    ''' 随机森林机器学习模型 '''
    rfr   = RandomForestRegressor(random_state=37, n_estimators=300,
                                  min_weight_fraction_leaf=0.0,
                                  oob_score=True)
    # train
    rfr.fit(X, y)
    # score = rfr.score(X, y)
    # print(' * train set score: ',score)
    feature_importances = rfr.feature_importances_
    # print(' * feature importances: \n')
    # print(feature_importances)
    return rfr

def pred(t='Individuals.traj',g=None,f=1,den=1.88,ids=None,step=300,ncpu=8,dat='data',tolerance=0.001):
    ''' calculate the density of the crystal with DFT and High-Throughtput Screening 
        use with commmand like:
          ./uspexkit.py pred --i='253' --n=24 --dat=data11_44 --f=1
        where:
          i:   index
          den: density bigger than this while be computed
          n:   number of cpu
          dat: datafile directiory
    '''
    images = Trajectory(t)
    if not ids:
       ids = []
       res = read_individuals(g)
       for i,e,d,f in res:
           if d>den and f<0.0:
              ids.append(i)
    else:
        ids = [int(i) for i in ids.split()]

    root_dir   = getcwd()
    if not exists('density_predict.log'):
       with open('density_predict.log','w') as fd:
            print('# Crystal_id Density_mlp Density_rf Density_gp Energy std_den std_eng',file=fd)
         
    for s in ids:
        dir_list = root_dir.split('/')
        rootdir  = '/'.join(dir_list[:-1])
        data_dir = '{:s}/{:s}'.format(rootdir,dat)
        atoms = images[s-1]

        chdir(data_dir)
        atoms_mlp,e,density = get_gulp_energy(atoms,ncpu=ncpu)
        if f==1:
           feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
        else:
           feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
        
        assert exists('structures.traj'),'Error, datafile not found in data directory!'
        data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
        data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
        struc= Trajectory('structures.traj')
        
        D    = data[:,1:]         # 去掉索引
        D_   = data_[:,1:]
        ind,imin,res_ = search_structure(feature,D,tolerance=tolerance)
 
        X_raw  = data[:,1:]
        y      = data_[:,-1] #8
        y_eng  = data_[:,1]
        scaler = preprocessing.StandardScaler().fit(X_raw)
        X      = scaler.transform(X_raw)
        gpr_energy,gpr_density = load_gaussian_process(X,y,y_eng)  ## 
        rfr = load_rfr(X,y)
        mlp = load_mlp(X,y)
        
        X_ = scaler.transform(np.expand_dims(feature,axis=0))
        density, std_den_pred  = gpr_density.predict(X_, return_std=True)
        energy, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        energy  = energy[0]
        density = density[0]
        std_den_pred = std_den_pred[0]
        std_eng_pred = std_eng_pred[0]
        density_rf = rfr.predict(X_)
        density_mlp = mlp.predict(X_)
        density_rf = density_rf[0]
        density_mlp = density_mlp[0]
        if f==1:
           print('{:5d} rf: {:9.4f} mlp: {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} gp: {:7.4f} uncert: {:7.4f}'.format(s,
                 density_rf,density_mlp,feature[4],feature[5],feature[6],feature[7],density,std_den_pred))  
        else:
           print('{:5d} rf: {:9.4f} mlp: {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} gp: {:7.4f} uncert: {:7.4f}'.format(s,
                 density_rf,density_mlp,feature[3],feature[4],feature[5],feature[6],density,std_den_pred))
        chdir(root_dir)
        with open('density_predict.log','a') as fd:
             print('{:5d} {:9.6f} {:9.6f} {:9.6f} {:10.6f} {:9.6f} {:9.6f}'.format(s,
                    density_mlp,density_rf,density,energy,std_den_pred,std_eng_pred),file=fd)

def calc(t='Individuals.traj',den=1.88,ids=None,step=300,ncpu=8,dat='data',tolerance=0.01):
    ''' calculate the density of the crystal with DFT and High-Throughtput Screening '''
    images = Trajectory(t)
    if not ids:
       ids = []
       res = read_individuals()
       for i,e,d,f in res:
           if d>den and f<0.0:
              ids.append(i)
    else:
        ids = [int(i) for i in ids.split()]

    root_dir   = getcwd()
    if not exists('density.log'):
       with open('density.log','w') as fd:
            print('# Crystal_id Density Energy',file=fd)
         
    for s in ids:
        dir_list = root_dir.split('/')
        rootdir  = '/'.join(dir_list[:-1])
        data_dir = '{:s}/{:s}'.format(rootdir,dat)
        work_dir = root_dir+'/'+str(s)
        atoms = images[s-1]
        if exists(str(s)):
           continue  
        else:
           mkdir(str(s))

        chdir(data_dir)
        # print('change to data dir:',data_dir)
        atoms_mlp,e,density = get_gulp_energy(atoms,ncpu=ncpu)
        feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
        
        if exists('structures.traj'):
           data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
           data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
           struc= Trajectory('structures.traj')
           try:
              D    = data[:,1:]         # 去掉索引
              D_   = data_[:,1:]
           except IndexError:
              D    = data[1:]         # 去掉索引
              D_   = data_[1:]
           ind,imin,res_ = search_structure(feature,D,tolerance=tolerance)

        else:
           ind = [[]]  
           with open('feature_mlp.csv','w') as fd:
                print(', etot, ebond, eang, etor, evdw, ehb, ecoul, density',file=fd)
           with open('feature.csv','w') as fd_:
                print(', etot, ebond, eang, etor, evdw, ehb, ecoul, density',file=fd_)  
           masses  = np.sum(atoms.get_masses())
           volume  = atoms.get_volume()
           density = masses/volume/0.602214129
           res_    = 0.0
        # X_raw  = data[:,1:]
        # y      = data_[:,8]
        # y_eng  = data_[:,1]
        # scaler = preprocessing.StandardScaler().fit(X_raw)
        # X      = scaler.transform(X_raw)
        # gpr_energy,gpr_density = load_gaussian_process()
        
        chdir(work_dir)
        if len(ind[0])>0:
           atoms.write('POSCAR.{:d}'.format(s))
           struc[imin].write('POSCAR.{:d}_opt'.format(s))
           if D.ndim==2:
              energy  = D_[imin,0]
              density = D_[imin,7]
           else:
              energy  = D_[0]
              density = D_[7]
           print('{:5d} mt {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:7.4f} {:7.4f}'.format(s,
                 energy,feature[1],feature[2],feature[3],feature[4],feature[5],feature[6],density,res_))  
           traj  = TrajectoryWriter('id_{:d}.traj'.format(s),mode='w')
           traj.write(atoms=struc[imin])
           traj.close()
        else:
           subprocess.call('cp {:s}/Specific/*.psf ./'.format(rootdir),shell=True)
           img = siesta_opt(atoms,ncpu=ncpu,us='F',VariableCell='true',tstep=step,
                         xcf='GGA',xca='PBE',basistype='split')
                         # xcf='VDW',xca='DRSLL',basistype='split')
           subprocess.call('mv siesta.out siesta-{:d}.out'.format(s),shell=True)
           subprocess.call('mv siesta.MDE siesta-{:d}.MDE'.format(s),shell=True)
           subprocess.call('mv siesta.MD_CAR siesta-{:d}.MD_CAR'.format(s),shell=True)
           subprocess.call('mv siesta.traj id_{:d}.traj'.format(s),shell=True)
           subprocess.call('rm siesta.* ',shell=True)
           subprocess.call('rm *.xml ',shell=True)
           subprocess.call('rm INPUT_TMP.* ',shell=True)
           subprocess.call('rm fdf-* ',shell=True)
           img[0].write('POSCAR.{:d}'.format(s))
           atoms = img[-1]
           atoms.write('POSCAR.{:d}_opt'.format(s))
           masses = np.sum(atoms.get_masses())
           volume = atoms.get_volume()
           density = masses/volume/0.602214129
           energy  = atoms.get_potential_energy()
           
           print('{:5d} cl {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:9.4f} {:7.4f} {:7.4f}'.format(s,
                  energy,feature[1],feature[2],feature[3],feature[4],feature[5],feature[6],density,res_))  
            
           chdir(data_dir)
           with open('feature_mlp.csv','a') as fd:
                print(0,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',
                        feature[4],',',feature[5],',',feature[6],',',feature[7],file=fd) 
           with open('feature.csv','a') as fd:
                print(0,',',energy,',',feature[1],',',feature[2],',',
                feature[3],',',feature[4],',',feature[5],',',feature[6],',',density,file=fd)  
        
           atoms.calc = SinglePointCalculator(atoms,energy=energy)
           with TrajectoryWriter('structures_mlp.traj',mode='a') as traj:
                traj.write(atoms=atoms_mlp)
           with TrajectoryWriter('structures.traj',mode='a') as traj:
                traj.write(atoms=atoms)

        chdir(root_dir)
        with open('density.log','a') as fd:
             print('{:5d} {:10.6f} {:10.8f}'.format(s,density,energy),file=fd)

def traj(fposcar='gatheredPOSCARS'):
    fbp = open(fposcar,'r')
    lines = fbp.readlines()
    fbp.close()
    traj =  TrajectoryWriter('Individuals.traj',mode='w')
    k        = 0
    s        = 0 
    energies = []
    
    with open('Individuals') as f:
         for line in f.readlines():
             st = Stack([])
             for x in line:
                if x!=']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ !='[':
                        x_ = st.pop()
             line = ''.join(st.entry)
             l = line.split()
             
             if len(l)>=10:
                if l[0] != 'Gen':
                   energies.append(float(l[3]))
         st.close()

    for line in lines:
        if line.find('EA')>=0:
            if k>0:
                fpos.close()
                atoms = read('POSCAR')

                atoms.calc = SinglePointCalculator(atoms,energy=energies[s])
                traj.write(atoms=atoms)
                s += 1

            fpos = open('POSCAR','w')
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)
    
    fpos.close()
    
    atoms = read('POSCAR')
    atoms.calc = SinglePointCalculator(atoms,energy=energies[s])
    traj.write(atoms=atoms)
    traj.close()

def zmat(geo='POSCAR',i=-1):
    atoms  = read(geo,index=i)
    ad     = AtomDance(atoms=atoms,rcut={'H-O':2.7,'O-H':2.7})
    zmat   = ad.InitZmat
    
    ad.write_zmat(zmat,uspex=True)
    ad.close()

def fdf(gen='poscar.gen',xcf='gga',i=-1):
    A = read(gen,index=i)
    print('\n-  writing siesta input ...')
    if xcf=='gga':
       write_siesta_in(A,coord='cart', md=False, opt='CG',
                    VariableCell='true', 
                    xcf='GGA',xca='PBE',basistype='split' )
    elif xcf=='vdw':
       write_siesta_in(A,coord='cart', md=False, opt='CG',
                    VariableCell='true', xcf='VDW', xca='DRSLL',
                    basistype='split') # DZP
       # siesta_opt(A,ncpu=ncpu,us=us,VariableCell=vc,tstep=step,
       #            xcf='GGA',xca='PBE',basistype='split')
       #            xcf='VDW',xca='DRSLL',basistype='split')
    else:
       print('Not supported yet!')

def sample(ind='',t=None):
    ''' ./uspexkit.py sample --d=1.83
        ./uspexkit.py sample --i='2332 2338 2360'
        ./uspexkit.py sample --i='2332 2338 2360' --t=Individuals.traj
    '''
    # cdir    = getcwd()
    atoms   = None
    traj    = TrajectoryWriter('samples.traj',mode='w')

    if ind:
       ids = [int(i) for i in ind.split()]
       if t is not None:
          images = Trajectory(t)
          for i in ids:
              traj.write(atoms=images[i])
       else:
          for i in ids:
              atoms = read('{:d}/POSCAR.{:d}_opt'.format(i,i))
              atoms.calc = SinglePointCalculator(atoms,energy=0.0)
              traj.write(atoms=atoms) 
   
    traj.close()

if __name__=='__main__': 
   ''' A tool kit for USPEX crystal structure post process
       use commond like: 
          ./uspexkit.py calc --den=1.848 --s=300 --n=16
       to run this script.
       ---------------------------------------------
       calc: optimze the structures with DFT
       den:  density critia
       dat:  data dir
       计算密度和能量 (密度阈值 1.88，300 个结构，16 核)
           ./uspexkit.py calc --den=1.88 --ids=300 --ncpu=16
    
       转换 POSCAR 到轨迹
          ./uspexkit.py traj 
    
       生成 Z-matrix
          ./uspexkit.py zmat --geo=structure.vasp --i=0
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [calc,pred,traj,zmat,fdf,sample])
   argh.dispatch(parser)
   
