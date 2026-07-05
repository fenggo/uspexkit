* uspexkit包使用说明 

0、下载安装
```
git clone https://github.com/fenggo/uspexkit.git
cd uspexkit
pip install .
```

1、traj
```
uspexkit traj
```
将结果文件转换为ASE轨迹.traj文件。 

2、calc
```
uspexkit calc --n=24   # 进行高通量筛选及DFT软件计算
```
3、zmat 将结构坐标文件转成USPEX内坐标MOL_*文件。
```
uspexkit zmat  --g=POSCAR
```

4、预测某个结构的能量
```
uspexkit pred --n=24 --i='214 215' --dat=data*
#或者
uspexkit pred  --g=POSCAR --n=24 --dat=data*
```

5、查看所有命令
```
uspexkit --help 
#or
uspexkit -h
```
