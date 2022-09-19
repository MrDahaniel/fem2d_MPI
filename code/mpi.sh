#!/bin/bash
#SBATCH --partition=normal
#SBATCH -o mpi.%j.out       #Nombre del archivo de salida
#SBATCH -J fem2d_MPI        #Nombre del trabajo
#SBATCH --nodes=1           #Numero de nodos para correr el trabajo
#SBATCH --ntasks=2         #Numero de procesos
#SBATCH --tasks-per-node=2   #Numero de trabajos por nodo

#Prepara el ambiente de trabajo
#export I_MPI_PMI_LIBRARY=/usr/local/slurm/lib/libpmi.so

module load devtools/mpi/openmpi/4.0.1
modele load devtools/gcc/9.2.0

# ulimit -l unlimited
# export OMPI_MCA_btl=^openib

mpic++ fem2d_poisson_mpi.cpp -o fem2d_mpi.out

#Ejecuta el programa paralelo
mpirun ./fem2d_mpi.out
