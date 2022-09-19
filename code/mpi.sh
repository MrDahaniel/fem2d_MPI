#!/bin/bash
#SBATCH --partition=normal
#SBATCH -o mpi.%j.out       #Nombre del archivo de salida
#SBATCH -J fem2d_MPI        #Nombre del trabajo
#SBATCH --nodes=1           #Numero de nodos para correr el trabajo
#SBATCH --ntasks=2         #Numero de procesos
#SBATCH --tasks-per-node=2   #Numero de trabajos por nodo


module load devtools/mpi/openmpi/4.0.1
module load devtools/gcc/9.2.0

mpic++ fem2d_poisson_mpi.cpp -o fem2d_mpi.out

mpirun ./fem2d_mpi.out