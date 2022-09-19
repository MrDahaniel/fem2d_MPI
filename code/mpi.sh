#!/bin/bash
#SBATCH --partition=all     #Seleccione los nodos para el trabajo
                            # de todos el conjunto de nodos de cómputo del cluster
#SBATCH -o mpi.%j.out       #Nombre del archivo de salida
#SBATCH -J fem2d_MPI_job    #Nombre del trabajo
#SBATCH --nodes=1           #Numero de nodos para correr el trabajo
#SBATCH --ntasks=10         #Numero de procesos
#SBATCH --tasks-per-node=10   #Numero de trabajos por nodo

#Prepara el ambiente de trabajo
export I_MPI_PMI_LIBRARY=module load devtools/mpi/openmpi/4.0.1
ulimit -l unlimited
export OMPI_MCA_btl=^openib

#Ejecuta el programa paralelo
srun ./fem2d_poisson_mpi.cpp