#!/bin/bash
#SBATCH --ntasks=1024
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1Gb
#SBATCH --time=1:00:00
#SBATCH --array=0-71%1
#SBATCH --partition=express
#SBATCH --mail-type=END
#SBATCH --mail-user=weinbe58@gmail.com
#SBATCH --job-name=floquet_band
#SBATCH --output=outfiles/run_1/%a_%A.out


export OMP_NUM_THREADS=1
cd ~/floquet_spectrum

mpiexec -n 64 python band_model.py runfiles/run_1.in






