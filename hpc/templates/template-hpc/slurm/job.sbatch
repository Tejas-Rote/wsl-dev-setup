#!/bin/bash
#SBATCH --job-name=hpc_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module purge
module load gcc

mkdir -p logs
./build/hpc_app
