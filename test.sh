#!/usr/bin/env zsh

#SBATCH -J final759
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o final759.out -e final759.err

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

rm -rf build

mkdir build
cd build
cmake ..
make
make test
./run 131072