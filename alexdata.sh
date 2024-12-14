#!/usr/bin/env zsh

#SBATCH -J final759
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH -t 0-00:05:00
#SBATCH -o final759.out -e final759.err

set -e

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

cd build
make

# GPU
for ((i=8; i<=28; i++)); do
    n=$((2**i))
    ./run "$n" "$n" | grep CUDA | sed 's/.*[^0-9.]//'
done

# # CPU
# for ((i=8; i<=28; i++)); do
#     n=$((2**i))
#     ./run "$n" "$n" | grep CPU | sed 's/.*[^0-9.]//'
# done