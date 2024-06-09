#!/bin/bash
#SBATCH --job-name=dis_benchmarking
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=750G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=<my@mail.com>

module purge
module load anaconda3/2024.2
conda activate weirdest_manga

python benchmark.py