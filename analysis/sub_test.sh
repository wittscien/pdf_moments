#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=out.out
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=112G

while true
do
    sleep 100
done
