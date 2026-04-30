#!/bin/bash

#SBATCH --job-name=an_ENSEMBLE
#SBATCH --output=anouts/out_ENSEMBLE.out
#SBATCH --partition=vip1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=112G

PY main.py --ensemble=ENSEMBLE --tech=TECH --plotdata=1 --two=TWO --read2=R2 --read2_Nsrc_sink=R2s --read3=R3 --read3_conserved=R3c --read4=R4 > anlogs/an_ENSEMBLE.log 2>&1
