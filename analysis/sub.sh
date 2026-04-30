#!/bin/bash

#SBATCH --job-name=analysis
#SBATCH --output=analysis.out
#SBATCH --partition=C064M0256G
##SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
##SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1

mkdir -p logs

for tech in bootstrap
do
    for ens in 5
    do
        > logs/fit_${ens}.log
        for isospin in 1d2
        do
            #for ei in 000_A1+ 000_T1- 000_T2+ 000_E+ 001_A1 001_E2 001_B1 001_B2 011_A1 011_B1 011_B2 011_A2 111_A1 111_E2 002_A1 002_E2
            for ei in 000_A1+
            do
                ptot=${ei%_*}
                irrep=${ei#*_}
                echo ${tech} ${ens} ${isospin}

                if [ ${ei} = "000_A1+" ]
                then
                    one=1
                    readone="fast"
                else
                    one=0
                    readone="fast"
                fi

                ./main.py --ens=${ens} --isospin=${isospin} --ptot=${ptot} --irrep=${irrep} --tech=${tech} --plotdata=1 --one=${one} --dispersion=${one} --GEVP=1 --readone=${readone} --readtwo=fast >> logs/fit_${ens}.log 2>&1
            done
        done
    done
done
exit
cd ../luscher
mkdir -p logs

for spectype in FH FL
do
    for isospin in 1d2 3d2
    do
        for lmode in l=0 l=1 l=2 l=01
        do
            echo ${spectype} ${isospin} ${lmode}
            ./lumain.py --spectype=${spectype} --isospin=${isospin} --lmode=${lmode} --start=0 --stop=3 --step=0 --plain_spectrum=1 --plain_phaseshift=1 --core=1 --plot=1 > logs/lulog_${isospin}_${lmode}.log 2>&1
        done
    done
done
