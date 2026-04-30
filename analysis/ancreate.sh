#!/bin/bash

source ./env.sh
echo "Submitting analysis"

mkdir -p ansubs
mkdir -p anouts
mkdir -p anlogs

two=0
r2=direct
r2s=direct
r3=direct
r3c=direct
r4=direct

r2=fast
r2s=fast
r3=fast
r3c=fast
r4=fast

for tech in jackknife
do
    for ensemble in C24P29
    do
        echo "Submitting analysis for ${ensemble}"
        sed -e "s/PY/${py//\//\\/}/g; s/TECH/$tech/g; s/ENSEMBLE/$ensemble/g; s/TWO/$two/g; s/R2s/$r2s/g; s/R2/$r2/g; s/R3c/$r3c/g; s/R3/$r3/g; s/R4/$r4/g" sub_part.sh > ansubs/sub_${tech}_${ensemble}.sh
        sbatch ansubs/sub_${tech}_${ensemble}.sh
    done
done
