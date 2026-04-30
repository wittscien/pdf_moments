#!/bin/bash

# sn
if [ ${HOME} == "/public/home/yanhb" ]
then
    py="/public/home/yanhb/software/miniconda3/bin/python3"
    myuser="yanhb"
    partition="cpueicc,cpu6248R"
    #ntpn=48
    ntpn=1
    clim=180
# wmsk
elif [ ${HOME} == "/gpfs/share/home/2101110113" ]
then
    py="/gpfs/share/home/2101110113/software/miniconda3/bin/python3"
    myuser="2101110113"
    partition=C032M0128G
    #ntpn=32
    ntpn=1
    clim=2000
# wm2
elif [ ${HOME} == "/lustre/home/2101110113" ]
then
    py="/lustre/home/2101110113/software/miniconda3/bin/python3"
    myuser="2101110113"
    partition=C064M0256G
    #ntpn=64
    ntpn=1
    clim=1000
# zz
elif [ ${HOME} == "/public/home/liuchuan" ]
then
    py="/public/home/liuchuan/software/miniconda3/bin/python3"
    myuser="liuchuan"
    partition=vip1
    #ntpn=32
    ntpn=4
    clim=2990
fi
