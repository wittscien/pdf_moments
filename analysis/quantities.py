import sys
from pathlib import Path
import numpy as np
import gvar as gv
import funcs as tp
import inputs

def quantities(params, params2, relen, result, Gresult, weighto):
    weight = 0 if weighto == 'no' else weighto

    relen = params['relen']
    tech = params['tech']
    # aaa
    L = params['L']
    mpi_matrix = np.zeros(relen)
    for i in range(relen):
        mpi_matrix[i] = result['pion_0'][i][1]
    mpi_mean = tp.cal_mean(mpi_matrix)
    mpi_err = tp.cal_err(mpi_matrix,tech)
    print('mpi = '+repr(gv.gvar(mpi_mean * params['hca'],mpi_err * params['hca'])), 'MeV')
    #print(2 * np.sqrt(mpi**2+(2*np.pi/L)**2) + np.sqrt(mpi**2+2*(2*np.pi/L)**2))

    # mpi L
    mpiL_matrix = np.zeros(relen)
    for ls in range(relen):
        mpiL_matrix[ls] = result['pion_0'][ls][1] * params['L']
    mpiL_mean = tp.cal_mean(mpiL_matrix)
    mpiL_err = tp.cal_err(mpiL_matrix,tech)
    print('mpiL = '+repr(gv.gvar(mpiL_mean,mpiL_err)))

    # In mpi unit
    print('In mpi unit:')
    for k in Gresult.keys():
        Gmass_mpi_matrix = np.zeros(relen)
        for ls in range(relen):
            Gmass_mpi_matrix[ls] = (Gresult[k][ls][0] + weight) / result['pion_0'][ls][1]
        Gmass_mpi_mean = tp.cal_mean(Gmass_mpi_matrix)
        Gmass_mpi_err = tp.cal_err(Gmass_mpi_matrix,tech)
        print(k, 'm = '+repr(gv.gvar(Gmass_mpi_mean,Gmass_mpi_err)))

    # Writing to txt
    with open('../%s/spectra/%s/results_one_%s_%s.txt'%(params['datadir']['mydata'],params['ensname'],params['ensname'],params['tech']), 'w') as txt_file:
        for ls in range(relen):
            txt_file.write("%d %f %f \n" % (ls, mpi_matrix[ls] * params['hca'], mpiL_matrix[ls]))
