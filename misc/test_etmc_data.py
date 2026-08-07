import h5py

filename = '../data/traceless_operators/cC211a.20.48/1824_pion_uins.h5'

with h5py.File(filename, 'r') as f:
    print(list(f.keys()))       # 根目录下有哪些 operator
    print(dict(f.attrs))       # 根目录的全部 attributes

    print(f.attrs['tau_list'])
    print(f.attrs['flow_dt'])
    print(f.attrs['t0'])

    print(list(f['O44'].keys()))
    print(f['O44/dt32/st05/tau00'][:])