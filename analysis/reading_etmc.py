from pathlib import Path

import h5py as h5
import numpy as np
import tqdm

import funcs as tp


def reading(params, read2, read3):
    """Read source-averaged ETMC 2pt and traceless 3pt correlators.

        data2['pion']                 -> [configuration, time]
        data3['pion-PDF-n_2'][dt]     -> [configuration, flow, insertion time]

        metadata['t0']                 -> sqrt(t0) [fm]
        metadata['flow_dt']            -> delta(t)/sqrt(t0) [fm]
        metadata['flow_times']         -> t/sqrt(t0) [fm]

    The three dt values use the same configurations and source times, so the
    2pt and PDF data stay statistically paired.  Equal configuration numbers
    in the a and b sets are kept as different samples.
    """
    hadrons = params['key_3pt']
    moments = range(1, 7)
    dt_list = params['tsep_list']

    # kaon_s is an s-quark insertion in a kaon and uses data2['kaon'].
    hadron_files = {
        'pion': 'pion_uins',
        'kaon': 'kaon_uins',
        'kaon_s': 'kaon_sins',
    }

    # The HDF5 files already contain the symmetric traceless combinations.
    # No Lorentz-index or forward/backward-derivative combination is repeated.
    operators = {
        1: 'O4',
        2: 'O44',
        3: 'O444',
        4: 'O4444',
        5: 'O44444',
        6: 'O444444',
    }

    root = Path('../data/traceless_operators')
    ensemble = params['ensemble']
    corr_dir = Path('../%s/corr/%s' % (params['datadir']['mydata'], ensemble))
    corr_dir.mkdir(parents=True, exist_ok=True)
    file_2pt = corr_dir / ('data_2pt_%s.pckl' % ensemble)
    file_3pt = corr_dir / ('data_3pt_%s.pckl' % ensemble)
    file_metadata = corr_dir / ('metadata_%s.pckl' % ensemble)

    data2 = {}
    data3 = {}
    metadata = {}

    if read2 != 'direct':
        data2 = tp.fast_read(file_2pt)
    if read3 != 'direct':
        data3 = tp.fast_read(file_3pt)
        metadata = tp.fast_read(file_metadata)

    if read2 != 'direct' and read3 != 'direct':
        return data2, data3, metadata

    metadata = {
        'ensemble': ensemble,
        'confs': params['confs'],
    }

    for hadron in hadrons:
        file_label = hadron_files[hadron]
        data2_list = []
        data3_lists = {
            moment: {dt: [] for dt in dt_list}
            for moment in moments
        }
        for ensemble_name, conf in tqdm.tqdm(params['confs'], desc=f'Reading {ensemble} {hadron}'):
            ensemble_dir = root / ensemble_name
            twopt = []
            corr_sources = {
                moment: {dt: [] for dt in dt_list}
                for moment in moments
            }
            file = ensemble_dir / f'{conf:04d}_{file_label}.h5'
            with h5.File(file, 'r') as three_file:
                tau_list = [int(value) for value in three_file.attrs['tau_list']]

                # Use only source times shared by all three dt values.
                source_names = set(three_file[f'O4/dt{dt_list[0]}'].keys())
                for dt in dt_list[1:]:
                    source_names &= set(three_file[f'O4/dt{dt}'].keys())
                source_names = sorted(source_names)

                if read2 == 'direct' and hadron in params['key_2pt']:
                    conf_name = f'{conf:04d}'
                    for source_name in source_names:
                        source_time = int(source_name[2:])
                        twopt_file = root / 'twop' / ensemble_name / f'twop_stoch_st{source_time:02d}_chi.{conf_name}_{hadron}.h5'

                        with h5.File(twopt_file, 'r') as two_file:
                            source_group = next(iter(two_file.keys()))
                            raw = two_file[f'{source_group}/mesons/twop_meson'][:]
                            # Axes: [time, momentum, channel, real/imag].
                            # Momentum 0 and channel 0 select the pseudoscalar.
                            twopt.append(raw[:, 0, 0, 0] + 1j * raw[:, 0, 0, 1])

                if read3 == 'direct':
                    for dt in dt_list:
                        for moment in moments:
                            operator = operators[moment]
                            for source_name in source_names:
                                corr_sources[moment][dt].append(np.asarray([
                                    three_file[f'{operator}/dt{dt}/{source_name}/tau{flow_step:02d}'][:dt + 1]
                                    for flow_step in tau_list
                                ]))

                metadata['tau_list'] = tau_list
                # Keep the original HDF5 names, although t0 stores sqrt(t0) rather than t0.
                metadata['flow_dt'] = float(three_file.attrs['flow_dt'])
                metadata['t0'] = float(three_file.attrs['t0'])

            if read2 == 'direct' and hadron in params['key_2pt']:
                data2_list.append(np.mean(twopt, axis=0))

            if read3 == 'direct':
                for dt in dt_list:
                    for moment in moments:
                        data3_lists[moment][dt].append(
                            np.mean(corr_sources[moment][dt], axis=0)
                        )

        if read2 == 'direct' and hadron in params['key_2pt']:
            data2[hadron] = np.asarray(data2_list)

        if read3 == 'direct':
            for moment in moments:
                key = f'{hadron}-PDF-n_{moment}'
                data3[key] = {
                    dt: np.asarray(values)
                    for dt, values in data3_lists[moment].items()
                }

    # These values have units of fm and equal t/sqrt(t0), not physical flow times in fm^2.
    metadata['flow_times'] = np.asarray(metadata['tau_list']) * metadata['flow_dt']

    if read2 == 'direct':
        tp.write_data(file_2pt, data2)
    if read3 == 'direct':
        tp.write_data(file_3pt, data3)
    tp.write_data(file_metadata, metadata)

    return data2, data3, metadata
