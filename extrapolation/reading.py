from pathlib import Path
import funcs as tp
import inputs


def reading(ensembles, tech, result_root, metadata_root):
    result_root = Path(result_root)
    metadata_root = Path(metadata_root)
    params = {}
    metadata = {}
    data = {}
    for ensemble in ensembles:
        class Args: pass
        Args.ensemble = ensemble
        Args.tech = tech
        Args.read2 = 'fast'
        Args.read3 = 'fast'
        params[ensemble] = inputs.cal_params(Args)
        metadata_file = metadata_root / ensemble / ('metadata_%s.pckl' % ensemble)
        result_file = result_root / ensemble / ('results_three_largest_tsep_%s.pckl' % tech)
        metadata[ensemble] = tp.fast_read(metadata_file)
        data[ensemble] = tp.fast_read(result_file)
    return params, metadata, data
