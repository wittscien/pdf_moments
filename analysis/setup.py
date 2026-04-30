from pathlib import Path
import inputs

def setup(params):
    # Setup
    Path("../%s/corr/%s/"%(params['datadir']['mydata'],params['ensemble'])).mkdir(parents=True, exist_ok=True)
    Path("../%s/spectra/%s/"%(params['datadir']['mydata'],params['ensemble'])).mkdir(parents=True, exist_ok=True)
    Path("../%s/spectra_full/%s/"%(params['datadir']['mydata'],params['ensemble'])).mkdir(parents=True, exist_ok=True)
