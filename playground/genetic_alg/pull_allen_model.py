from allensdk.api.queries.biophysical_api import BiophysicalApi
import cfg

bp = BiophysicalApi()
bp.cache_stimulus = True # change to False to not download the large stimulus NWB file
bp.cache_data(int(cfg.biophys_model_id), working_directory='allen_model')

