# neuroncompare/src/config_manager.py
import os
from typing import Dict, List, Any, Optional, Union
import h5py
import sys
import pandas as pd

class ConfigManager:
    """
    Central configuration manager for the neuroncompare pipeline.
    Handles loading and validation of configuration from input.txt.
    """
    
    # Required parameters that must be present in input.txt
    REQUIRED_PARAMS = [
        'model', 'peeling', 'user', 'data_dir', 'params', 
        'seed', 'stim_file'
    ]
    
    # Parameters that should be convertsed to integers
    INT_PARAMS = [
        'num_nodes', 'num_volts', 'timesteps', 'nSubZones', 
        'nPerSubZone', 'OFFSPRING_SIZE', 'MAX_NGEN', 'seed'
    ]
    
    # Parameters that should be converted to floats
    FLOAT_PARAMS = [
        'norm', 'dx'
    ]
    
    # Parameters that should be converted to booleans
    BOOL_PARAMS = [
        'passive', 'ingestCell', 'makeStims', 'makeParams', 'usePrevParams',
        'usePassiveParams', 'makeVolts', 'wait4volts', 'makeVoltsGPU',
        'makeScores', 'wait4scores', 'makeOpt', 'allenOpt', 'makeObj',
        'runGA', 'log_transform_params', 'gaGPU', 'sbatch', 'srun', 'shell'
    ]
    
    # Valid model options
    VALID_MODELS = ['allen', 'mainen', 'bbp', 'compare_bbp', 'M1_TTPC_NA_HH']
    
    # Valid peeling options
    VALID_PEELINGS = ['passive', 'potassium', 'sodium', 'calcium', 'full']
    
    def __init__(self, input_file_path: str = None):
        """
        Initialize the ConfigManager.
        
        Args:
            input_file_path: Path to the input.txt file. If None, uses ./input.txt
        """
        self.input_file_path = input_file_path or './input.txt'
        if not os.path.isfile(self.input_file_path):
            tmp = self.input_file_path 
            self.input_file_path = os.path.join(os.environ['NEURON_COMPARE_ROOT'] , 'input.txt')
            print(f"resorted to config at {self.input_file_path } instead of {tmp}")
        
        self.input_file_path = os.path.abspath(self.input_file_path)
        self.input_dir = os.path.dirname(self.input_file_path)

        self.config: Dict[str, Any] = {}
        self.load_config()
        self.validate_config()
        dt = None
    
    def load_config(self):
        """
        Load configuration from input.txt file.
        """
        try:
            with open(self.input_file_path, "r") as input_file:
                for line in input_file:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        self.config[key] = value
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.input_file_path}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")
        
        # Convert parameters to appropriate types
        self._convert_param_types()
        self._load_misc_params()

    def _convert_param_types(self):
        """
        Convert configuration parameters to their appropriate types.
        """
        # Convert integer parameters
        for param in self.INT_PARAMS:
            if param in self.config:
                try:
                    self.config[param] = int(self.config[param])
                except ValueError:
                    print(f"Warning: Could not convert {param} to integer: {self.config[param]}")
        
        # Convert float parameters
        for param in self.FLOAT_PARAMS:
            if param in self.config:
                try:
                    self.config[param] = float(self.config[param])
                except ValueError:
                    print(f"Warning: Could not convert {param} to float: {self.config[param]}")
        
        # Convert boolean parameters
        for param in self.BOOL_PARAMS:
            if param in self.config:
                self.config[param] = self.config[param].lower() == 'true'
        
        # Special case: params should be split into a list
        if 'params' in self.config:
            self.config['params_list'] = [int(p) for p in self.config['params'].split(',')]
            

    def validate_config(self):
        """
        Validate the loaded configuration.
        """
        # Check for required parameters
        missing_params = [param for param in self.REQUIRED_PARAMS if param not in self.config]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Validate model
        if self.config['model'] not in self.VALID_MODELS:
            raise ValueError(f"Invalid model: {self.config['model']}. Must be one of: {', '.join(self.VALID_MODELS)}")
        
        # Validate peeling
        if self.config['peeling'] not in self.VALID_PEELINGS:
            raise ValueError(f"Invalid peeling: {self.config['peeling']}. Must be one of: {', '.join(self.VALID_PEELINGS)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key to retrieve
            default: Default value to return if key not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary-like access.
        
        Args:
            key: The configuration key to retrieve
            
        Returns:
            The configuration value
            
        Raises:
            KeyError: If the key is not in the configuration
        """
        if key not in self.config:
            raise KeyError(f"Configuration key not found: {key}")
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: The configuration key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in self.config
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary containing all configuration values
        """
        return self.config.copy()
    
    def get_run_directory(self) -> str:
        """
        Get the run directory path based on configuration.
        
        Returns:
            Path to the run directory
        """
        model = self.config.get('model', '')
        peeling = self.config.get('peeling', '')
        runDate = self.config.get('runDate', '')
        custom = self.config.get('custom', '')
        
        if custom:
            return f"runs/{model}_{peeling}_{runDate}_{custom}"
        else:
            return f"runs/{model}_{peeling}_{runDate}"

    def _load_misc_params(self):
        """
        Loads miscellaneous parameters needed across different modules.
        Sets attributes on self for easy access throughout the application.
        """
        # Neural model paths and configuration
        if self.config['model'] == 'bbp':
            self.neuron_path = './cell_models/bbp/'
            self.run_file = './cell_models/bbp/run_model_cori.hoc'
        elif self.config['model'] == 'compare_bbp':
            self.neuron_path = './cell_models/compare_bbp/'
            self.run_file = './cell_models/compare_bbp/run_model_cori.hoc'
        elif self.config['model'] == 'allen':
            self.hoc_files = ["stdgui.hoc", "import3d.hoc", "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/cell.hoc"]
            self.compiled_mod_library = "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/x86_64/.libs/libnrnmech.so"
            self.args = {'manifest_file': '/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/manifest.json','axon_type': 'truncated'}
        elif self.config['model'] == 'M1_TTPC_NA_HH':
            self.neuron_path = 'cell_models/M1_TTPC_NA_HH'
            self.run_file = None
        
        
        # we are either in run folder or in root dir
        if 'runs' in self.input_dir:
            self.run_dir = f"./"
            self.params_file_path = '/params/params_' + self.config['model'] + '_' + self.config['peeling']+ '.hdf5'
            self.stims_file_path = os.path.join("stims", self.config['stim_file'] + '.hdf5')
        else:
            self.run_dir = os.path.join(self.input_dir , f"runs/{self.config['model']}_{self.config['peeling']}_{self.config['runDate']}_{self.config.get('custom', '')}")
            self.params_file_path = os.path.join(self.input_dir , f"runs/{self.config['model']}_{self.config['peeling']}_{self.config['runDate']}_{self.config['custom']}", 'params/params_' + self.config['model'] + '_' + self.config['peeling']+ '.hdf5')
            self.stims_file_path = os.path.join(self.input_dir , f"runs/{self.config['model']}_{self.config['peeling']}_{self.config['runDate']}_{self.config['custom']}",'stims',  self.config['stim_file'] + '.hdf5')
        
        self.volts_path = os.path.join(self.run_dir, 'volts','')
        self.output_path = os.path.join(self.run_dir, 'scores','')
        self.scores_path =os.path.join(self.run_dir, 'scores','')
        self.objectives_path = os.path.join(self.run_dir, 'genetic_alg', 'objectives','')
        self.objectives_file = os.path.join(self.objectives_path, 'multi_stim_without_sensitivity_' + self.config['model'] + '_' + self.config['peeling'] + "_" + self.config['runDate'] + '_stims.hdf5')
        if self.config['usePrevParams'] == True:
            self.params_csv = os.path.join(self.run_dir, 'params/params_' + self.config['model'] + '_' + self.config['peeling'] + '_prev.csv')
        else:
            self.params_csv = os.path.join(self.run_dir, 'params/params_' + self.config['model'] + '_' + self.config['peeling'] + '.csv')
    
        self.target_volts_path = os.path.join(self.run_dir, 'target_volts/target_volts_{}.hdf5'.format(self.config['modelNum']))
        self.target_volts_path_2 = os.path.join(self.run_dir,'target_volts/allen_data_target_volts_{}.hdf5'.format(self.config['modelNum']))
        # Parameter indices handling
        self.params_opt_ind = [int(p)-1 for p in self.config['params'].split(",")]
        
        # Simulation parameters
        self.ntimestep = int(self.config.get('timesteps', 10000))
        
        # Timestep handling
        if 'dt' in self.config and self.config['dt'] != 'null':
            self.dt = float(self.config['dt'])
        else:
            self.dt = None
        
        # Negative parameter indices (for e_pas)
        
        self.negative_param_inds = []
        for idx, param in enumerate(pd.read_csv(self.params_csv).to_dict(orient='records')):
            if 'e_pas' in param['Param name']:
                self.negative_param_inds.append(idx)
        
        # Additional stims
        if 'added_stims' in self.config:
            self.added_stims = [elem.encode('ASCII') for elem in self.config['added_stims'].split(',')]
        else:
            self.added_stims = []
        
        # Starting population configuration
        if 'data_dir' in self.config:
            self.starting_pop_hack = os.path.join(self.config['data_dir'], 'populations', 'starting_pop.pkl')
            
            # Log transform params handling
            if self.config.get('log_transform_params', False) and self.starting_pop_hack:
                self.starting_pop_hack = None
        else:
            self.starting_pop_hack = None
        
        # Passive configuration
        self.passive_scaler = 2
        if self.config.get('passive', False):
            self.PASSIVE_PERCENTAGE = 1
        else:
            self.PASSIVE_PERCENTAGE = 1  # was 2
        
        # Custom score functions
        self.custom_score_functions = [
            'chi_square_normal',
            'traj_score_1',
            'traj_score_2',
            'traj_score_3',
            'isi',
            'rev_dot_product',
            'KL_divergence'
        ]
        self.model = self.config['model']        
        self.log_transform_params =  self.config.get('log_transform_params', False)
        # set constants
        self.base_thresh = 50
        self.PASSIVE_PERCTENAGE = 1

# Global instance for singleton-like access
_config_instance = None

def get_config(input_file_path: str = None) -> ConfigManager:
    """
    Get a global instance of the ConfigManager.
    
    Args:
        input_file_path: Path to the input.txt file
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(input_file_path)
    return _config_instance
