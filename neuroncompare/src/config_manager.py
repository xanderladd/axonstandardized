# neuroncompare/src/config_manager.py
import os
from typing import Dict, List, Any, Optional, Union

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
        self.config: Dict[str, Any] = {}
        self.load_config()
        self.validate_config()
    
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