# neuroncompare/src/path_manager.py
import os
from typing import Optional

from neuroncompare.src.config_manager import get_config

class PathManager:
    """
    Manages paths for the neuroncompare pipeline.
    Provides consistent path resolution across the pipeline.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the PathManager.
        
        Args:
            base_dir: Base directory for relative paths. If None, uses current directory.
        """
        self.config = get_config()
        self.base_dir = base_dir or os.getcwd()
        
        # Ensure data_dir is an absolute path
        self.data_dir = self._ensure_absolute_path(self.config['data_dir'])
    
    def _ensure_absolute_path(self, path: str) -> str:
        """
        Ensure path is absolute, converting if necessary.
        
        Args:
            path: Path to convert
            
        Returns:
            Absolute path
        """
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base_dir, path))
    
    def get_run_dir(self) -> str:
        """
        Get the run directory path.
        
        Returns:
            Absolute path to the run directory
        """
        run_dir = self.config.get_run_directory()
        return self._ensure_absolute_path(run_dir)
    
    def get_subdirectory(self, subdir: str, within_run_dir: bool = True) -> str:
        """
        Get a subdirectory path, optionally within the run directory.
        
        Args:
            subdir: Subdirectory name
            within_run_dir: If True, the subdirectory is within the run directory.
                            If False, it's relative to the base directory.
            
        Returns:
            Absolute path to the subdirectory
        """
        if within_run_dir:
            return os.path.join(self.get_run_dir(), subdir)
        return self._ensure_absolute_path(subdir)
    
    def ensure_directory_exists(self, path: str) -> str:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Absolute path to the directory
        """
        abs_path = self._ensure_absolute_path(path)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    def get_data_file(self, file_path: str) -> str:
        """
        Get the absolute path to a file in the data directory.
        
        Args:
            file_path: Relative path within the data directory
            
        Returns:
            Absolute path to the file
        """
        return os.path.join(self.data_dir, file_path)
    
    def get_run_file(self, file_path: str) -> str:
        """
        Get the absolute path to a file in the run directory.
        
        Args:
            file_path: Relative path within the run directory
            
        Returns:
            Absolute path to the file
        """
        return os.path.join(self.get_run_dir(), file_path)
    
    def get_params_path(self) -> str:
        """
        Get the absolute path to the params directory.
        
        Returns:
            Absolute path to the params directory
        """
        if self.config.get('usePrevParams', False):
            params_filename = f"params_{self.config['model']}_{self.config['peeling']}_prev.hdf5"
        else:
            params_filename = f"params_{self.config['model']}_{self.config['peeling']}.hdf5"
        
        return self.get_data_file(os.path.join('params', params_filename))
    
    def get_stims_path(self) -> str:
        """
        Get the absolute path to the stims file.
        
        Returns:
            Absolute path to the stims file
        """
        if self.config.get('passive', False):
            stims_filename = f"{self.config['stim_file']}_passive.hdf5"
        else:
            stims_filename = f"{self.config['stim_file']}.hdf5"
        
        return self.get_data_file(os.path.join('stims', stims_filename))
    
    def get_target_volts_path(self) -> str:
        """
        Get the absolute path to the target volts file.
        
        Returns:
            Absolute path to the target volts file
        """
        if self.config.get('passive', False):
            target_volts_filename = f"target_volts_{self.config['modelNum']}_passive.hdf5"
        else:
            target_volts_filename = f"target_volts_{self.config['modelNum']}.hdf5"
        
        return self.get_run_file(os.path.join('target_volts', target_volts_filename))
    
    def create_run_subdirs(self) -> None:
        """
        Create all standard subdirectories within the run directory.
        """
        subdirs = ['volts', 'scores', 'slurm', 'stims', 'target_volts', 'objectives', 'logs']
        
        for subdir in subdirs:
            self.ensure_directory_exists(self.get_subdirectory(subdir))
    
    def get_script_path(self, script_name: str) -> str:
        """
        Get the absolute path to a script.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Absolute path to the script
        """
        # Special cases for scripts that don't follow the prefix naming convention
        special_scripts = {
            'check_files': 'scripts/shell_scripts/check_files.sh',
            'genetic_algorithm': os.getcwd(),
            "compare_models": os.getcwd()
            # ^ these aren't really specical cases - fix if we add many more strages

        }
        
        # Default behavior for normal scripts
        if self.config.get('sbatch', False):
            prefix = 'sbatch'
            extension = '.slr'
            scripts_dir = 'scripts/slurm'
        elif self.config.get('srun', False):
            prefix = 'srun'
            extension = '.sh'
            scripts_dir = 'scripts/shell_scripts'
        else:
            prefix = 'shell'
            extension = '.sh'
            scripts_dir = 'scripts/shell_scripts'

        if script_name in special_scripts:
            script_path = special_scripts[script_name]
            if script_name == 'genetic_algorithm' or script_name == "compare_models":
                script_path = os.path.join(script_path,f"{prefix}_{script_name}{extension}")
            return self._ensure_absolute_path(script_path)
        else:
            script_path = os.path.join(scripts_dir, f"{prefix}_{script_name}{extension}")
            return self._ensure_absolute_path(script_path)

# Global instance for singleton-like access
_path_manager_instance = None

def get_path_manager(base_dir: Optional[str] = None) -> PathManager:
    """
    Get a global instance of the PathManager.
    
    Args:
        base_dir: Base directory for relative paths
        
    Returns:
        PathManager instance
    """
    global _path_manager_instance
    if _path_manager_instance is None:
        _path_manager_instance = PathManager(base_dir)
    return _path_manager_instance