# neuroncompare/src/file_waiter.py
import os
import time
import h5py
import numpy as np
from typing import List, Optional, Callable, Union

from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.logger import get_logger

class FileWaiter:
    """
    Waits for files to be created or modified.
    Provides a robust waiting mechanism with timeout and progress reporting.
    """
    
    def __init__(self, timeout: int = 3600, sleep_interval: int = 5, 
                 progress_interval: int = 60):
        """
        Initialize the FileWaiter.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 1 hour)
            sleep_interval: Time to sleep between checks in seconds
            progress_interval: Time between progress reports in seconds
        """
        self.config = get_config()
        self.path_manager = get_path_manager()
        self.logger = get_logger()
        
        self.timeout = timeout
        self.sleep_interval = sleep_interval
        self.progress_interval = progress_interval
    
    def wait_for_file(self, file_path: str, condition: Optional[Callable] = None) -> bool:
        """
        Wait for a file to be created or modified.
        
        Args:
            file_path: Path to the file to wait for
            condition: Optional function that takes the file path as input and returns 
                       True if the file meets the required condition
            
        Returns:
            True if the file was created and meets the condition, False otherwise
        """
        start_time = time.time()
        last_progress_time = start_time
        
        self.logger.info(f"Waiting for file: {file_path}")
        
        while True:
            if os.path.exists(file_path):
                if condition is None or condition(file_path):
                    self.logger.info(f"File found: {file_path}")
                    return True
            
            # Check if timeout has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                self.logger.error(f"Timeout waiting for file: {file_path}")
                return False
            
            # Log progress periodically
            current_time = time.time()
            if current_time - last_progress_time > self.progress_interval:
                last_progress_time = current_time
                self.logger.debug(f"Still waiting for file: {file_path} "
                                 f"(elapsed: {int(elapsed_time)}s)")
            
            # Sleep before checking again
            time.sleep(self.sleep_interval)
    
    def wait_for_hdf5_datasets(self, file_path: str, dataset_names: List[str]) -> bool:
        """
        Wait for an HDF5 file with specific datasets to be created.
        
        Args:
            file_path: Path to the HDF5 file
            dataset_names: List of dataset names that should be in the file
            
        Returns:
            True if the file was created with all datasets, False otherwise
        """
        def check_datasets(path):
            try:
                with h5py.File(path, 'r') as f:
                    for name in dataset_names:
                        if name not in f:
                            return False
                return True
            except Exception:
                return False
        
        return self.wait_for_file(file_path, check_datasets)
    
    def wait_for_volts_files(self, stim_file: str, volt_dir: str) -> bool:
        """
        Wait for voltage files to be created based on the stim file.
        
        Args:
            stim_file: Path to the stim HDF5 file
            volt_dir: Directory where voltage files should be created
            
        Returns:
            True if all voltage files were created, False otherwise
        """
        try:
            # Get list of expected stim names from the stim file
            stim_names = []
            with h5py.File(stim_file, 'r') as f:
                for key in f.keys():
                    if not key.endswith('_dt') and key != 'stim_types':
                        stim_names.append(key)
            
            # Wait for each volt file
            total_files = len(stim_names)
            found_files = 0
            
            self.logger.info(f"Waiting for {total_files} voltage files in {volt_dir}")
            
            start_time = time.time()
            last_progress_time = start_time
            
            for stim_name in stim_names:
                volt_file = os.path.join(volt_dir, f"{stim_name}_volts.hdf5")
                
                while not os.path.exists(volt_file):
                    # Check if timeout has been reached
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.timeout:
                        self.logger.error(f"Timeout waiting for voltage files "
                                         f"({found_files}/{total_files} found)")
                        return False
                    
                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_progress_time > self.progress_interval:
                        last_progress_time = current_time
                        self.logger.debug(f"Still waiting for voltage files "
                                         f"({found_files}/{total_files} found, "
                                         f"elapsed: {int(elapsed_time)}s)")
                    
                    # Sleep before checking again
                    time.sleep(self.sleep_interval)
                
                found_files += 1
                
                # Log progress after finding a file
                if found_files % 10 == 0 or found_files == total_files:
                    self.logger.info(f"Found {found_files}/{total_files} voltage files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error waiting for voltage files: {str(e)}")
            return False
    
    def wait_for_scores_files(self, stim_file: str, score_dir: str) -> bool:
        """
        Wait for score files to be created based on the stim file.
        
        Args:
            stim_file: Path to the stim HDF5 file
            score_dir: Directory where score files should be created
            
        Returns:
            True if all score files were created, False otherwise
        """
        # Similar to wait_for_volts_files but for score files
        try:
            # Get list of expected stim names from the stim file
            stim_names = []
            with h5py.File(stim_file, 'r') as f:
                for key in f.keys():
                    if not key.endswith('_dt') and key != 'stim_types':
                        stim_names.append(key)
            
            # Wait for each score file
            total_files = len(stim_names)
            found_files = 0
            
            self.logger.info(f"Waiting for {total_files} score files in {score_dir}")
            
            start_time = time.time()
            last_progress_time = start_time
            
            for stim_name in stim_names:
                score_file = os.path.join(score_dir, f"{stim_name}_scores.hdf5")
                
                while not os.path.exists(score_file):
                    # Check if timeout has been reached
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.timeout:
                        self.logger.error(f"Timeout waiting for score files "
                                         f"({found_files}/{total_files} found)")
                        return False
                    
                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_progress_time > self.progress_interval:
                        last_progress_time = current_time
                        self.logger.debug(f"Still waiting for score files "
                                         f"({found_files}/{total_files} found, "
                                         f"elapsed: {int(elapsed_time)}s)")
                    
                    # Sleep before checking again
                    time.sleep(self.sleep_interval)
                
                found_files += 1
                
                # Log progress after finding a file
                if found_files % 10 == 0 or found_files == total_files:
                    self.logger.info(f"Found {found_files}/{total_files} score files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error waiting for score files: {str(e)}")
            return False

# Global instance for singleton-like access
_file_waiter_instance = None

def get_file_waiter(timeout: int = 3600, sleep_interval: int = 5, 
                   progress_interval: int = 60) -> FileWaiter:
    """
    Get a global instance of the FileWaiter.
    
    Args:
        timeout: Maximum time to wait in seconds
        sleep_interval: Time to sleep between checks in seconds
        progress_interval: Time between progress reports in seconds
        
    Returns:
        FileWaiter instance
    """
    global _file_waiter_instance
    if _file_waiter_instance is None:
        _file_waiter_instance = FileWaiter(timeout, sleep_interval, progress_interval)
    return _file_waiter_instance