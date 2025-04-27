# neuroncompare/src/execution_manager.py
import os
import subprocess
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.logger import get_logger

class ExecutionManager:
    """
    Manages execution of commands and scripts for the neuroncompare pipeline.
    Handles different execution modes (local, sbatch, srun) based on configuration.
    """
    
    # Scripts with shell/batch versions
    SHELL_SCRIPTS = {
        'volts': True,
        'scores': True,
        'opt': True,
        'runGA': True,
        'check_files': True,
        'hack_interactive': True, 
        'volts_scores': True,
    }
    
    # Scripts that are always Python modules
    PYTHON_MODULES = {
        'make_params': 'neuroncompare.src.make_params',
        'cell_ingest': 'neuroncompare.src.cell_ingest',
        'analyze_p_multistims': 'neuroncompare.src.analyze_p_multistims',
        'modifySandboxArray': 'neuroncompare.src.modifySandboxArray',
    }
    
    # Scripts that require interactive input
    INTERACTIVE_SCRIPTS = {
        'make_params': True,
        'volts': True,
        'scores': True
    }
    
    def __init__(self):
        """Initialize the ExecutionManager."""
        self.config = get_config()
        self.path_manager = get_path_manager()
        self.logger = get_logger()
        
        # Determine execution mode
        self.sbatch = self.config.get('sbatch', False)
        self.srun = self.config.get('srun', False)
        self.shell = self.config.get('shell', True)
    
    def execute_command(self, command: str, wait: bool = True, 
                       cwd: Optional[str] = None, 
                       interactive: bool = False) -> Tuple[int, str, str]:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            wait: If True, wait for command to complete
            cwd: Working directory for command execution
            interactive: If True, allow command to interact with user
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        self.logger.log_command(command)
        
        try:
            if interactive:
                print(f"HIIII {command} ")
                # For interactive scripts, don't capture stdout/stderr
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd
                )
                
                if wait:
                    process.wait()
                    return_code = process.returncode
                    
                    if return_code != 0:
                        self.logger.error(f"Interactive command failed with return code {return_code}", 
                                          command=command)
                    else:
                        self.logger.debug(f"Interactive command completed successfully", command=command)
                    
                    return return_code, "", ""
                
                return 0, "", ""
            else:
                # For non-interactive scripts, capture stdout/stderr
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    cwd=cwd
                )
                
                if wait:
                    stdout, stderr = process.communicate()
                    return_code = process.returncode
                    
                    if return_code != 0:
                        self.logger.error(f"Command failed with return code {return_code}", 
                                          command=command, stderr=stderr)
                    else:
                        self.logger.debug(f"Command completed successfully", command=command)
                    
                    return return_code, stdout, stderr
                
                return 0, "", ""  # Return default values if not waiting
            
        except Exception as e:
            self.logger.error(f"Failed to execute command: {e}", command=command)
            return 1, "", str(e)
    
    def execute_python_module(self, module_path: str, args: List[str] = None, 
                         parallel: bool = False, wait: bool = True,
                         cwd: Optional[str] = None,
                         interactive: bool = False) -> Tuple[int, str, str]:
        """
        Execute a Python module.
        
        Args:
            module_path: Path to the Python module (dot notation)
            args: List of arguments to pass to the module (preserving order)
            parallel: If True, execute using srun if configured
            wait: If True, wait for command to complete
            cwd: Working directory for command execution
            interactive: If True, allow module to interact with user
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        args_str = " ".join(args) if args else ""
        
        if parallel and self.srun:
            command = f"srun -n {self.config.get('num_nodes', 1)} python -m {module_path} {args_str}"
        else:
            command = f"python -m {module_path} {args_str}"
        
        return self.execute_command(command, wait, cwd, interactive)

    
    def execute_stage(self, stage_name: str, args: List[str] = None, 
                 wait: bool = True, cwd: Optional[str] = None) -> Tuple[int, str, str]:
        """
        Execute a pipeline stage by name, determining the appropriate execution method.
        
        Args:
            stage_name: Name of the pipeline stage
            args: List of arguments to pass to the stage
            wait: If True, wait for stage to complete
            cwd: Working directory for command execution
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        args_list = args if args else []
        interactive = stage_name in self.INTERACTIVE_SCRIPTS
        
        # Log the stage and arguments
        stage_logger = self.logger.create_stage_logger(stage_name)
        args_str = " ".join(args_list) if args_list else ""
        stage_logger.info(f"Executing stage: {stage_name}" + (f" with args: {args_str}" if args_str else ""))
        
        # If it's a Python module that's always executed directly
        if stage_name in self.PYTHON_MODULES:
            module_path = self.PYTHON_MODULES[stage_name]
            
            # Pass args directly as a list to preserve order
            return self.execute_python_module(
                module_path, 
                args_list, 
                wait=wait, 
                cwd=cwd, 
                interactive=interactive
            )
    
        # If it's a script with shell/batch versions
        elif stage_name in self.SHELL_SCRIPTS:
            script_path = self.path_manager.get_script_path(stage_name)
            
            if self.sbatch:
                command = f"sbatch {script_path} {args_str}"
            else:
                command = f"sh {script_path} {args_str}"
                
            return self.execute_command(command, wait, cwd, interactive)
        
        # For other cases, try to determine the best approach
        else:
            self.logger.warning(f"Unknown stage: {stage_name}, attempting direct execution")
            
            # Try as Python module first
            module_path = f"neuroncompare.src.{stage_name}"
            dict_args = {}
            if args:
                for arg in args:
                    if arg.startswith('--'):
                        arg = arg[2:]  # Remove leading --
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            dict_args[key] = value
                        else:
                            # Handle arguments without values
                            dict_args[arg] = ""
                    else:
                        # Handle positional arguments
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            dict_args[key] = value
                        else:
                            # Add as flag
                            dict_args[arg] = ""
            
            return self.execute_python_module(
                module_path, 
                dict_args, 
                wait=wait, 
                cwd=cwd, 
                interactive=interactive
            )
    
    def execute_pipeline_stage(self, stage_name: str, args: List[str] = None, 
                              wait: bool = True) -> bool:
        """
        Execute a pipeline stage with logging.
        
        Args:
            stage_name: Name of the pipeline stage
            args: List of arguments to pass to the stage
            wait: If True, wait for stage to complete
            
        Returns:
            True if stage completed successfully, False otherwise
        """
        stage_logger = self.logger.create_stage_logger(stage_name)
        stage_logger.log_stage_start(stage_name)
        
        # Log the arguments
        if args:
            args_str = " ".join(args)
            stage_logger.info(f"Stage arguments: {args_str}")
        
        try:
            return_code, stdout, stderr = self.execute_stage(stage_name, args, wait)
            
            success = return_code == 0
            stage_logger.log_stage_end(stage_name, success=success)
            
            if not success:
                stage_logger.error(f"Stage failed: {stderr}")
            
            return success
            
        except Exception as e:
            stage_logger.error(f"Stage exception: {str(e)}")
            stage_logger.log_stage_end(stage_name, success=False)
            return False

# Global instance for singleton-like access
_execution_manager_instance = None

def get_execution_manager() -> ExecutionManager:
    """
    Get a global instance of the ExecutionManager.
    
    Returns:
        ExecutionManager instance
    """
    global _execution_manager_instance
    if _execution_manager_instance is None:
        _execution_manager_instance = ExecutionManager()
    return _execution_manager_instance