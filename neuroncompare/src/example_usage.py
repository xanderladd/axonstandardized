# example_usage.py
from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.logger import get_logger

def main():
    # Initialize components
    config = get_config()
    path_manager = get_path_manager()
    logger = get_logger()
    
    # Log some basic info
    logger.info("Starting example script")
    
    # Create run directory and subdirectories
    path_manager.create_run_subdirs()
    logger.info(f"Created run directory at {path_manager.get_run_dir()}")
    
    # Log configuration details
    logger.info(f"Running with model: {config['model']}")
    logger.info(f"Peeling type: {config['peeling']}")
    
    # Get various paths
    params_path = path_manager.get_params_path()
    stims_path = path_manager.get_stims_path()
    target_volts_path = path_manager.get_target_volts_path()
    
    logger.info(f"Params path: {params_path}")
    logger.info(f"Stims path: {stims_path}")
    logger.info(f"Target volts path: {target_volts_path}")
    
    # Example of stage-specific logging
    stage_logger = logger.create_stage_logger("make_volts")
    stage_logger.log_stage_start("make_volts")
    stage_logger.info("Generating voltage traces")
    stage_logger.log_stage_end("make_volts")
    
    # Example of logging a command
    volts_script = path_manager.get_script_path("volts")
    logger.log_command(f"sh {volts_script}")
    
    # Example of logging with context
    logger.info("Processing completed", files_processed=10, errors=0)
    
    logger.info("Example script completed")

if __name__ == "__main__":
    main()