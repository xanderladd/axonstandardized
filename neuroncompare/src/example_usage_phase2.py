# example_phase2_usage.py
from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.logger import get_logger
from neuroncompare.src.execution_manager import get_execution_manager
from neuroncompare.src.file_waiter import get_file_waiter

def main():
    # Initialize components
    config = get_config()
    path_manager = get_path_manager()
    logger = get_logger()
    execution_manager = get_execution_manager()
    file_waiter = get_file_waiter()
    
    # Log start of execution
    logger.info("Starting pipeline execution")
    
    # Create run directory and subdirectories
    path_manager.create_run_subdirs()
    logger.info(f"Created run directory at {path_manager.get_run_dir()}")
    
    # Example: Execute cell ingestion if configured
    if config.get('ingestCell', False):
        logger.info("Starting cell ingestion")
        success = execution_manager.execute_pipeline_stage('cell_ingest', 
                                                         [f"--pull", f"--cell_id={config['modelNum']}"])
        if not success:
            logger.error("Cell ingestion failed")
            return
    
    # Example: Generate parameters with auto-confirm
    if config.get('makeParams', False):
        logger.info("Starting parameter generation")
        # Add auto-confirm flag to make it non-interactive in automation
        success = execution_manager.execute_pipeline_stage('make_params', ['--auto-confirm'])
        if not success:
            logger.error("Parameter generation failed")
            return
    
    # Rest of the pipeline stages...
    
    logger.info("Pipeline execution completed successfully")

if __name__ == "__main__":
    main()