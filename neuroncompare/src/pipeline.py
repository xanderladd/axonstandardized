# neuroncompare/src/pipeline.py
import os
import time
import shutil
from typing import List, Dict, Any, Optional

from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.logger import get_logger
from neuroncompare.src.execution_manager import get_execution_manager
from neuroncompare.src.file_waiter import get_file_waiter

class Pipeline:
    """
    Main pipeline class for neuroncompare.
    Orchestrates the execution of the entire pipeline.
    """
    
    def __init__(self):
        """Initialize the Pipeline."""
        self.root_dir = os.getcwd()
        os.environ['NEURON_COMPARE_ROOT'] = self.root_dir
        assert os.path.isfile(os.path.join( self.root_dir, 'input.txt')), "must launch from a path with input.txt in the root dir"
        self.config = get_config()
        self.path_manager = get_path_manager()
        self.logger = get_logger()
        self.execution_manager = get_execution_manager()
        self.file_waiter = get_file_waiter()
        
        # Create pipeline logger
        self.logger = get_logger("neuroncompare.pipeline")
    
    def setup_directories(self) -> bool:
        """
        Create and set up the directory structure for the pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the main run directory and subdirectories
            self.path_manager.create_run_subdirs()
            
            # Copy input.txt to the run directory
            input_file_path = self.config.input_file_path
            run_input_path = self.path_manager.get_run_file("input.txt")
            shutil.copy2(input_file_path, run_input_path)
            
            self.logger.info(f"Created run directory at {self.path_manager.get_run_dir()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up directories: {str(e)}")
            return False
    
    def ingest_cell(self) -> bool:
        """
        Run cell ingestion.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.config.get('ingestCell', False):
            self.logger.info("Skipping cell ingestion (not configured)")
            return True
            
        self.logger.info("Starting cell ingestion")
        # Now passing arguments properly preserving order
        return self.execution_manager.execute_pipeline_stage(
            'cell_ingest', 
            ['pull', f"--cell_id={self.config['modelNum']}"]
        )
    
    def make_stims(self) -> bool:
        """
        Generate stimulus files.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.config.get('makeStims', False):
            self.logger.info("Skipping stim generation (not configured)")
            return True
            
        self.logger.info("Starting stim generation")
        return self.execution_manager.execute_pipeline_stage(
            'cell_ingest',
            ['assemble', f"--model={self.config['modelNum']}", 
             '--pdf', '--force', f"--timestep={self.config['timesteps']}"]
        )
    
    def check_files(self) -> bool:
        """
        Check that required files exist.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Checking required files")
        return self.execution_manager.execute_pipeline_stage(
            'check_files', 
            [self.config['modelNum'], 
             str(self.config.get('passive', False)),
             self.config['data_dir']]
        )
    
    def copy_to_run(self) -> bool:
        """
        Copy files to the run directory.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Copying files to run directory")
        return self.execution_manager.execute_pipeline_stage(
            'cell_ingest',
            ['copy_to_run', f"--model={self.config['modelNum']}", 
             f"--passive={self.config.get('passive', False)}", 
             f"--dest={self.path_manager.get_run_dir()}"]
        )
    
    def make_params(self) -> bool:
        """
        Generate parameter files.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.config.get('makeParams', False):
            self.logger.info("Skipping parameter generation (not configured)")
            return True
            
        self.logger.info("Starting parameter generation")
        return self.execution_manager.execute_pipeline_stage(
            'make_params', 
            ['--auto-confirm']
        )
    
    def modify_sandbox_array(self) -> bool:
        """
        Configure sandbox array for parallelization.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Configuring sandbox array")
        return self.execution_manager.execute_pipeline_stage(
            'modifySandboxArray', 
            [str(self.config.get('num_volts', 0)), 
             str(self.config.get('num_nodes', 1))]
        )
    
    def make_volts(self) -> bool:
        """
        Generate voltage traces.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('makeVolts', False):
            self.logger.info("Skipping voltage generation (not configured)")
            return True
            
        self.logger.info("Starting voltage generation")

        success = self.execution_manager.execute_pipeline_stage('volts')
            
        if not success:
            return False
        return True

    def wait_for_volts(self) -> bool:
        # Wait for voltage files if configured
        if self.config.get('wait4volts', False):
            self.logger.info("Waiting for voltage files")
            
            stim_file = self.path_manager.get_stims_path()
            volt_dir = self.path_manager.get_subdirectory('volts')
            
            if not self.file_waiter.wait_for_volts_files(stim_file, volt_dir):
                self.logger.error("Timed out waiting for voltage files")
                return False
                
            self.logger.info("All voltage files found")
            
        return True
    
    def make_scores(self) -> bool:
        """
        Calculate scores for voltage traces.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('makeScores', False):
            self.logger.info("Skipping score calculation (not configured)")
            return True
            
        self.logger.info("Starting score calculation")
        success = self.execution_manager.execute_pipeline_stage('scores')
            
        if not success:
            return False
            
        # Wait for score files if configured
        if self.config.get('wait4scores', False):
            self.logger.info("Waiting for score files")
            
            stim_file = self.path_manager.get_stims_path()
            score_dir = self.path_manager.get_subdirectory('scores')
            
            if not self.file_waiter.wait_for_scores_files(stim_file, score_dir):
                self.logger.error("Timed out waiting for score files")
                return False
                
            self.logger.info("All score files found")
            
        return True
    
    def setup_genetic_alg(self) -> bool:
        """
        Set up genetic algorithm directories.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Setting up genetic algorithm directories")
            
            run_dir = self.path_manager.get_run_dir()
            ga_dir = os.path.join(run_dir, "genetic_alg")
            os.makedirs(ga_dir, exist_ok=True)
            
            # Copy ga_dir directory
            source_neuron_ga_dir = os.path.join(self.path_manager.base_dir, "genetic_alg")
            self.copy_directory(source_neuron_ga_dir, ga_dir)
            
            # Copy models into neuron_genetic_alg directory
            source_cell_dir = os.path.join(self.path_manager.base_dir, "cell_models")
            dest_cell_dir = os.path.join(ga_dir,"genetic_alg","cell_models")
            os.makedirs(dest_cell_dir, exist_ok=True)
            self.copy_directory(source_cell_dir, dest_cell_dir)
            

            # Create directories for results
            os.makedirs(os.path.join(ga_dir, "optimization_results"), exist_ok=True)
            os.makedirs(os.path.join(ga_dir, "objectives"), exist_ok=True)
            
            self.logger.info("Genetic algorithm directories set up")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up genetic algorithm directories: {str(e)}")
            return False
    
    def make_opt(self) -> bool:
        """
        Run optimization.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('makeOpt', False):
            self.logger.info("Skipping optimization (not configured)")
            return True
            
        self.logger.info("Starting optimization")
        
        # Change to run directory for optimization
        run_dir = self.path_manager.get_run_dir()
        
        success = self.execution_manager.execute_pipeline_stage(
            'analyze_p_parallel',
            [f"--model={self.config['model']}",
             f"--peeling={self.config['peeling']}",
             f"--CURRENTDATE={self.config['runDate']}",
             f"--custom={self.config.get('custom', '')}"],
            work_dir=run_dir
        )
            
        if not success:
            return False
            
        # Wait for optimization file
        self.logger.info("Waiting for optimization file")
        
        opt_file = os.path.join(
            run_dir, 
            f"genetic_alg/optimization_results/opt_result_single_stim_{self.config['model']}_{self.config['peeling']}_full.hdf5"
        )
        
        if not self.file_waiter.wait_for_file(opt_file):
            self.logger.error("Timed out waiting for optimization file")
            return False
            
        self.logger.info("Optimization file found")
        return True
    
    def make_obj(self) -> bool:
        """
        Generate objectives.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('makeObj', False):
            self.logger.info("Skipping objective generation (not configured)")
            return True
            
        self.logger.info("Starting objective generation")
        
        # Change to run directory for objective generation
        run_dir = self.path_manager.get_run_dir()
        os.makedirs(os.path.join(run_dir,'analyze_p_bbp_full'), exist_ok=True)
        return self.execution_manager.execute_pipeline_stage(
            'analyze_p_multistims',
            [f"--model={self.config['model']}",
             f"--peeling={self.config['peeling']}",
             f"--CURRENTDATE={self.config['runDate']}",
             f"--custom={self.config.get('custom', '')}"],
            work_dir=run_dir
        )
    
    def run_ga(self) -> bool:
        """
        Run genetic algorithm.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('runGA', False):
            self.logger.info("Skipping genetic algorithm (not configured)")
            return True
            
        self.logger.info("Starting genetic algorithm")
        
        # Change to the appropriate directory for GA
        ga_dir = os.path.join(self.path_manager.get_run_dir(), "genetic_alg/neuron_genetic_alg/slurm_scripts")
        
        return self.execution_manager.execute_pipeline_stage(
                'genetic_algorithm', 
                [],
                work_dir=ga_dir
            )

    def compare_models(self) -> bool:
        """
        Run genetic algorithm.
        
        Returns:
            True if successful or not configured, False otherwise
        """
        if not self.config.get('compare_models', False):
            self.logger.info(" compare cells (not configured)")
            return True
            
        self.logger.info("Starting compare cells")
        
        # Change to the appropriate directory for GA
        ga_dir = os.path.join(self.path_manager.get_run_dir(), "genetic_alg")
        
        return self.execution_manager.execute_pipeline_stage(
                'compare_models', 
                [],
                work_dir=ga_dir
            )
    
    def copy_directory(self, src: str, dst: str) -> None:
        """
        Copy a directory and its contents.
        
        Args:
            src: Source directory path
            dst: Destination directory path
        """
        if not os.path.exists(dst):
            os.makedirs(dst)
            
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            
            if os.path.isdir(s):
                self.copy_directory(s, d)
            else:
                shutil.copy2(s, d)
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        self.logger.info("Starting neuroncompare pipeline")
        
        # Set up directories
        if not self.setup_directories():
            self.logger.error("Failed to set up directories")
            return False
        
        # Run the pipeline stages
        pipeline_stages = [
            ("Cell Ingestion", self.ingest_cell),
            ("Stim Generation", self.make_stims),
            ("File Check", self.check_files),
            ("Copy to Run", self.copy_to_run),
            ("Parameter Generation", self.make_params),
            ("Sandbox Configuration", self.modify_sandbox_array),
            ("Voltage Generation", self.make_volts),
            ("Wait for Volts", self.wait_for_volts),
            ("Score Calculation", self.make_scores),
            ("Genetic Algorithm Setup", self.setup_genetic_alg),
            ("Optimization", self.make_opt),
            ("Objective Generation", self.make_obj),
            ("Genetic Algorithm", self.run_ga),
            ("Run Comparision", self.compare_models)

        ]
        
        for stage_name, stage_func in pipeline_stages:
            stage_start_time = time.time()
            self.logger.info(f"Starting {stage_name}")
            
            if not stage_func():
                self.logger.error(f"{stage_name} failed")
                return False
                
            stage_end_time = time.time()
            stage_duration = stage_end_time - stage_start_time
            self.logger.info(f"Completed {stage_name} in {stage_duration:.2f} seconds")
        
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
        return True

def main():
    """
    Main entry point for running the neuroncompare pipeline.
    """
    pipeline = Pipeline()
    success = pipeline.run()
    
    if not success:
        print("Pipeline failed. Check logs for details.")
        return 1
        
    print("Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    exit(main())