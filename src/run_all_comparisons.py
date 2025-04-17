import os
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_plotting_scripts():
    """
    Run all the comparison plotting scripts to generate visualizations.
    """
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of scripts to run
    scripts = [
        "plot_strategy_comparisons.py",
        "plot_multimetric_comparisons.py",
        "plot_strategy_performance.py",
        "plot_skill_types.py"
    ]
    
    start_time = time.time()
    
    logger.info("=== Starting to generate all comparison plots ===")
    
    # Run each script
    for script in scripts:
        script_path = os.path.join(current_dir, script)
        
        if os.path.exists(script_path):
            logger.info(f"Running {script}...")
            
            try:
                # Run the script
                result = subprocess.run(
                    ["python", script_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Log the output
                for line in result.stdout.splitlines():
                    if line.strip():
                        logger.info(f"{script}: {line}")
                
                logger.info(f"Successfully completed {script}")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running {script}: {e}")
                logger.error(f"Error output: {e.stderr}")
        else:
            logger.error(f"Script not found: {script_path}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"=== All comparison plots generated in {elapsed_time:.2f} seconds ===")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(current_dir), "results", "strategy_comparison")
    logger.info(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    run_plotting_scripts() 