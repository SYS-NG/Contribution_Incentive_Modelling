import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_comparison_bar_charts(data_path, output_base_dir):
    """
    Create bar charts comparing different reward strategies for each mean effort cost.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Metrics to plot - added no_contribution_rate
    metrics = [
        'avg_quality', 
        'participation_rate', 
        'high_quality_rate', 
        'low_quality_rate', 
        'no_contribution_rate',
        'extrinsic_participation', 
        'intrinsic_participation', 
        'platform_utility'
    ]
    
    # Create output directories based on mean effort cost
    effort_levels = df['mean_effort_cost'].unique()
    
    for effort_cost in effort_levels:
        # Create directory for this effort cost level
        effort_dir = os.path.join(output_base_dir, f"effort_cost_{effort_cost}")
        os.makedirs(effort_dir, exist_ok=True)
        logger.info(f"Creating plots for mean effort cost = {effort_cost}")
        
        # Filter data for this effort cost
        df_effort = df[df['mean_effort_cost'] == effort_cost]
        
        # Plot each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use seaborn for better aesthetics
            sns.barplot(x='reward_strategy', y=metric, data=df_effort, palette='viridis', ax=ax)
            
            # Add title and labels
            metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
            ax.set_title(f'{metric_name} by Reward Strategy (Effort Cost = {effort_cost})', fontsize=14)
            ax.set_xlabel('Reward Strategy', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            
            # Rotate x-tick labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_height():.3f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom', fontsize=10, rotation=0)
            
            # Adjust layout and save figure
            plt.tight_layout()
            output_path = os.path.join(effort_dir, f"{metric}_comparison.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved {metric} comparison to {output_path}")

    logger.info("All comparison bar charts created successfully.")

def create_strategy_overview_grid(data_path, output_base_dir):
    """
    Create a grid of bar charts for each metric showing all effort costs across reward strategies.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create the overview directory
    overview_dir = os.path.join(output_base_dir, "strategy_overview")
    os.makedirs(overview_dir, exist_ok=True)
    
    # Metrics to plot - added no_contribution_rate
    metrics = [
        'avg_quality', 
        'participation_rate', 
        'high_quality_rate', 
        'low_quality_rate', 
        'no_contribution_rate',
        'extrinsic_participation', 
        'intrinsic_participation', 
        'platform_utility'
    ]
    
    # Plot each metric across all effort costs and strategies
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use seaborn to create a grouped bar chart
        sns.barplot(x='mean_effort_cost', y=metric, hue='reward_strategy', data=df, palette='viridis', ax=ax)
        
        # Add title and labels
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        ax.set_title(f'{metric_name} by Reward Strategy and Effort Cost', fontsize=14)
        ax.set_xlabel('Mean Effort Cost', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        
        # Adjust legend
        ax.legend(title='Reward Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save figure
        plt.tight_layout()
        output_path = os.path.join(overview_dir, f"{metric}_all_strategies.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved {metric} overview to {output_path}")
    
    logger.info("All overview charts created successfully.")

def main():
    # Input file containing all experiment results
    data_path = "results/comparison/experiment_all_results.csv"
    
    # Base output directory for plots
    output_base_dir = "results/strategy_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create the comparison bar charts by effort cost
    create_comparison_bar_charts(data_path, output_base_dir)
    
    # Create the strategy overview grid
    create_strategy_overview_grid(data_path, output_base_dir)
    
    logger.info(f"All plots saved to {output_base_dir}")

if __name__ == "__main__":
    main() 