import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_multimetric_comparison(data_path, output_base_dir):
    """
    Create multi-metric comparison plots for each mean effort cost level.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    multimetric_dir = os.path.join(output_base_dir, "multimetric_comparison")
    os.makedirs(multimetric_dir, exist_ok=True)
    
    # Group metrics for different subplots
    metric_groups = [
        # Performance metrics
        ['avg_quality', 'participation_rate', 'platform_utility'],
        # Participation breakdown - now including no_contribution_rate
        ['high_quality_rate', 'low_quality_rate', 'no_contribution_rate'],
        # Contributor type metrics
        ['extrinsic_participation', 'intrinsic_participation']
    ]
    
    metric_titles = {
        'avg_quality': 'Average Quality',
        'participation_rate': 'Participation Rate',
        'high_quality_rate': 'High Quality Rate',
        'low_quality_rate': 'Low Quality Rate',
        'no_contribution_rate': 'No Contribution Rate',
        'extrinsic_participation': 'Extrinsic Participation',
        'intrinsic_participation': 'Intrinsic Participation',
        'platform_utility': 'Platform Utility'
    }
    
    # Create plots for each effort cost level
    effort_levels = df['mean_effort_cost'].unique()
    
    for effort_cost in effort_levels:
        logger.info(f"Creating multi-metric plots for mean effort cost = {effort_cost}")
        
        # Filter data for this effort cost
        df_effort = df[df['mean_effort_cost'] == effort_cost]
        
        # Create directory for this effort cost level
        effort_dir = os.path.join(multimetric_dir, f"effort_cost_{effort_cost}")
        os.makedirs(effort_dir, exist_ok=True)
        
        # Create a subplot for each group of metrics
        for i, metrics in enumerate(metric_groups):
            group_name = ["performance", "contribution_types", "participant_types"][i]
            
            # Determine grid size based on number of metrics
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6), sharey=False)
            
            # If there's only one metric, wrap axes in a list
            if n_metrics == 1:
                axes = [axes]
                
            # Special handling for platform utility which can have very different scale
            utility_scale = None
            if 'platform_utility' in metrics:
                utility_idx = metrics.index('platform_utility')
                utility_scale = (df_effort['platform_utility'].min(), df_effort['platform_utility'].max())
            
            # Plot each metric
            for j, metric in enumerate(metrics):
                ax = axes[j]
                
                # Adjust y-limits for platform utility
                if metric == 'platform_utility' and utility_scale:
                    # Add some padding to the limits
                    y_min = utility_scale[0] - abs(utility_scale[0] * 0.1)
                    y_max = utility_scale[1] + abs(utility_scale[1] * 0.1)
                    ax.set_ylim(y_min, y_max)
                
                # Create bar plot
                sns.barplot(x='reward_strategy', y=metric, data=df_effort, palette='viridis', ax=ax)
                
                # Set titles and labels
                ax.set_title(metric_titles[metric], fontsize=14)
                ax.set_xlabel('Reward Strategy', fontsize=12)
                ax.set_ylabel(metric_titles[metric], fontsize=12)
                
                # Rotate x-tick labels
                plt.sca(ax)
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.3f}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha='center', va='bottom', fontsize=9, rotation=0)
            
            # Add overall title
            fig.suptitle(f'Reward Strategy Comparison - Effort Cost {effort_cost} - {group_name.replace("_", " ").title()}', 
                        fontsize=16, y=1.05)
            
            # Adjust layout and save figure
            plt.tight_layout()
            output_path = os.path.join(effort_dir, f"{group_name}_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved {group_name} comparison to {output_path}")
    
    logger.info("All multi-metric comparison charts created successfully.")

def create_heatmap_comparison(data_path, output_base_dir):
    """
    Create heatmap visualizations for comparing metrics across strategies and effort costs.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    heatmap_dir = os.path.join(output_base_dir, "heatmap_comparison")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Key metrics to visualize - now including no_contribution_rate
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
    
    # Create pivot tables for each metric and plot heatmaps
    for metric in metrics:
        logger.info(f"Creating heatmap for {metric}")
        
        # Create pivot table with effort cost as rows and reward strategy as columns
        pivot_df = df.pivot_table(
            index='mean_effort_cost', 
            columns='reward_strategy', 
            values=metric,
            aggfunc='mean'
        )
        
        # Choose appropriate colormap based on metric
        if metric == 'platform_utility':
            # Use diverging colormap for utility (which can be negative)
            cmap = 'RdBu_r'
            center = 0
        elif metric == 'no_contribution_rate':
            # Use reversed colormap for no_contribution_rate (higher is worse)
            cmap = 'viridis_r'
            center = None
        else:
            # Use sequential colormap for percentage metrics
            cmap = 'viridis'
            center = None
        
        # Create figure and plot heatmap
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.3f', 
            cmap=cmap,
            center=center,
            linewidths=.5,
            cbar_kws={'label': metric}
        )
        
        # Set title and labels
        metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
        plt.title(f'{metric_name} by Reward Strategy and Effort Cost', fontsize=14)
        plt.ylabel('Mean Effort Cost', fontsize=12)
        plt.xlabel('Reward Strategy', fontsize=12)
        
        # Adjust layout and save figure
        plt.tight_layout()
        output_path = os.path.join(heatmap_dir, f"{metric}_heatmap.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved {metric} heatmap to {output_path}")
    
    logger.info("All heatmap comparisons created successfully.")

def main():
    # Input file containing all experiment results
    data_path = "results/comparison/experiment_all_results.csv"
    
    # Base output directory for plots
    output_base_dir = "results/strategy_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create multi-metric comparison plots
    create_multimetric_comparison(data_path, output_base_dir)
    
    # Create heatmap visualizations
    create_heatmap_comparison(data_path, output_base_dir)
    
    logger.info(f"All plots saved to {output_base_dir}")

if __name__ == "__main__":
    main() 