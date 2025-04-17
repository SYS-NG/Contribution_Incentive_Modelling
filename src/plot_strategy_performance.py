import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_quality_participation_relationship(data_path, output_base_dir):
    """
    Create plots that visualize the relationship between quality and participation
    for different reward strategies across effort costs.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    perf_dir = os.path.join(output_base_dir, "strategy_performance")
    os.makedirs(perf_dir, exist_ok=True)
    
    # Plot 1: Scatter plot of quality vs participation across all strategies and effort costs
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot with different colors for each strategy
    sns.scatterplot(
        data=df,
        x='participation_rate',
        y='avg_quality',
        hue='reward_strategy',
        size='mean_effort_cost',  # Size indicates effort cost
        sizes=(50, 200),          # Range of point sizes
        style='reward_strategy',  # Different marker styles for strategies
        s=100,                    # Base point size
        alpha=0.8                 # Transparency
    )
    
    # Add text labels for each point showing the effort cost
    for idx, row in df.iterrows():
        plt.text(row['participation_rate'] + 0.02, 
                 row['avg_quality'] + 0.02, 
                 f"{row['mean_effort_cost']}", 
                 fontsize=9)
    
    # Customize the plot
    plt.title('Quality vs Participation by Reward Strategy and Effort Cost', fontsize=14)
    plt.xlabel('Participation Rate', fontsize=12)
    plt.ylabel('Average Quality', fontsize=12)
    
    # Add custom legend
    plt.legend(title='Reward Strategy', fontsize=10, title_fontsize=12)
    
    # Save the plot
    output_path = os.path.join(perf_dir, "quality_vs_participation_scatter.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved quality vs participation scatter plot to {output_path}")
    
    # Plot 2: Line plots showing how quality and participation change with effort cost
    # One subplot per reward strategy
    reward_strategies = df['reward_strategy'].unique()
    
    fig, axes = plt.subplots(len(reward_strategies), 1, figsize=(10, 4 * len(reward_strategies)), sharex=True)
    
    for i, strategy in enumerate(reward_strategies):
        ax = axes[i]
        strategy_data = df[df['reward_strategy'] == strategy].sort_values('mean_effort_cost')
        
        # Primary y-axis for quality
        color = "tab:blue"
        ax.set_xlabel('Mean Effort Cost', fontsize=12)
        ax.set_ylabel('Average Quality', color=color, fontsize=12)
        ax.plot(strategy_data['mean_effort_cost'], strategy_data['avg_quality'], 
                color=color, marker='o', linewidth=2, label='Avg Quality')
        ax.tick_params(axis='y', labelcolor=color)
        
        # Create a secondary y-axis for participation rate
        ax2 = ax.twinx()
        color = "tab:red"
        ax2.set_ylabel('Participation Rate', color=color, fontsize=12)
        ax2.plot(strategy_data['mean_effort_cost'], strategy_data['participation_rate'], 
                 color=color, marker='s', linewidth=2, label='Participation')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title for each subplot
        ax.set_title(f'Quality and Participation vs Effort Cost for {strategy} Strategy', fontsize=14)
        
        # Add text labels for each point
        for j, row in strategy_data.iterrows():
            ax.text(row['mean_effort_cost'] + 0.05, row['avg_quality'] - 0.02, 
                   f"{row['avg_quality']:.2f}", color='tab:blue', fontsize=9)
            ax2.text(row['mean_effort_cost'] + 0.05, row['participation_rate'] + 0.02, 
                    f"{row['participation_rate']:.2f}", color='tab:red', fontsize=9)
        
        # Create a combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    output_path = os.path.join(perf_dir, "quality_participation_by_effort_cost.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved quality and participation vs effort cost plot to {output_path}")
    
    # Plot 3: Platform utility vs effort cost for each strategy
    plt.figure(figsize=(12, 8))
    
    # Plot utility lines for each strategy
    for strategy in reward_strategies:
        strategy_data = df[df['reward_strategy'] == strategy].sort_values('mean_effort_cost')
        plt.plot(strategy_data['mean_effort_cost'], strategy_data['platform_utility'], 
                marker='o', linewidth=2, label=strategy)
        
        # Add text labels for utility values
        for idx, row in strategy_data.iterrows():
            plt.text(row['mean_effort_cost'] + 0.05, row['platform_utility'] + 5, 
                    f"{row['platform_utility']:.1f}", fontsize=9)
    
    # Add horizontal line at y=0 to show break-even point
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    
    # Customize the plot
    plt.title('Platform Utility vs Effort Cost by Reward Strategy', fontsize=14)
    plt.xlabel('Mean Effort Cost', fontsize=12)
    plt.ylabel('Platform Utility', fontsize=12)
    plt.legend(title='Strategy', loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(perf_dir, "platform_utility_by_effort_cost.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved platform utility vs effort cost plot to {output_path}")

def plot_contributor_type_comparison(data_path, output_base_dir):
    """
    Create plots comparing extrinsic vs intrinsic participation across strategies.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    contrib_dir = os.path.join(output_base_dir, "contributor_comparison")
    os.makedirs(contrib_dir, exist_ok=True)
    
    # Create a stacked bar chart for each effort cost level
    effort_levels = df['mean_effort_cost'].unique()
    
    for effort_cost in effort_levels:
        plt.figure(figsize=(12, 7))
        
        # Filter data for this effort cost
        df_effort = df[df['mean_effort_cost'] == effort_cost]
        
        # Prepare data for stacked bars
        strategies = df_effort['reward_strategy'].values
        ext_part = df_effort['extrinsic_participation'].values
        int_part = df_effort['intrinsic_participation'].values
        
        # Set width of bars
        bar_width = 0.35
        x = np.arange(len(strategies))
        
        # Create stacked bars
        plt.bar(x, ext_part, bar_width, label='Extrinsic Participants', color='tab:blue')
        plt.bar(x, int_part, bar_width, bottom=ext_part, label='Intrinsic Participants', color='tab:orange')
        
        # Add text labels for each segment
        for i in range(len(strategies)):
            # Extrinsic label (middle of its segment)
            plt.text(i, ext_part[i]/2, f"{ext_part[i]:.2f}", ha='center', va='center', color='white', fontweight='bold')
            
            # Intrinsic label (middle of its segment)
            plt.text(i, ext_part[i] + int_part[i]/2, f"{int_part[i]:.2f}", ha='center', va='center', color='black', fontweight='bold')
        
        # Customize the plot
        plt.xlabel('Reward Strategy', fontsize=12)
        plt.ylabel('Participation Rate', fontsize=12)
        plt.title(f'Extrinsic vs Intrinsic Participation by Strategy (Effort Cost = {effort_cost})', fontsize=14)
        plt.xticks(x, strategies)
        plt.legend()
        plt.ylim(0, 2.0)  # Set y-limit to accommodate stacked values up to 2.0 (theoretical max)
        
        # Add a secondary y-axis for the ratio of intrinsic to extrinsic
        ax2 = plt.twinx()
        ratio = int_part / ext_part
        ax2.plot(x, ratio, 'r-', marker='o', linewidth=2, label='Intrinsic/Extrinsic Ratio')
        ax2.set_ylabel('Intrinsic/Extrinsic Ratio', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add ratio values as text
        for i in range(len(strategies)):
            ax2.text(i + 0.1, ratio[i] + 0.05, f"{ratio[i]:.2f}", color='r', fontweight='bold')
        
        # Create a combined legend
        lines, labels = ax2.get_legend_handles_labels()
        plt.legend(lines, labels, loc='upper right')
        
        # Save the plot
        output_path = os.path.join(contrib_dir, f"contributor_type_comparison_{effort_cost}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved contributor type comparison for effort cost {effort_cost} to {output_path}")
    
    # Create an overall view across all effort costs
    plt.figure(figsize=(15, 10))
    
    # Set up the grid
    reward_strategies = df['reward_strategy'].unique()
    n_strategies = len(reward_strategies)
    n_effort_levels = len(effort_levels)
    
    # Define the bar positions
    group_width = 0.8
    bar_width = group_width / n_effort_levels
    offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, n_effort_levels)
    
    # Create grouped bars for extrinsic and intrinsic participation
    for i, strategy in enumerate(reward_strategies):
        strategy_df = df[df['reward_strategy'] == strategy]
        
        for j, effort in enumerate(effort_levels):
            effort_row = strategy_df[strategy_df['mean_effort_cost'] == effort]
            if not effort_row.empty:
                ext_val = effort_row['extrinsic_participation'].values[0]
                int_val = effort_row['intrinsic_participation'].values[0]
                
                # Position for this bar
                x_pos = i + offsets[j]
                
                # Create the stacked bars
                plt.bar(x_pos, ext_val, bar_width, label=f'Extrinsic (Cost={effort})' if i == 0 else "", 
                        color=plt.cm.Blues(j/n_effort_levels + 0.3))
                plt.bar(x_pos, int_val, bar_width, bottom=ext_val, 
                        label=f'Intrinsic (Cost={effort})' if i == 0 else "", 
                        color=plt.cm.Oranges(j/n_effort_levels + 0.3))
    
    # Customize the plot
    plt.xlabel('Reward Strategy', fontsize=14)
    plt.ylabel('Participation Rate', fontsize=14)
    plt.title('Extrinsic vs Intrinsic Participation by Strategy and Effort Cost', fontsize=16)
    plt.xticks(np.arange(n_strategies), reward_strategies)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(contrib_dir, "contributor_type_all_effort_costs.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overall contributor type comparison to {output_path}")

def main():
    # Input file containing all experiment results
    data_path = "results/comparison/experiment_all_results.csv"
    
    # Base output directory for plots
    output_base_dir = "results/strategy_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create quality and participation relationship plots
    plot_quality_participation_relationship(data_path, output_base_dir)
    
    # Create contributor type comparison plots
    plot_contributor_type_comparison(data_path, output_base_dir)
    
    logger.info(f"All strategy performance plots saved to {output_base_dir}")

if __name__ == "__main__":
    main() 