import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_skill_type_comparisons(data_path, output_base_dir):
    """
    Create plots comparing skill types (high vs. low quality contributions) across different reward
    strategies and effort costs.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    skill_dir = os.path.join(output_base_dir, "skill_comparison")
    os.makedirs(skill_dir, exist_ok=True)
    
    # Create a stacked bar chart for each effort cost level
    effort_levels = df['mean_effort_cost'].unique()
    
    for effort_cost in effort_levels:
        # Create directory for this effort cost level
        effort_dir = os.path.join(skill_dir, f"effort_cost_{effort_cost}")
        os.makedirs(effort_dir, exist_ok=True)
        
        logger.info(f"Creating skill type comparison plots for mean effort cost = {effort_cost}")
        
        # Filter data for this effort cost
        df_effort = df[df['mean_effort_cost'] == effort_cost]
        
        # Plot 1: Stacked bar chart of high quality, low quality, and no contribution
        plt.figure(figsize=(12, 8))
        
        # Prepare data for stacked bars
        strategies = df_effort['reward_strategy'].values
        high_quality = df_effort['high_quality_rate'].values
        low_quality = df_effort['low_quality_rate'].values
        no_contribution = df_effort['no_contribution_rate'].values
        
        # Set width of bars
        bar_width = 0.6
        x = np.arange(len(strategies))
        
        # Create stacked bars
        plt.bar(x, high_quality, bar_width, label='High Quality', color='#2ca02c')
        plt.bar(x, low_quality, bar_width, bottom=high_quality, label='Low Quality', color='#ff7f0e')
        plt.bar(x, no_contribution, bar_width, bottom=high_quality+low_quality, label='No Contribution', color='#d62728')
        
        # Add text labels for each segment
        for i in range(len(strategies)):
            # High quality label (middle of its segment)
            plt.text(i, high_quality[i]/2, f"{high_quality[i]:.2f}", 
                    ha='center', va='center', color='white', fontweight='bold')
            
            # Low quality label (middle of its segment)
            plt.text(i, high_quality[i] + low_quality[i]/2, f"{low_quality[i]:.2f}", 
                    ha='center', va='center', color='black', fontweight='bold')
            
            # No contribution label (middle of its segment)
            plt.text(i, high_quality[i] + low_quality[i] + no_contribution[i]/2, f"{no_contribution[i]:.2f}", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        # Customize the plot
        plt.xlabel('Reward Strategy', fontsize=14)
        plt.ylabel('Rate', fontsize=14)
        plt.title(f'Contribution Types by Reward Strategy (Effort Cost = {effort_cost})', fontsize=16)
        plt.xticks(x, strategies)
        plt.legend(loc='upper right')
        plt.ylim(0, 1.05)  # Set y-limit to 1.05 to accommodate all stacked values and labels
        plt.grid(axis='y', alpha=0.3)
        
        # Save the plot
        output_path = os.path.join(effort_dir, "contribution_types_stacked.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved contribution types stacked bar chart to {output_path}")
        
        # Plot 2: High quality to low quality ratio
        plt.figure(figsize=(10, 6))
        
        # Calculate high quality to low quality ratio
        high_to_low_ratio = np.divide(high_quality, low_quality, out=np.zeros_like(high_quality), where=low_quality!=0)
        
        # Create bar chart
        bars = plt.bar(x, high_to_low_ratio, color='#1f77b4')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Customize the plot
        plt.xlabel('Reward Strategy', fontsize=14)
        plt.ylabel('High Quality / Low Quality Ratio', fontsize=14)
        plt.title(f'High to Low Quality Ratio by Reward Strategy (Effort Cost = {effort_cost})', fontsize=16)
        plt.xticks(x, strategies)
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at y=1 to show equal high and low quality
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equal high and low')
        plt.legend()
        
        # Save the plot
        output_path = os.path.join(effort_dir, "high_to_low_quality_ratio.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved high to low quality ratio chart to {output_path}")
        
        # Plot 3: Participation quality breakdown (grouped bar chart)
        plt.figure(figsize=(12, 7))
        
        # Set position of bars on X axis
        bar_width = 0.25
        index = np.arange(len(strategies))
        
        # Create grouped bars
        plt.bar(index - bar_width, high_quality, bar_width, label='High Quality', color='#2ca02c')
        plt.bar(index, low_quality, bar_width, label='Low Quality', color='#ff7f0e')
        plt.bar(index + bar_width, no_contribution, bar_width, label='No Contribution', color='#d62728')
        
        # Add value labels above bars
        for i in range(len(strategies)):
            plt.text(i - bar_width, high_quality[i] + 0.02, f"{high_quality[i]:.2f}", 
                    ha='center', va='bottom', fontsize=9)
            plt.text(i, low_quality[i] + 0.02, f"{low_quality[i]:.2f}", 
                    ha='center', va='bottom', fontsize=9)
            plt.text(i + bar_width, no_contribution[i] + 0.02, f"{no_contribution[i]:.2f}", 
                    ha='center', va='bottom', fontsize=9)
        
        # Customize the plot
        plt.xlabel('Reward Strategy', fontsize=14)
        plt.ylabel('Rate', fontsize=14)
        plt.title(f'Contribution Types by Reward Strategy (Effort Cost = {effort_cost})', fontsize=16)
        plt.xticks(index, strategies)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Save the plot
        output_path = os.path.join(effort_dir, "contribution_types_grouped.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved contribution types grouped bar chart to {output_path}")
    
    # Plot 4: High quality rates across all reward strategies and effort costs
    plt.figure(figsize=(14, 8))
    
    # Create line plot for high quality rate
    sns.lineplot(
        data=df,
        x='mean_effort_cost',
        y='high_quality_rate',
        hue='reward_strategy',
        marker='o',
        linewidth=2,
        markersize=8
    )
    
    # Customize the plot
    plt.title('High Quality Contribution Rate vs Effort Cost by Reward Strategy', fontsize=16)
    plt.xlabel('Mean Effort Cost', fontsize=14)
    plt.ylabel('High Quality Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Reward Strategy', fontsize=12, title_fontsize=13)
    
    # Save the plot
    output_path = os.path.join(skill_dir, "high_quality_rate_by_effort_cost.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved high quality rate line plot to {output_path}")
    
    # Plot 5: No contribution rates across all reward strategies and effort costs
    plt.figure(figsize=(14, 8))
    
    # Create line plot for no contribution rate
    sns.lineplot(
        data=df,
        x='mean_effort_cost',
        y='no_contribution_rate',
        hue='reward_strategy',
        marker='o',
        linewidth=2,
        markersize=8
    )
    
    # Customize the plot
    plt.title('No Contribution Rate vs Effort Cost by Reward Strategy', fontsize=16)
    plt.xlabel('Mean Effort Cost', fontsize=14)
    plt.ylabel('No Contribution Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Reward Strategy', fontsize=12, title_fontsize=13)
    
    # Save the plot
    output_path = os.path.join(skill_dir, "no_contribution_rate_by_effort_cost.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved no contribution rate line plot to {output_path}")
    
    # Plot 6: Quality breakdown heatmap
    # Create a figure with 3 subplots for high quality, low quality, and no contribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Quality metrics to visualize
    quality_metrics = ['high_quality_rate', 'low_quality_rate', 'no_contribution_rate']
    titles = ['High Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
    
    for i, (metric, title) in enumerate(zip(quality_metrics, titles)):
        # Create pivot table
        pivot_df = df.pivot_table(
            index='mean_effort_cost', 
            columns='reward_strategy', 
            values=metric,
            aggfunc='mean'
        )
        
        # Create heatmap on corresponding subplot
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.3f', 
            cmap='viridis',
            linewidths=.5,
            ax=axes[i],
            cbar_kws={'label': metric}
        )
        
        # Set title and labels
        axes[i].set_title(title, fontsize=14)
        axes[i].set_ylabel('Mean Effort Cost' if i == 0 else '', fontsize=12)
        axes[i].set_xlabel('Reward Strategy', fontsize=12)
    
    # Add overall title
    fig.suptitle('Quality Type Distribution by Reward Strategy and Effort Cost', fontsize=16, y=1.05)
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_path = os.path.join(skill_dir, "quality_types_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved quality types heatmap to {output_path}")

def plot_participation_breakdown(data_path, output_base_dir):
    """
    Create plots specifically focused on participation breakdown including no contribution rates.
    
    Args:
        data_path (str): Path to the CSV file containing all experiment results
        output_base_dir (str): Base directory to save the plots
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Create output directory
    participation_dir = os.path.join(output_base_dir, "participation_breakdown")
    os.makedirs(participation_dir, exist_ok=True)
    
    # Create plots for each effort cost level
    effort_levels = df['mean_effort_cost'].unique()
    reward_strategies = df['reward_strategy'].unique()
    
    # Plot 1: Area chart showing participation breakdown across effort costs for each strategy
    for strategy in reward_strategies:
        plt.figure(figsize=(10, 6))
        
        # Filter data for this strategy
        strategy_data = df[df['reward_strategy'] == strategy].sort_values('mean_effort_cost')
        
        # Extract data for stacked area chart
        x = strategy_data['mean_effort_cost']
        y1 = strategy_data['high_quality_rate']
        y2 = strategy_data['low_quality_rate']
        y3 = strategy_data['no_contribution_rate']
        
        # Create stacked area chart
        plt.stackplot(x, y1, y2, y3, 
                     labels=['High Quality', 'Low Quality', 'No Contribution'],
                     colors=['#2ca02c', '#ff7f0e', '#d62728'],
                     alpha=0.8)
        
        # Add data points and values
        for i, effort_cost in enumerate(x):
            # High quality (bottom of stack)
            plt.scatter(effort_cost, y1[i]/2, color='white', s=30, zorder=5)
            plt.text(effort_cost, y1[i]/2, f"{y1[i]:.2f}", ha='center', va='center', fontsize=9, 
                    fontweight='bold', color='black', zorder=6)
            
            # Low quality (middle of stack)
            mid_low = y1[i] + y2[i]/2
            plt.scatter(effort_cost, mid_low, color='white', s=30, zorder=5)
            plt.text(effort_cost, mid_low, f"{y2[i]:.2f}", ha='center', va='center', fontsize=9, 
                    fontweight='bold', color='black', zorder=6)
            
            # No contribution (top of stack)
            mid_no = y1[i] + y2[i] + y3[i]/2
            plt.scatter(effort_cost, mid_no, color='white', s=30, zorder=5)
            plt.text(effort_cost, mid_no, f"{y3[i]:.2f}", ha='center', va='center', fontsize=9, 
                    fontweight='bold', color='black', zorder=6)
        
        # Customize the plot
        plt.title(f'Participation Breakdown by Effort Cost for {strategy} Strategy', fontsize=14)
        plt.xlabel('Mean Effort Cost', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.ylim(0, 1.05)
        
        # Save the plot
        output_path = os.path.join(participation_dir, f"{strategy}_participation_breakdown.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved participation breakdown for {strategy} strategy to {output_path}")
    
    # Plot 2: All strategies in one plot - line chart with no contribution rate
    plt.figure(figsize=(12, 8))
    
    # Plot lines for each strategy
    for strategy in reward_strategies:
        strategy_data = df[df['reward_strategy'] == strategy].sort_values('mean_effort_cost')
        plt.plot(strategy_data['mean_effort_cost'], 
                strategy_data['no_contribution_rate'], 
                marker='o', linewidth=2, markersize=8, label=strategy)
    
    # Customize the plot
    plt.title('No Contribution Rate by Reward Strategy and Effort Cost', fontsize=16)
    plt.xlabel('Mean Effort Cost', fontsize=14)
    plt.ylabel('No Contribution Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Reward Strategy', fontsize=12, title_fontsize=13)
    
    # Save the plot
    output_path = os.path.join(participation_dir, "no_contribution_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved no contribution comparison to {output_path}")
    
    # Plot 3: Effort cost impact on participation (all strategies)
    plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Plot data on each subplot
    for i, effort_cost in enumerate(effort_levels):
        ax = axes[i]
        df_effort = df[df['mean_effort_cost'] == effort_cost]
        
        # Set width and positions of bars
        bar_width = 0.25
        index = np.arange(len(reward_strategies))
        
        # Get data for plotting
        high_quality = [df_effort[df_effort['reward_strategy'] == s]['high_quality_rate'].values[0] for s in reward_strategies]
        low_quality = [df_effort[df_effort['reward_strategy'] == s]['low_quality_rate'].values[0] for s in reward_strategies]
        no_contribution = [df_effort[df_effort['reward_strategy'] == s]['no_contribution_rate'].values[0] for s in reward_strategies]
        
        # Create grouped bars
        ax.bar(index - bar_width, high_quality, bar_width, label='High Quality', color='#2ca02c')
        ax.bar(index, low_quality, bar_width, label='Low Quality', color='#ff7f0e')
        ax.bar(index + bar_width, no_contribution, bar_width, label='No Contribution', color='#d62728')
        
        # Add value labels
        for j in range(len(reward_strategies)):
            ax.text(j - bar_width, high_quality[j] + 0.02, f"{high_quality[j]:.2f}", ha='center', fontsize=9)
            ax.text(j, low_quality[j] + 0.02, f"{low_quality[j]:.2f}", ha='center', fontsize=9)
            ax.text(j + bar_width, no_contribution[j] + 0.02, f"{no_contribution[j]:.2f}", ha='center', fontsize=9)
        
        # Customize the subplot
        ax.set_title(f'Effort Cost = {effort_cost}', fontsize=14)
        ax.set_xlabel('Reward Strategy', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(reward_strategies)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend to the first subplot only
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add overall title
    fig.suptitle('Participation Breakdown by Effort Cost and Reward Strategy', fontsize=16)
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    output_path = os.path.join(participation_dir, "effort_cost_impact_grid.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved effort cost impact grid to {output_path}")

def main():
    # Input file containing all experiment results
    data_path = "results/comparison/experiment_all_results.csv"
    
    # Base output directory for plots
    output_base_dir = "results/strategy_comparison"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create skill type comparison plots
    plot_skill_type_comparisons(data_path, output_base_dir)
    
    # Create participation breakdown plots including no contribution rate
    plot_participation_breakdown(data_path, output_base_dir)
    
    logger.info(f"All skill type and participation breakdown plots saved to {output_base_dir}")

if __name__ == "__main__":
    main() 