import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging

from model import IncentiveModel

# Configure logging with more control over verbosity
# Set root logger to WARNING to silence most third-party logs
logging.getLogger().setLevel(logging.WARNING)

# Create a custom logger for our simulation
logger = logging.getLogger("simulation")
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Set third-party loggers to WARNING or higher
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("seaborn").setLevel(logging.WARNING)
logging.getLogger("pandas").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)

# Log divider function for better readability in console output
def log_divider(message="", char="=", length=80):
    """Print a divider in the log with an optional message."""
    if message:
        # Calculate padding to center the message
        padding = (length - len(message) - 2) // 2
        divider = char * padding + " " + message + " " + char * padding
        # Adjust if odd length
        if len(divider) < length:
            divider += char
    else:
        divider = char * length
    logger.info(divider)

class SimulationRunner:
    """Class to run experiments with different parameter combinations."""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized SimulationRunner with results directory: {self.results_dir}")
        
    def run_parameter_sweep(self, 
                          mean_effort_costs,
                          num_steps,
                          num_trials,
                          num_contributors,
                          effort_cost_std,
                          reward_strategy,
                          extrinsic_ratio,
                          balanced_ratio,
                          intrinsic_ratio):
        """Run simulations across parameter combinations."""
        log_divider("STARTING PARAMETER SWEEP")
        results = []
        # Create a dictionary to store time series data for selected parameter combinations
        time_series_data = {}
        
        param_combinations = list(mean_effort_costs)
        
        total_combinations = len(param_combinations) * num_trials
        logger.info(f"Starting parameter sweep with {len(param_combinations)} parameter combinations and {num_trials} trials each ({total_combinations} total simulations)")
        logger.info(f"Parameter ranges: mean_effort_costs={mean_effort_costs}")
        logger.info(f"Agent distribution: extrinsic={extrinsic_ratio}, balanced={balanced_ratio}, intrinsic={intrinsic_ratio}")
        logger.info(f"Reward strategy: {reward_strategy}")
        
        start_time = time.time()
        completed = 0
        
        for mean_effort_cost in param_combinations:
            log_divider(f"EFFORT COST: {mean_effort_cost:.2f}", char="-")
            for trial in range(num_trials):
                trial_start_time = time.time()
                logger.info(f"Running simulation with mean_effort_cost={mean_effort_cost:.2f}, trial={trial+1}/{num_trials}")
                
                model = IncentiveModel(
                    num_contributors=num_contributors,
                    mean_effort_cost=mean_effort_cost,
                    effort_cost_std=effort_cost_std,
                    extrinsic_ratio=extrinsic_ratio,
                    balanced_ratio=balanced_ratio,
                    intrinsic_ratio=intrinsic_ratio,
                    reward_strategy=reward_strategy
                )
                
                logger.info(f"Created model with {len(model.agents)} agents and reward strategy: {model.platform.get_reward_strategy_info()['name']}")

                # Run the simulation steps
                for step in range(num_steps):
                    model.step()
                    # if step % 10 == 0:
                    #     model_data = model.datacollector.get_model_vars_dataframe()
                    #     latest_metrics = model_data.iloc[-1]
                    #     logger.info(f"  Step {step+1}/{num_steps}: Participation={latest_metrics['Participation Rate']:.2f}, Quality={latest_metrics['Average Quality']:.2f}, Utility={latest_metrics['Platform Utility']:.2f}")
                
                # Collect final metrics
                model_data = model.datacollector.get_model_vars_dataframe()
                final_metrics = model_data.iloc[-1]
                
                logger.info(f"  Final metrics: Participation={final_metrics['Participation Rate']:.2f}, Quality={final_metrics['Average Quality']:.2f}, Utility={final_metrics['Platform Utility']:.2f}")
                
                # Calculate stability metrics (variation over time)
                participation_stability = 1.0 - model_data['Participation Rate'].std()
                quality_stability = 1.0 - model_data['Average Quality'].std()
                
                # Calculate trajectory metrics (improvement over time)
                if len(model_data) > 10:
                    early_participation = model_data['Participation Rate'][5:15].mean()
                    late_participation = model_data['Participation Rate'][-10:].mean()
                    participation_growth = (late_participation - early_participation) / (early_participation + 0.01)
                    
                    early_quality = model_data['Average Quality'][5:15].mean()
                    late_quality = model_data['Average Quality'][-10:].mean()
                    quality_growth = (late_quality - early_quality) / (early_quality + 0.01)
                    
                    logger.info(f"  Growth metrics: Participation growth={participation_growth:.2f}, Quality growth={quality_growth:.2f}")
                else:
                    participation_growth = 0
                    quality_growth = 0
                
                # Log participation by contributor type
                logger.info(f"  Participation by type: Extrinsic={final_metrics['Extrinsic Participation']:.2f}, Balanced={final_metrics['Balanced Participation']:.2f}, Intrinsic={final_metrics['Intrinsic Participation']:.2f}")
                
                results.append({
                    "mean_effort_cost": mean_effort_cost,
                    "reward_strategy": str(reward_strategy),
                    "trial": trial,
                    "avg_quality": final_metrics["Average Quality"],
                    "participation_rate": final_metrics["Participation Rate"],
                    "high_quality_rate": final_metrics["High Quality Rate"],
                    "med_quality_rate": final_metrics["Med Quality Rate"],
                    "low_quality_rate": final_metrics["Low Quality Rate"],
                    "no_contribution_rate": final_metrics["No Contribution Rate"],
                    "extrinsic_participation": final_metrics["Extrinsic Participation"],
                    "balanced_participation": final_metrics["Balanced Participation"],
                    "intrinsic_participation": final_metrics["Intrinsic Participation"],
                    "reputation_inequality": final_metrics["Reputation Inequality (Gini)"],
                    "platform_utility": final_metrics["Platform Utility"],
                    "participation_stability": participation_stability,
                    "quality_stability": quality_stability,
                    "participation_growth": participation_growth,
                    "quality_growth": quality_growth
                })
                
                # Store time series data for first trial of each parameter combination
                # We only need to store representative cases to save memory
                if trial == 0:
                    key = (mean_effort_cost, reward_strategy)
                    time_series_data[key] = model_data
                    logger.info(f"  Saved time series data for visualization (effort={mean_effort_cost:.2f})")
                
                trial_time = time.time() - trial_start_time
                logger.info(f"  Trial completed in {trial_time:.2f} seconds")
                
                completed += 1
                elapsed_time = time.time() - start_time
                avg_time_per_sim = elapsed_time / completed
                remaining_sims = total_combinations - completed
                est_remaining_time = avg_time_per_sim * remaining_sims
                
                logger.info(f"Completed {completed}/{total_combinations} simulations ({completed/total_combinations*100:.1f}%). Est. remaining time: {est_remaining_time/60:.1f} minutes")
        
        log_divider("PARAMETER SWEEP COMPLETED")
        logger.info(f"Parameter sweep completed in {(time.time() - start_time)/60:.2f} minutes")
        return pd.DataFrame(results), time_series_data
    
    def run_experiment(self, output_prefix="experiment"):
        """Run a complete experiment and save results."""
        log_divider("STARTING EXPERIMENT")
        logger.info("Starting experiment run")
        start_time = time.time()
        
        # Log system configuration
        logger.info(f"Running experiment with output prefix: {output_prefix}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Define reward strategies to test
        reward_strategies = ['weighted', 'linear', 'threshold'] #TODO: add rep_focus here
        all_results = []
        all_time_series = {}  # Dictionary to store time series data
        
        for strategy in reward_strategies:
            log_divider(f"REWARD STRATEGY: {strategy.upper()}")
            logger.info(f"Running parameter sweep with reward strategy: {strategy}")
            
            results, time_series_data = self.run_parameter_sweep(
                mean_effort_costs=np.linspace(0.5, 2.0, 3),
                num_steps=50,
                num_trials=1,
                num_contributors=500,
                effort_cost_std=0.5,
                reward_strategy=strategy,
                extrinsic_ratio=0.3,
                balanced_ratio=0.5,
                intrinsic_ratio=0.2
            )
            
            logger.info(f"Parameter sweep for {strategy} completed, shape of results DataFrame: {results.shape}")
            
            # Save strategy-specific results
            strategy_output_path = self.results_dir / f"{output_prefix}_{strategy}_results.csv"
            results.to_csv(strategy_output_path, index=False)
            logger.info(f"Saved {strategy} results to {strategy_output_path}")
            
            # Store time series data
            all_time_series.update(time_series_data)
            
            # Create visualizations for this strategy using the collected time series data
            log_divider(f"CREATING VISUALIZATIONS: {strategy}", char="-")
            logger.info(f"Creating visualizations for {strategy} reward strategy")
            self.plot_results(results, f"{output_prefix}_{strategy}")
            self.plot_quality_distribution(results, f"{output_prefix}_{strategy}")
            self.plot_time_series(results, f"{output_prefix}_{strategy}", time_series_data=time_series_data)
            
            # Add to combined results
            all_results.append(results)
        
        # Combine all results
        log_divider("COMBINING RESULTS")
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_output_path = self.results_dir / f"{output_prefix}_all_results.csv"
        combined_results.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined results to {combined_output_path}")
        
        # Run comparison across strategies
        log_divider("CROSS-STRATEGY COMPARISON")
        logger.info("Running cross-strategy comparison")
        self.compare_strategies(combined_results, output_prefix)
        
        total_time = time.time() - start_time
        log_divider("EXPERIMENT COMPLETED")
        logger.info(f"Experiment completed in {total_time/60:.2f} minutes")
        
        return combined_results, all_time_series
    
    def compare_strategies(self, results, output_prefix):
        """Compare different reward strategies."""
        logger.info("Comparing reward strategies")
        
        # Ensure reward_strategy is treated as a string category
        if 'reward_strategy' in results.columns:
            results['reward_strategy'] = results['reward_strategy'].astype(str)
        
        # Get the number of unique strategies
        unique_strategies = results['reward_strategy'].unique()
        num_strategies = len(unique_strategies)
        
        logger.info(f"Found {num_strategies} different strategies to compare: {unique_strategies}")
        
        # Group by strategy and calculate means, with explicit handling for each column
        strategy_comparison = results.groupby('reward_strategy').agg({
            'avg_quality': 'mean',
            'participation_rate': 'mean',
            'high_quality_rate': 'mean',
            'platform_utility': 'mean',
            'extrinsic_participation': 'mean',
            'balanced_participation': 'mean',
            'intrinsic_participation': 'mean',
            'participation_stability': 'mean',
            'quality_stability': 'mean'
        }).reset_index()
        
        logger.info(f"Strategy comparison summary:\n{strategy_comparison}")
        
        # Save comparison to CSV
        comparison_path = self.results_dir / f"{output_prefix}_compare_strategies.csv"
        strategy_comparison.to_csv(comparison_path, index=False)
        logger.info(f"Saved strategy comparison to {comparison_path}")
        
        # Create comparison visualizations
        metrics = ['avg_quality', 'participation_rate', 'high_quality_rate', 
                  'platform_utility', 'participation_stability', 'quality_stability']
        
        # Create a combined metrics comparison plot
        plt.figure(figsize=(15, 10))
        
        # Set up the bar positions
        x = np.arange(len(metrics))
        width = 0.8 / num_strategies
        
        # Plot bars for each strategy
        for i, strategy in enumerate(unique_strategies):
            strategy_data = [strategy_comparison[strategy_comparison['reward_strategy'] == strategy][metric].iloc[0] 
                            for metric in metrics]
            position = x + (i - num_strategies/2 + 0.5) * width
            plt.bar(position, strategy_data, width, label=strategy.title())
            
            # Add value labels on top of bars
            for pos, value in zip(position, strategy_data):
                plt.text(pos, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Strategy Comparison Across All Metrics')
        plt.xticks(x, [metric.replace('_', ' ').title() for metric in metrics], rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = self.results_dir / f"{output_prefix}_compare_metrics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics comparison to {output_path}")
        
        # Create participation by contributor type comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        strategies = strategy_comparison['reward_strategy']
        x = np.arange(len(strategies))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bars for each contributor type
        ax.bar(x - width, strategy_comparison['extrinsic_participation'], width, label='Extrinsic')
        ax.bar(x, strategy_comparison['balanced_participation'], width, label='Balanced')
        ax.bar(x + width, strategy_comparison['intrinsic_participation'], width, label='Intrinsic')
        
        # Add labels and legend
        ax.set_ylabel('Participation Rate')
        ax.set_title('Participation by Contributor Type Across Reward Strategies')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(strategy_comparison['extrinsic_participation']):
            ax.text(i - width, v + 0.01, f"{v:.2f}", ha='center')
        for i, v in enumerate(strategy_comparison['balanced_participation']):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        for i, v in enumerate(strategy_comparison['intrinsic_participation']):
            ax.text(i + width, v + 0.01, f"{v:.2f}", ha='center')
        
        output_path = self.results_dir / f"{output_prefix}_compare_contributor_types.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved contributor type comparison to {output_path}")
        
        logger.info("Strategy comparison completed")
    
    def plot_results(self, results_df, output_prefix="experiment"):
        """Process results for a single strategy."""
        log_divider("PLOTTING RESULTS", char="-")
        # Extract only numeric columns for mean calculations
        numeric_df = results_df.select_dtypes(include=['number'])
        
        # Make sure we keep the effort cost column
        if "mean_effort_cost" not in numeric_df.columns and "mean_effort_cost" in results_df.columns:
            numeric_df["mean_effort_cost"] = results_df["mean_effort_cost"]
        
        # Calculate means across trials for numeric columns only
        mean_results = numeric_df.groupby(["mean_effort_cost"]).mean().reset_index()
        
        # Create line plots for different metrics
        metrics = ["avg_quality",
                   "participation_rate",
                   "high_quality_rate",
                   "med_quality_rate",
                   "low_quality_rate",
                   "no_contribution_rate",
                   "platform_utility", 
                   "reputation_inequality"]
        
        # Create a multi-metric line plot
        plt.figure(figsize=(14, 10))
        
        # Plot metrics that use the same y-axis scale (0-1)
        percent_metrics = ["participation_rate", "high_quality_rate", "med_quality_rate", 
                          "low_quality_rate", "no_contribution_rate"]
        
        for metric in percent_metrics:
            plt.plot(mean_results["mean_effort_cost"], mean_results[metric], 
                     marker='o', linewidth=2, label=metric.replace('_', ' ').title())
        
        plt.xlabel("Mean Effort Cost (γ)")
        plt.ylabel("Rate")
        plt.title("Participation and Quality Rates by Effort Cost")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        output_path = self.results_dir / f"{output_prefix}_participation_rates.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved participation rates plot to {output_path}")
        
        # Create a separate plot for average quality
        plt.figure(figsize=(12, 6))
        plt.plot(mean_results["mean_effort_cost"], mean_results["avg_quality"], 
                 marker='o', linewidth=3, color='blue')
        
        plt.xlabel("Mean Effort Cost (γ)")
        plt.ylabel("Average Quality (0-3 scale)")
        plt.title("Average Contribution Quality by Effort Cost")
        plt.grid(True, alpha=0.3)
        
        # Add data point annotations
        for i, row in mean_results.iterrows():
            plt.annotate(f"{row['avg_quality']:.2f}", 
                        (row["mean_effort_cost"], row["avg_quality"]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        output_path = self.results_dir / f"{output_prefix}_average_quality.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved average quality plot to {output_path}")
        
        # Create a plot for platform utility
        plt.figure(figsize=(12, 6))
        plt.plot(mean_results["mean_effort_cost"], mean_results["platform_utility"], 
                 marker='o', linewidth=3, color='green')
        
        plt.xlabel("Mean Effort Cost (γ)")
        plt.ylabel("Platform Utility")
        plt.title("Platform Utility by Effort Cost")
        plt.grid(True, alpha=0.3)
        
        # Add data point annotations
        for i, row in mean_results.iterrows():
            plt.annotate(f"{row['platform_utility']:.2f}", 
                        (row["mean_effort_cost"], row["platform_utility"]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        output_path = self.results_dir / f"{output_prefix}_platform_utility.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved platform utility plot to {output_path}")
        
        # Create a multi-line plot for participation by contributor type
        if all(col in mean_results.columns for col in ["extrinsic_participation", "balanced_participation", "intrinsic_participation"]):
            plt.figure(figsize=(12, 6))
            
            plt.plot(mean_results["mean_effort_cost"], mean_results["extrinsic_participation"], 
                    marker='o', linewidth=2, label="Extrinsic")
            plt.plot(mean_results["mean_effort_cost"], mean_results["balanced_participation"], 
                    marker='s', linewidth=2, label="Balanced")
            plt.plot(mean_results["mean_effort_cost"], mean_results["intrinsic_participation"], 
                    marker='^', linewidth=2, label="Intrinsic")
            
            plt.xlabel("Mean Effort Cost (γ)")
            plt.ylabel("Participation Rate")
            plt.title("Participation by Contributor Type and Effort Cost")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
            
            output_path = self.results_dir / f"{output_prefix}_participation_by_type.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved participation by type plot to {output_path}")

    def plot_time_series(self, results_df, output_prefix, time_series_data):
        """Plot time series data for selected parameter combinations."""
        log_divider("PLOTTING TIME SERIES", char="-")
        # Get numeric data for means calculation
        numeric_df = results_df.select_dtypes(include=['number'])
        
        # Make sure we keep the effort cost column
        if "mean_effort_cost" not in numeric_df.columns and "mean_effort_cost" in results_df.columns:
            numeric_df["mean_effort_cost"] = results_df["mean_effort_cost"]
        
        # Choose representative parameter combinations
        selected_params = numeric_df.groupby(["mean_effort_cost"]).mean().reset_index()        
        logger.info(f"Creating time series plots for {len(selected_params)} parameter combinations")
        
        # Extract reward strategy from dataframe if present
        reward_strategy = 'weighted'  # Default
        if 'reward_strategy' in results_df.columns:
            # Use most common strategy if multiple exist
            reward_strategy = results_df['reward_strategy'].mode().iloc[0]
            logger.info(f"Using reward strategy: {reward_strategy} for time series plots")
        
        for idx, params in selected_params.iterrows():
            effort = params["mean_effort_cost"]
            
            logger.info(f"Generating time series for params: effort={effort:.2f}")
            
            # Get the model data from stored time series data if available
            key = (effort, reward_strategy)
            if time_series_data is not None and key in time_series_data:
                model_data = time_series_data[key]
                logger.info(f"Using stored time series data for parameters: effort={effort:.2f}")
            else:
                # Fall back to running a new simulation if data not available
                logger.warning(f"No stored time series data found for parameters: effort={effort:.2f}")
                logger.warning(f"Running a new simulation to generate time series data (less efficient)")
                model_data = self.get_model_data_for_params(effort, reward_strategy)
            
            if model_data is None:
                logger.warning(f"Failed to get model data for parameter combination: effort={effort:.2f}")
                continue
                
            logger.info(f"Time series data shape: {model_data.shape}")
            # logger.info(f"Time series columns: {model_data.columns}")
            
            # Plot participation rates over time
            plt.figure(figsize=(12, 8))
            plt.plot(model_data.index, model_data['Participation Rate'], label='Overall Participation', linewidth=2)
            plt.plot(model_data.index, model_data['Extrinsic Participation'], label='Extrinsic Types', alpha=0.7)
            plt.plot(model_data.index, model_data['Balanced Participation'], label='Balanced Types', alpha=0.7)
            plt.plot(model_data.index, model_data['Intrinsic Participation'], label='Intrinsic Types', alpha=0.7)
            
            # Update title to focus on effort cost only
            plt.title(f"Participation Rates by Contributor Type\nEffort Cost (γ)={effort:.2f}")
            plt.xlabel("Time Steps")
            plt.ylabel("Participation Rate")
            plt.ylim(0, 1.05)  # Make y-axis consistent and leave room for legend
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Add annotations for final values
            last_step = model_data.index[-1]
            for col, style in zip(['Participation Rate', 'Extrinsic Participation', 'Balanced Participation', 'Intrinsic Participation'], 
                                 ['k', 'b', 'g', 'r']):
                plt.annotate(f"{model_data[col].iloc[-1]:.2f}", 
                            xy=(last_step, model_data[col].iloc[-1]),
                            xytext=(5, 0), textcoords='offset points',
                            color=style, fontweight='bold')
            
            output_path = self.results_dir / f"{output_prefix}_time_participation_e{effort:.2f}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved participation time series to {output_path}")
            
            # NEW IMPROVED GRAPH: All quality rates over time
            quality_cols = ['High Quality Rate', 'Med Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            if all(col in model_data.columns for col in quality_cols):
                plt.figure(figsize=(12, 8))
                colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Green, Blue, Orange, Red
                
                for col, color in zip(quality_cols, colors):
                    plt.plot(model_data.index, model_data[col], color=color, 
                            label=col.replace('_', ' '), linewidth=2)
                
                plt.title(f"Contribution Quality Breakdown Over Time (γ={effort:.2f})")
                plt.xlabel("Time Steps")
                plt.ylabel("Proportion of Contributors")
                plt.ylim(0, 1.05)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                # Add annotations for final values
                last_step = model_data.index[-1]
                for col, color in zip(quality_cols, colors):
                    plt.annotate(f"{model_data[col].iloc[-1]:.2f}", 
                                xy=(last_step, model_data[col].iloc[-1]),
                                xytext=(5, 0), textcoords='offset points',
                                color=color, fontweight='bold')
                
                output_path = self.results_dir / f"{output_prefix}_all_quality_rates_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved all quality rates time series to {output_path}")
            
            # Plot quality and platform metrics over time
            plt.figure(figsize=(12, 8))
            
            # Create a subplot layout for better visualization
            gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)
            
            # Top subplot for quality metrics
            ax1 = plt.subplot(gs[0])
            
            # Plot all quality metrics with different line styles
            quality_cols = ['High Quality Rate', 'Med Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Green, Blue, Orange, Red
            
            for col, color in zip(quality_cols, colors):
                if col in model_data.columns:
                    ax1.plot(model_data.index, model_data[col], color=color, 
                            label=col.replace(' Rate', ''), linewidth=2)
            
            ax1.plot(model_data.index, model_data['Average Quality'], 'k-', 
                    label='Avg Quality (0-3 scale)', linewidth=2)
            
            ax1.set_ylabel('Quality Metrics (Proportion)', fontweight='bold')
            ax1.set_title(f"Quality & Platform Performance (γ={effort:.2f})")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.set_ylim(bottom=0)
            
            # Create a secondary y-axis for the average quality on 0-3 scale
            ax1_2 = ax1.twinx()
            ax1_2.set_ylabel('Avg Quality (0-3 scale)', color='k')
            ax1_2.set_ylim(0, 3.2)  # Set max slightly above 3 to allow space
            
            # Bottom subplot for platform utility - focused on per-step metrics
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(model_data.index, model_data['Platform Utility'], 'r-', label='Platform Utility Per Step', linewidth=2)
            
            if 'Platform Step Benefit' in model_data.columns and 'Platform Step Cost' in model_data.columns:
                # Fill the area between benefit and cost
                ax2.fill_between(model_data.index, model_data['Platform Step Benefit'], model_data['Platform Step Cost'], 
                                alpha=0.3, color='g', label='Net Benefit Area')
            
            ax2.set_ylabel('Utility Per Step', color='r', fontweight='bold')
            ax2.set_xlabel('Time Steps', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
            # Remove top subplot's x-tick labels
            plt.setp(ax1.get_xticklabels(), visible=False)
            
            # Adjust layout
            # Use tight_layout with appropriate padding to avoid warning
            plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=1.0)
            
            output_path = self.results_dir / f"{output_prefix}_platform_metrics_e{effort:.2f}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            logger.info(f"Saved platform metrics to {output_path}")
            
            # NEW GRAPH: Contribution breakdown - Area chart
            plt.figure(figsize=(12, 8))
            quality_cols = ['High Quality Rate', 'Med Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            if all(col in model_data.columns for col in quality_cols):
                plt.stackplot(model_data.index, 
                              [model_data[col] for col in quality_cols],
                              labels=[col.replace(' Rate', '') for col in quality_cols],
                              colors=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'],  # Green, Blue, Orange, Red 
                              alpha=0.7)
                
                plt.title(f"Contribution Breakdown by Quality (γ={effort:.2f})")
                plt.xlabel("Time Steps")
                plt.ylabel("Proportion of Contributors")
                plt.ylim(0, 1.05)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(quality_cols))
                plt.grid(True, alpha=0.3)
                
                output_path = self.results_dir / f"{output_prefix}_contribution_breakdown_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved contribution breakdown to {output_path}")
            
            # NEW GRAPH: Platform economics - costs, benefits and utility per step
            if 'Platform Step Cost' in model_data.columns and 'Platform Step Benefit' in model_data.columns:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                # Plot benefit as a line
                ax.plot(model_data.index, model_data['Platform Step Benefit'], 'g-', label='Benefit', linewidth=2)
                
                # Plot cost as a line
                ax.plot(model_data.index, model_data['Platform Step Cost'], 'r-', label='Cost', linewidth=2)
                
                # Fill the area between for net benefit visualization
                ax.fill_between(model_data.index, model_data['Platform Step Benefit'], model_data['Platform Step Cost'], 
                               where=(model_data['Platform Step Benefit'] >= model_data['Platform Step Cost']),
                               color='green', alpha=0.3, label='Positive Utility')
                
                ax.fill_between(model_data.index, model_data['Platform Step Benefit'], model_data['Platform Step Cost'], 
                               where=(model_data['Platform Step Benefit'] < model_data['Platform Step Cost']),
                               color='red', alpha=0.3, label='Negative Utility')
                
                # Plot net utility line
                ax2 = ax.twinx()
                ax2.plot(model_data.index, model_data['Platform Utility'], 'b-', label='Net Utility', linewidth=2)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Cost / Benefit', fontweight='bold')
                ax2.set_ylabel('Net Utility', color='blue', fontweight='bold')
                
                # Set axis limits with some padding
                max_val = max(
                    model_data['Platform Step Benefit'].max(), 
                    model_data['Platform Step Cost'].max()
                ) * 1.1
                
                ax.set_ylim(0, max_val)  # Always start at 0 for costs/benefits
                
                util_range = model_data['Platform Utility'].max() - model_data['Platform Utility'].min()
                util_min = model_data['Platform Utility'].min() - (util_range * 0.1)
                util_max = model_data['Platform Utility'].max() + (util_range * 0.1)
                
                # Ensure there's always some range
                if util_range < 0.1:  
                    util_min = model_data['Platform Utility'].min() - 0.1
                    util_max = model_data['Platform Utility'].max() + 0.1
                    
                ax2.set_ylim(util_min, util_max)
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
                
                plt.title(f"Platform Economics Per Step (γ={effort:.2f})")
                plt.grid(True, alpha=0.3)
                
                output_path = self.results_dir / f"{output_prefix}_platform_economics_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved platform economics to {output_path}")

    def get_model_data_for_params(self, effort, reward_strategy='weighted'):
        """Run a single simulation and return time-series data. This is a fallback method."""
        logger.info(f"Running simulation to get time-series data for effort={effort:.2f}, strategy={reward_strategy}")
        logger.warning("Note: This is running an extra simulation. For efficiency, time series should be stored during parameter sweep.")
        
        model = IncentiveModel(
            num_contributors=500,  # Larger population for better statistics
            mean_effort_cost=effort,
            effort_cost_std=0.5,
            reward_strategy=reward_strategy
        )
        
        logger.info(f"Created time-series model with {len(model.agents)} agents using {model.platform.get_reward_strategy_info()['name']} strategy")
        
        for step in range(100):  # Run for 100 steps
            model.step()
            if step % 20 == 0:
                model_data = model.datacollector.get_model_vars_dataframe()
                latest_metrics = model_data.iloc[-1]
                logger.info(f"  Time-series step {step+1}/100: Participation={latest_metrics['Participation Rate']:.2f}, Quality={latest_metrics['Average Quality']:.2f}")
            
        return model.datacollector.get_model_vars_dataframe()

    def run_sensitivity_analysis(self, results_df, output_prefix):
        """Perform sensitivity analysis on key parameters."""
        log_divider("STARTING SENSITIVITY ANALYSIS", char="-")
        
        params = ["mean_effort_cost"]
        metrics = ["avg_quality", "participation_rate", "platform_utility", 
                  "extrinsic_participation", "balanced_participation", "intrinsic_participation"]
        
        for param in params:
            logger.info(f"Analyzing sensitivity to {param}")
            plt.figure(figsize=(12, 8))
            for metric in metrics:
                # Group by parameter and calculate mean and std
                grouped = results_df.groupby(param)[metric].agg(['mean', 'std'])
                plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                            label=metric.replace('_', ' ').title(), linewidth=2, marker='o')
                
            plt.title(f"Sensitivity to {param.replace('_', ' ').title()}")
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel("Metric Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = self.results_dir / f"{output_prefix}_sensitivity_{param}.png"
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved sensitivity plot to {output_path}")
        
        logger.info("Sensitivity analysis completed")

    def plot_quality_distribution(self, results_df, output_prefix="experiment"):
        """Create visualizations showing the distribution of contribution quality."""
        logger.info("Creating quality distribution visualizations")
        
        # Get a copy of the dataframe with only numeric columns for mean calculations
        numeric_df = results_df.select_dtypes(include=['number'])
        
        # Make sure we keep the effort cost column if it might have been filtered out
        if "mean_effort_cost" not in numeric_df.columns and "mean_effort_cost" in results_df.columns:
            numeric_df["mean_effort_cost"] = results_df["mean_effort_cost"]
        
        # Calculate means across trials for numeric columns only, grouped only by effort cost
        mean_results = numeric_df.groupby(["mean_effort_cost"]).mean().reset_index()
        
        # Select a few representative parameter combinations
        selected_params = mean_results.iloc[:min(9, len(mean_results))]
        
        # Create a grid of pie charts showing quality distribution
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        quality_cols = ['high_quality_rate', 'med_quality_rate', 'low_quality_rate', 'no_contribution_rate']
        quality_labels = ['High Quality', 'Medium Quality', 'Low Quality', 'No Contribution']
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Green, Blue, Orange, Red
        
        for i, (_, param) in enumerate(selected_params.iterrows()):
            if i >= len(axes):
                break
            
            effort = param['mean_effort_cost']
            
            # Extract quality distribution
            quality_values = [param[col] for col in quality_cols]
            
            # Create pie chart
            wedges, texts, autotexts = axes[i].pie(
                quality_values, 
                labels=quality_labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Styling
            for autotext in autotexts:
                autotext.set_weight('bold')
                autotext.set_fontsize(9)
            
            for text in texts:
                text.set_fontsize(9)
            
            axes[i].set_title(f"γ={effort:.2f}\nAvg Quality: {param['avg_quality']:.2f}")
        
        # Hide any unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle("Contribution Quality Distribution by Effort Cost", 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = self.results_dir / f"{output_prefix}_quality_distribution_pies.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved quality distribution pie charts to {output_path}")
        
        # Create bar charts instead of heatmaps since we only have one parameter dimension now
        for metric in ['avg_quality', 'participation_rate', 'platform_utility']:
            plt.figure(figsize=(10, 6))
            
            # Plot bar chart of metric by effort cost
            plt.bar(mean_results['mean_effort_cost'], mean_results[metric], 
                    width=0.1, color='#1f77b4', alpha=0.8)
            
            # Add data point values
            for i, v in enumerate(mean_results[metric]):
                plt.text(mean_results['mean_effort_cost'].iloc[i], v + 0.02, 
                        f"{v:.2f}", ha='center')
            
            # Get strategy name if available
            strategy_name = ""
            if 'reward_strategy' in results_df.columns and results_df['reward_strategy'].nunique() == 1:
                strategy_name = f" - {results_df['reward_strategy'].iloc[0].title()} Strategy"
            
            plt.title(f'{metric.replace("_", " ").title()}{strategy_name}')
            plt.xlabel('Effort Cost (γ)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            output_path = self.results_dir / f"{output_prefix}_barchart_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved bar chart for {metric} to {output_path}")
        
        logger.info("Quality distribution visualization completed")


if __name__ == "__main__":
    logger.info("Starting simulation runner")
    runner = SimulationRunner()
    results, time_series = runner.run_experiment() 
    logger.info("Simulation complete")