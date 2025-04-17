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
        
    def _get_strategy_dir(self, strategy):
        """Create and return a strategy-specific subdirectory."""
        strategy_dir = self.results_dir / strategy
        strategy_dir.mkdir(exist_ok=True)
        return strategy_dir
        
    def run_parameter_sweep(self, 
                          mean_effort_costs,
                          num_steps,
                          num_trials,
                          num_contributors,
                          effort_cost_std,
                          reward_strategy,
                          extrinsic_ratio,
                          intrinsic_ratio,
                          low_skill_ratio,
                          phi,
                          delta,
                          lambda_param,
                          reward_learning_rate,
                          epsilon,
                          use_ewa):
        """Run simulations across parameter combinations."""
        log_divider("STARTING PARAMETER SWEEP")
        results = []
        # Create a dictionary to store time series data for selected parameter combinations
        time_series_data = {}
        
        param_combinations = list(mean_effort_costs)
        
        total_combinations = len(param_combinations) * num_trials
        logger.info(f"Starting parameter sweep with {len(param_combinations)} parameter combinations and {num_trials} trials each ({total_combinations} total simulations)")
        logger.info(f"Parameter ranges: mean_effort_costs={mean_effort_costs}")
        logger.info(f"Agent distribution: extrinsic={extrinsic_ratio}, intrinsic={intrinsic_ratio}")
        logger.info(f"Skill distribution: low_skill={low_skill_ratio}, high_skill={1-low_skill_ratio}")
        logger.info(f"Reward strategy: {reward_strategy}")
        logger.info(f"EWA parameters: phi={phi}, delta={delta}, lambda_param={lambda_param}")
        logger.info(f"Reward learning rate: {reward_learning_rate}")
        logger.info(f"ε-Nash equilibrium threshold: {epsilon}")
        logger.info(f"Using EWA learning: {use_ewa}")
        
        start_time = time.time()
        completed = 0
        
        for mean_effort_cost in param_combinations:
            log_divider(f"EFFORT COST: {mean_effort_cost:.2f}", char="-")
            for trial in range(num_trials):
                trial_start_time = time.time()
                logger.info(f"Running simulation with mean_effort_cost={mean_effort_cost:.2f}, trial={trial+1}/{num_trials}")
                
                model = IncentiveModel(
                    num_contributors=num_contributors,
                    reward_pool=num_contributors,
                    mean_effort_cost=mean_effort_cost,
                    effort_cost_std=effort_cost_std,
                    extrinsic_ratio=extrinsic_ratio,
                    intrinsic_ratio=intrinsic_ratio,
                    low_skill_ratio=low_skill_ratio,
                    reward_strategy=reward_strategy,
                    phi=phi,
                    delta=delta,
                    lambda_param=lambda_param,
                    reward_learning_rate=reward_learning_rate,
                    use_ewa=use_ewa
                )
                
                logger.info(f"Created model with {len(model.agents)} agents using {model.platform.get_reward_strategy_info()['name']} strategy")

                # Run the simulation steps
                epsilon_nash_reached = False
                nash_equilibrium_step = None
                min_utility_gain = float('inf')  # Track how close we get to epsilon-Nash
                
                # Create a list to track utility gain over time
                utility_gain_history = []
                
                # Track Nash equilibrium stability metrics
                nash_stability_window = 10  # Number of steps to track for stability after reaching Nash
                post_nash_participation_rates = []
                post_nash_quality_rates = []
                post_nash_platform_utility = []
                
                # Track time spent at or near Nash equilibrium
                steps_at_nash = 0
                steps_near_nash = 0  # Within 2x epsilon
                nash_breach_count = 0  # Times system moves out of Nash after reaching it
                
                # Track period-by-period Nash statistics for all time steps
                all_utility_gains = []  # One for each step checked
                all_steps_checked = []  # Steps when Nash condition was checked
                
                for step in range(num_steps):
                    model.step()
                    
                    # Check for ε-Nash equilibrium every 5 steps to reduce computational overhead
                    if step > 10 and step % 5 == 0:  # Start checking after step 10
                        is_nash, max_gain, agents_checked = model.is_epsilon_nash_equilibrium(epsilon=epsilon, sample_size=30)
                        
                        # Record for detailed analysis
                        all_utility_gains.append(max_gain)
                        all_steps_checked.append(step)
                        
                        # Store the utility gain for this step
                        utility_gain_history.append((step, max_gain))
                        
                        # Update the closest we've gotten to equilibrium
                        min_utility_gain = min(min_utility_gain, max_gain)
                        
                        # Track whether we're near Nash
                        if max_gain <= epsilon * 2:
                            steps_near_nash += 1
                        
                        # Track stability after reaching Nash
                        if epsilon_nash_reached:
                            # Count steps at Nash
                            if is_nash:
                                steps_at_nash += 1
                            else:
                                # System moved out of Nash equilibrium after reaching it
                                nash_breach_count += 1
                            
                            # Store post-Nash metrics for stability analysis
                            model_data = model.datacollector.get_model_vars_dataframe()
                            current_step_data = model_data.iloc[-1]
                            
                            post_nash_participation_rates.append(current_step_data['Participation Rate'])
                            post_nash_quality_rates.append(current_step_data['Average Quality'])
                            post_nash_platform_utility.append(current_step_data['Platform Utility'])
                        
                        # Log the current status
                        if step % 20 == 0 or is_nash:  # Log every 20 steps or if we reached equilibrium
                            logger.info(f"  Step {step}: Max utility gain = {max_gain:.4f} (ε = {epsilon}), agents checked: {agents_checked}")
                        
                        if is_nash and not epsilon_nash_reached:
                            logger.info(f"  ε-Nash steady state reached at step {step} with max utility gain {max_gain:.4f}")
                            epsilon_nash_reached = True
                            nash_equilibrium_step = step
                
                # Log how close we got if we didn't reach equilibrium
                if not epsilon_nash_reached:
                    logger.info(f"  Simulation completed without reaching ε-Nash equilibrium. Min utility gain: {min_utility_gain:.4f} (ε = {epsilon})")
                
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
                
                # Calculate post-Nash stability if applicable
                post_nash_participation_stability = 1.0
                post_nash_quality_stability = 1.0
                nash_state_persistence = 0
                
                if post_nash_participation_rates:
                    # Calculate stability after reaching Nash equilibrium
                    post_nash_participation_stability = 1.0 - np.std(post_nash_participation_rates)
                    post_nash_quality_stability = 1.0 - np.std(post_nash_quality_rates)
                    
                    # Calculate how much time was spent in Nash state after first reaching it
                    if nash_equilibrium_step is not None:
                        total_steps_after_nash = num_steps - nash_equilibrium_step
                        if total_steps_after_nash > 0:
                            nash_state_persistence = steps_at_nash / total_steps_after_nash
                
                # Log participation by contributor type
                logger.info(f"  Participation by type: Extrinsic={final_metrics['Extrinsic Participation']:.2f}, Intrinsic={final_metrics['Intrinsic Participation']:.2f}")
                
                results.append({
                    "mean_effort_cost": mean_effort_cost,
                    "reward_strategy": str(reward_strategy),
                    "trial": trial,
                    "avg_quality": final_metrics["Average Quality"],
                    "participation_rate": final_metrics["Participation Rate"],
                    "high_quality_rate": final_metrics["High Quality Rate"],
                    "low_quality_rate": final_metrics["Low Quality Rate"],
                    "no_contribution_rate": final_metrics["No Contribution Rate"],
                    "extrinsic_participation": final_metrics["Extrinsic Participation"],
                    "intrinsic_participation": final_metrics["Intrinsic Participation"],
                    "reputation_inequality": final_metrics["Reputation Inequality (Gini)"],
                    "platform_utility": final_metrics["Platform Utility"],
                    "participation_stability": participation_stability,
                    "quality_stability": quality_stability,
                    "participation_growth": participation_growth,
                    "quality_growth": quality_growth,
                    "epsilon_nash": epsilon_nash_reached or (final_metrics["Is Epsilon Nash"] if "Is Epsilon Nash" in final_metrics else False),
                    "nash_equilibrium_step": nash_equilibrium_step if epsilon_nash_reached else None,
                    "min_utility_gain": min_utility_gain,
                    "decision_method": final_metrics["Decision Method"] if "Decision Method" in final_metrics else ("EWA Learning" if use_ewa else "Direct Utility Max"),
                    # New Nash equilibrium metrics
                    "epsilon_value": epsilon,
                    "post_nash_participation_stability": post_nash_participation_stability,
                    "post_nash_quality_stability": post_nash_quality_stability,
                    "nash_state_persistence": nash_state_persistence,
                    "steps_at_nash": steps_at_nash,
                    "steps_near_nash": steps_near_nash,
                    "nash_breach_count": nash_breach_count
                })
                
                # Store time series data for first trial of each parameter combination
                # We only need to store representative cases to save memory
                if trial == 0:
                    key = (mean_effort_cost, reward_strategy)
                    time_series_data[key] = {
                        'model_data': model_data,
                        'utility_gain_history': utility_gain_history,
                        'nash_analysis': {
                            'all_utility_gains': all_utility_gains,
                            'all_steps_checked': all_steps_checked,
                            'epsilon_nash_reached': epsilon_nash_reached,
                            'nash_equilibrium_step': nash_equilibrium_step,
                            'post_nash_metrics': {
                                'participation_rates': post_nash_participation_rates,
                                'quality_rates': post_nash_quality_rates,
                                'platform_utility': post_nash_platform_utility
                            }
                        }
                    }
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
    
    def run_experiment(self, output_prefix="experiment", epsilon=0.2, use_ewa=True):
        """Run a complete experiment and save results."""
        log_divider("STARTING EXPERIMENT")
        logger.info("Starting experiment run")
        start_time = time.time()
        
        # Log system configuration
        logger.info(f"Running experiment with output prefix: {output_prefix}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        logger.info(f"ε-Nash equilibrium threshold: {epsilon}")
        logger.info(f"Using EWA learning: {use_ewa}")
        logger.info(f"Results will be organized as follows:")
        logger.info(f"  - {self.results_dir}/")
        logger.info(f"    - [strategy_name]/  (one directory per reward strategy)")
        logger.info(f"      - {output_prefix}_results.csv")
        logger.info(f"      - {output_prefix}_*.png (visualizations)")
        logger.info(f"    - comparison/  (cross-strategy comparison)")
        logger.info(f"      - {output_prefix}_all_results.csv")
        logger.info(f"      - {output_prefix}_compare_*.png (comparison visualizations)")
        logger.info(f"    - nash_analysis/  (detailed Nash equilibrium analysis)")
        logger.info(f"      - {output_prefix}_nash_*.png (Nash equilibrium visualizations)")
        logger.info(f"    - nash_convergence/  (detailed Nash convergence analysis)")
        logger.info(f"      - {output_prefix}_convergence_*.png (Nash convergence visualizations)")
        
        # Define reward strategies to test
        reward_strategies = ['weighted', 'linear', 'threshold', 'reputation_weighted']
        all_results = []
        all_time_series = {}  # Dictionary to store time series data
        
        # EWA parameters
        phi = 0.9                # Experience decay factor
        delta = 0.1              # Weight on foregone payoffs
        lambda_param = 5         # Softmax sharpness parameter
        reward_learning_rate = 0.2  # Learning rate for reward estimation
        
        for strategy in reward_strategies:
            log_divider(f"REWARD STRATEGY: {strategy.upper()}")
            logger.info(f"Running parameter sweep with reward strategy: {strategy}")
            
            results, time_series_data = self.run_parameter_sweep(
                mean_effort_costs=[0.5], # np.linspace(0.5, 2.0, 3),
                num_steps=100,
                num_trials=1,
                num_contributors=500,
                effort_cost_std=0.5,
                reward_strategy=strategy,
                extrinsic_ratio=0.5,
                intrinsic_ratio=0.5,
                low_skill_ratio=0.5,
                phi=phi,
                delta=delta,
                lambda_param=lambda_param,
                reward_learning_rate=reward_learning_rate,
                epsilon=epsilon,
                use_ewa=use_ewa
            )
            
            logger.info(f"Parameter sweep for {strategy} completed, shape of results DataFrame: {results.shape}")
            
            # Save strategy-specific results
            strategy_dir = self._get_strategy_dir(strategy)
            strategy_output_path = strategy_dir / f"{output_prefix}_results.csv"
            results.to_csv(strategy_output_path, index=False)
            logger.info(f"Saved {strategy} results to {strategy_output_path}")
            
            # Store time series data
            all_time_series.update(time_series_data)
            
            # Create visualizations for this strategy using the collected time series data
            log_divider(f"CREATING VISUALIZATIONS: {strategy}", char="-")
            logger.info(f"Creating visualizations for {strategy} reward strategy")
            self.plot_results(results, f"{output_prefix}", strategy_dir=strategy_dir)
            self.plot_quality_distribution(results, f"{output_prefix}", strategy_dir=strategy_dir)
            self.plot_time_series(results, f"{output_prefix}", time_series_data=time_series_data, strategy_dir=strategy_dir)
            self.plot_historical_rewards(time_series_data, f"{output_prefix}", strategy_dir=strategy_dir)
            
            # Add Nash equilibrium analysis for each strategy
            self.plot_nash_equilibrium_analysis(results, time_series_data, output_prefix, strategy_dir=strategy_dir)
            
            # Add to combined results
            all_results.append(results)
        
        # Combine all results
        log_divider("COMBINING RESULTS")
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Create a comparison directory
        comparison_dir = self.results_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        combined_output_path = comparison_dir / f"{output_prefix}_all_results.csv"
        combined_results.to_csv(combined_output_path, index=False)
        logger.info(f"Saved combined results to {combined_output_path}")
        
        # Run comparison across strategies
        log_divider("CROSS-STRATEGY COMPARISON")
        logger.info("Running cross-strategy comparison")
        self.compare_strategies(combined_results, output_prefix, comparison_dir)
        
        # Create a directory for detailed Nash equilibrium analysis across all strategies
        nash_dir = self.results_dir / "nash_analysis"
        nash_dir.mkdir(exist_ok=True)
        
        # Run combined Nash equilibrium analysis
        log_divider("CROSS-STRATEGY NASH EQUILIBRIUM ANALYSIS")
        logger.info("Running cross-strategy Nash equilibrium analysis")
        self.plot_nash_equilibrium_analysis(combined_results, all_time_series, output_prefix, strategy_dir=nash_dir)
        
        # Run detailed comparative analysis of Nash equilibrium between strategies
        self.compare_nash_equilibrium(combined_results, output_prefix, nash_dir)
        
        # Create a directory for Nash convergence analysis
        convergence_dir = self.results_dir / "nash_convergence"
        convergence_dir.mkdir(exist_ok=True)
        
        # Run detailed Nash convergence analysis
        log_divider("NASH CONVERGENCE ANALYSIS")
        logger.info("Running Nash convergence analysis")
        self.plot_nash_convergence_analysis(combined_results, all_time_series, output_prefix, convergence_dir)
        
        total_time = time.time() - start_time
        log_divider("EXPERIMENT COMPLETED")
        logger.info(f"Experiment completed in {total_time/60:.2f} minutes")
        
        return combined_results, all_time_series
    
    def compare_strategies(self, results, output_prefix, comparison_dir=None):
        """Compare different reward strategies."""
        logger.info("Comparing reward strategies")
        
        # Use comparison_dir if provided, otherwise fallback to results_dir
        output_dir = comparison_dir if comparison_dir is not None else self.results_dir
        
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
            'intrinsic_participation': 'mean',
            'participation_stability': 'mean',
            'quality_stability': 'mean',
            'epsilon_nash': lambda x: x.mean() if pd.api.types.is_bool_dtype(x) else np.nan,
            'nash_equilibrium_step': lambda x: x.mean() if all(pd.notna(v) for v in x) else np.nan,
            'min_utility_gain': 'mean'  # Average how close each strategy got to epsilon-Nash
        }).reset_index()
        
        logger.info(f"Strategy comparison summary:\n{strategy_comparison}")
        
        # Calculate percentage of runs that reached epsilon-Nash equilibrium
        if 'epsilon_nash' in results.columns:
            for strategy in unique_strategies:
                strategy_data = results[results['reward_strategy'] == strategy]
                equilibrium_count = strategy_data['epsilon_nash'].sum()
                total_count = len(strategy_data)
                percentage = (equilibrium_count / total_count) * 100 if total_count > 0 else 0
                
                # Get average minimum utility gain (how close we get to epsilon-Nash)
                avg_min_utility_gain = strategy_data['min_utility_gain'].mean()
                
                # Get epsilon value from the data if available
                epsilon = 0.05  # Default
                if 'epsilon_value' in strategy_data.columns:
                    epsilon_values = strategy_data['epsilon_value'].unique()
                    if len(epsilon_values) == 1:
                        epsilon = epsilon_values[0]
                
                logger.info(f"Strategy '{strategy}': {equilibrium_count}/{total_count} runs reached ε-Nash equilibrium ({percentage:.1f}%)")
                logger.info(f"  Average min utility gain: {avg_min_utility_gain:.4f} (ε = {epsilon})")
                
                if avg_min_utility_gain > epsilon:
                    gap_percentage = ((avg_min_utility_gain - epsilon) / epsilon) * 100
                    logger.info(f"  This strategy is not reaching equilibrium. Gap: {avg_min_utility_gain - epsilon:.4f} ({gap_percentage:.1f}% above threshold)")
                    
                    # Suggestions based on the gap size
                    if gap_percentage > 200:
                        logger.info(f"  RECOMMENDATION: Significantly increase epsilon (e.g., to {epsilon * 3:.2f}) or increase lambda_param (e.g., to 5.0)")
                    elif gap_percentage > 100:
                        logger.info(f"  RECOMMENDATION: Increase epsilon (e.g., to {epsilon * 2:.2f}) or increase lambda_param (e.g., to 2.0)")
                    elif gap_percentage > 50:
                        logger.info(f"  RECOMMENDATION: Slightly increase epsilon (e.g., to {epsilon * 1.5:.2f}) or increase num_steps")
                    else:
                        logger.info(f"  RECOMMENDATION: Epsilon is close - increase num_steps to allow more time to converge")
                else:
                    logger.info(f"  This strategy is getting close to equilibrium. Consider increasing num_steps to reach it consistently.")
                
                if equilibrium_count > 0:
                    equilibrium_steps = strategy_data.loc[strategy_data['epsilon_nash'], 'nash_equilibrium_step']
                    avg_steps = equilibrium_steps.mean()
                    logger.info(f"  Average steps to reach equilibrium: {avg_steps:.1f}")
        
        # Save comparison to CSV
        comparison_path = output_dir / f"{output_prefix}_compare_strategies.csv"
        strategy_comparison.to_csv(comparison_path, index=False)
        logger.info(f"Saved strategy comparison to {comparison_path}")
        
        # Create comparison visualizations
        metrics = ['avg_quality', 'participation_rate', 'high_quality_rate', 
                  'platform_utility', 'participation_stability', 'quality_stability']
        
        # Add epsilon-nash metrics if available
        if 'epsilon_nash' in strategy_comparison.columns:
            metrics.append('epsilon_nash')
            if 'nash_equilibrium_step' in strategy_comparison.columns:
                # Only include if we have valid data
                if not strategy_comparison['nash_equilibrium_step'].isna().all():
                    metrics.append('nash_equilibrium_step')
        
        # Get decision method if consistent across all results
        decision_method = ""
        if 'decision_method' in results.columns and results['decision_method'].nunique() == 1:
            decision_method = f" ({results['decision_method'].iloc[0]})"
        
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
        plt.title(f'Strategy Comparison Across All Metrics{decision_method}')
        plt.xticks(x, [metric.replace('_', ' ').title() for metric in metrics], rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = output_dir / f"{output_prefix}_compare_metrics.png"
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
        ax.bar(x, strategy_comparison['intrinsic_participation'], width, label='Intrinsic')
        
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
        for i, v in enumerate(strategy_comparison['intrinsic_participation']):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        output_path = output_dir / f"{output_prefix}_compare_contributor_types.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved contributor type comparison to {output_path}")
        
        # Create Nash equilibrium proximity analysis visualization
        if 'min_utility_gain' in strategy_comparison.columns:
            plt.figure(figsize=(10, 6))
            
            # Set up bar positions
            x = np.arange(len(strategies))
            width = 0.6
            
            # Plot min utility gain bars
            bars = plt.bar(x, strategy_comparison['min_utility_gain'], width, 
                   color='blue', alpha=0.7, label='Min Utility Gain')
            
            # Add horizontal line for epsilon threshold
            # Get epsilon value from the data if available
            epsilon = 0.05  # Default
            if 'epsilon_value' in results.columns:
                epsilon = results['epsilon_value'].iloc[0]
                
            plt.axhline(y=epsilon, color='red', linestyle='--', 
                       label=f'ε Threshold ({epsilon})')
            
            # Annotate the bars with values
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.4f}', ha='center', va='bottom')
            
            # Add labels and title
            plt.xlabel('Reward Strategy')
            plt.ylabel('Minimum Utility Gain')
            plt.title('How Close Each Strategy Gets to ε-Nash Equilibrium')
            plt.xticks(x, strategies)
            plt.legend()
            
            # Add grid for readability
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Save the figure
            output_path = output_dir / f"{output_prefix}_nash_equilibrium_proximity.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Nash equilibrium proximity analysis to {output_path}")
        
        logger.info("Strategy comparison completed")
    
    def plot_results(self, results_df, output_prefix="experiment", strategy_dir=None):
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
                   "low_quality_rate",
                   "no_contribution_rate",
                   "platform_utility", 
                   "reputation_inequality"]
        
        # Create a multi-metric line plot
        plt.figure(figsize=(14, 10))
        
        # Plot metrics that use the same y-axis scale (0-1)
        percent_metrics = ["participation_rate", "high_quality_rate", "low_quality_rate", 
                          "no_contribution_rate"]
        
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
        
        output_path = strategy_dir / f"{output_prefix}_participation_rates.png" if strategy_dir else self.results_dir / f"{output_prefix}_participation_rates.png"
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
        
        output_path = strategy_dir / f"{output_prefix}_average_quality.png" if strategy_dir else self.results_dir / f"{output_prefix}_average_quality.png"
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
        
        output_path = strategy_dir / f"{output_prefix}_platform_utility.png" if strategy_dir else self.results_dir / f"{output_prefix}_platform_utility.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved platform utility plot to {output_path}")
        
        # Create a multi-line plot for participation by contributor type
        if all(col in mean_results.columns for col in ["extrinsic_participation", "intrinsic_participation"]):
            plt.figure(figsize=(12, 6))
            
            plt.plot(mean_results["mean_effort_cost"], mean_results["extrinsic_participation"], 
                    marker='o', linewidth=2, label="Extrinsic")
            plt.plot(mean_results["mean_effort_cost"], mean_results["intrinsic_participation"], 
                    marker='^', linewidth=2, label="Intrinsic")
            
            plt.xlabel("Mean Effort Cost (γ)")
            plt.ylabel("Participation Rate")
            plt.title("Participation by Contributor Type and Effort Cost")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
            
            output_path = strategy_dir / f"{output_prefix}_participation_by_type.png" if strategy_dir else self.results_dir / f"{output_prefix}_participation_by_type.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved participation by type plot to {output_path}")
            
        # Create a plot for epsilon-Nash equilibrium if data is available
        if all(col in numeric_df.columns for col in ["epsilon_nash", "nash_equilibrium_step"]):
            # Calculate percentage of runs reaching epsilon-Nash by effort cost
            nash_results = numeric_df.groupby("mean_effort_cost").agg({
                "epsilon_nash": "mean",  # Gives percentage as true=1, false=0
                "nash_equilibrium_step": lambda x: x[pd.notna(x)].mean()  # Average step when reached
            }).reset_index()
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Percentage of runs reaching epsilon-Nash
            ax1.bar(nash_results["mean_effort_cost"], nash_results["epsilon_nash"] * 100, 
                   width=0.1, alpha=0.7, color='blue')
            ax1.set_xlabel("Mean Effort Cost (γ)")
            ax1.set_ylabel("Runs Reaching ε-Nash (%)")
            ax1.set_title("Frequency of ε-Nash Equilibrium")
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage labels
            for i, row in nash_results.iterrows():
                percentage = row["epsilon_nash"] * 100
                ax1.text(row["mean_effort_cost"], percentage + 2, 
                        f"{percentage:.1f}%", ha='center')
            
            # Plot 2: Average steps to reach equilibrium
            valid_data = nash_results.dropna(subset=["nash_equilibrium_step"])
            if not valid_data.empty:
                ax2.bar(valid_data["mean_effort_cost"], valid_data["nash_equilibrium_step"], 
                       width=0.1, alpha=0.7, color='green')
                ax2.set_xlabel("Mean Effort Cost (γ)")
                ax2.set_ylabel("Average Steps")
                ax2.set_title("Steps to Reach ε-Nash Equilibrium")
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add step count labels
                for i, row in valid_data.iterrows():
                    ax2.text(row["mean_effort_cost"], row["nash_equilibrium_step"] + 1, 
                            f"{row['nash_equilibrium_step']:.1f}", ha='center')
            else:
                ax2.text(0.5, 0.5, "No simulations reached ε-Nash equilibrium", 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            output_path = strategy_dir / f"{output_prefix}_epsilon_nash_analysis.png" if strategy_dir else self.results_dir / f"{output_prefix}_epsilon_nash_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved ε-Nash equilibrium analysis to {output_path}")

    def plot_time_series(self, results_df, output_prefix, time_series_data, strategy_dir=None):
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
                model_data = time_series_data[key]['model_data']
                utility_gain_history = time_series_data[key]['utility_gain_history']
                logger.info(f"Using stored time series data for parameters: effort={effort:.2f}")
            # else:
            #     # Fall back to running a new simulation if data not available
            #     logger.warning(f"No stored time series data found for parameters: effort={effort:.2f}")
            #     logger.warning(f"Running a new simulation to generate time series data (less efficient)")
            #     model_data = self.get_model_data_for_params(effort, reward_strategy)
            #     utility_gain_history = []
            
            if model_data is None:
                logger.warning(f"Failed to get model data for parameter combination: effort={effort:.2f}")
                continue
                
            logger.info(f"Time series data shape: {model_data.shape}")
            
            # Plot participation rates over time
            plt.figure(figsize=(12, 8))
            plt.plot(model_data.index, model_data['Participation Rate'], label='Overall Participation', linewidth=2)
            plt.plot(model_data.index, model_data['Extrinsic Participation'], label='Extrinsic Types', alpha=0.7)
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
            for col, style in zip(['Participation Rate', 'Extrinsic Participation', 'Intrinsic Participation'], 
                                 ['k', 'b', 'r']):
                plt.annotate(f"{model_data[col].iloc[-1]:.2f}", 
                            xy=(last_step, model_data[col].iloc[-1]),
                            xytext=(5, 0), textcoords='offset points',
                            color=style, fontweight='bold')
            
            output_path = strategy_dir / f"{output_prefix}_time_participation_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_time_participation_e{effort:.2f}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved participation time series to {output_path}")
            
            # NEW IMPROVED GRAPH: All quality rates over time
            quality_cols = ['High Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            if all(col in model_data.columns for col in quality_cols):
                plt.figure(figsize=(12, 8))
                colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
                
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
                
                output_path = strategy_dir / f"{output_prefix}_all_quality_rates_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_all_quality_rates_e{effort:.2f}.png"
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
            quality_cols = ['High Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
            
            for col, color in zip(quality_cols, colors):
                if col in model_data.columns:
                    ax1.plot(model_data.index, model_data[col], color=color, 
                            label=col.replace(' Rate', ''), linewidth=2)
            
            ax1.plot(model_data.index, model_data['Average Quality'], 'k-', 
                    label='Avg Quality (0-2 scale)', linewidth=2)
            
            ax1.set_ylabel('Quality Metrics (Proportion)', fontweight='bold')
            ax1.set_title(f"Quality & Platform Performance (γ={effort:.2f})")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.set_ylim(bottom=0)
            
            # Create a secondary y-axis for the average quality on 0-2 scale
            ax1_2 = ax1.twinx()
            ax1_2.set_ylabel('Avg Quality (0-2 scale)', color='k')
            ax1_2.set_ylim(0, 2.2)  # Set max slightly above 2 to allow space
            
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
            
            output_path = strategy_dir / f"{output_prefix}_platform_metrics_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_platform_metrics_e{effort:.2f}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            logger.info(f"Saved platform metrics to {output_path}")
            
            # NEW GRAPH: Contribution breakdown - Area chart
            plt.figure(figsize=(12, 8))
            quality_cols = ['High Quality Rate', 'Low Quality Rate', 'No Contribution Rate']
            if all(col in model_data.columns for col in quality_cols):
                plt.stackplot(model_data.index, 
                              [model_data[col] for col in quality_cols],
                              labels=[col.replace(' Rate', '') for col in quality_cols],
                              colors=['#2ca02c', '#ff7f0e', '#d62728'],  # Green, Orange, Red 
                              alpha=0.7)
                
                plt.title(f"Contribution Breakdown by Quality (γ={effort:.2f})")
                plt.xlabel("Time Steps")
                plt.ylabel("Proportion of Contributors")
                plt.ylim(0, 1.05)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(quality_cols))
                plt.grid(True, alpha=0.3)
                
                output_path = strategy_dir / f"{output_prefix}_contribution_breakdown_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_contribution_breakdown_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved contribution breakdown to {output_path}")
            
            # NEW GRAPH: Utility Gain over time (Nash Equilibrium proximity)
            if utility_gain_history:
                plt.figure(figsize=(12, 6))
                
                # Extract steps and gains from the history
                steps, gains = zip(*utility_gain_history)
                
                # Plot utility gain
                plt.plot(steps, gains, 'b-', linewidth=2, marker='o', markersize=4, label='Max Utility Gain')
                
                # Add epsilon threshold (get from parameters if available)
                epsilon = 0.05  # Default
                if 'epsilon_value' in results_df.columns:
                    epsilon = results_df['epsilon_value'].iloc[0]
                plt.axhline(y=epsilon, color='r', linestyle='--', label=f'ε Threshold ({epsilon})')
                
                plt.title(f"Proximity to ε-Nash Equilibrium Over Time (γ={effort:.2f})")
                plt.xlabel("Time Steps")
                plt.ylabel("Maximum Utility Gain")
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                
                # Calculate min utility gain
                min_gain = min(gains) if gains else float('inf')
                plt.annotate(f"Min Gain: {min_gain:.4f}", 
                            xy=(0.05, 0.05), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                output_path = strategy_dir / f"{output_prefix}_utility_gain_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_utility_gain_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved utility gain time series to {output_path}")
                
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
                
                output_path = strategy_dir / f"{output_prefix}_platform_economics_e{effort:.2f}.png" if strategy_dir else self.results_dir / f"{output_prefix}_platform_economics_e{effort:.2f}.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved platform economics to {output_path}")

    def get_model_data_for_params(self, effort, reward_strategy, use_ewa):
        """Run a single simulation and return time-series data. This is a fallback method."""
        logger.info(f"Running simulation to get time-series data for effort={effort:.2f}, strategy={reward_strategy}")
        logger.warning("Note: This is running an extra simulation. For efficiency, time series should be stored during parameter sweep.")
        
        # Default EWA parameters
        phi = 0.9                # Experience decay factor
        delta = 0.1              # Weight on foregone payoffs
        lambda_param = 0.5       # Softmax sharpness parameter
        reward_learning_rate = 0.2  # Learning rate for reward estimation
        
        model = IncentiveModel(
            num_contributors=500,  # Larger population for better statistics
            mean_effort_cost=effort,
            effort_cost_std=0.5,
            reward_strategy=reward_strategy,
            low_skill_ratio=0.5,
            extrinsic_ratio=0.5,
            intrinsic_ratio=0.5,
            phi=phi,
            delta=delta,
            lambda_param=lambda_param,
            reward_learning_rate=reward_learning_rate,
            use_ewa=use_ewa
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
                  "extrinsic_participation", "intrinsic_participation"]
        
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

    def plot_quality_distribution(self, results_df, output_prefix="experiment", strategy_dir=None):
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
        
        quality_cols = ['high_quality_rate', 'low_quality_rate', 'no_contribution_rate']
        quality_labels = ['High Quality', 'Low Quality', 'No Contribution']
        colors = ['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
        
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
        
        output_path = strategy_dir / f"{output_prefix}_quality_distribution_pies.png" if strategy_dir else self.results_dir / f"{output_prefix}_quality_distribution_pies.png"
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
            
            output_path = strategy_dir / f"{output_prefix}_barchart_{metric}.png" if strategy_dir else self.results_dir / f"{output_prefix}_barchart_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved bar chart for {metric} to {output_path}")
        
        logger.info("Quality distribution visualization completed")
        
    def plot_historical_rewards(self, time_series_data, output_prefix="experiment", strategy_dir=None):
        """
        Create visualizations showing the historical rewards per contribution quality.
        
        Args:
            time_series_data: Dictionary with time series data from simulation runs
            output_prefix: Prefix for output files
            strategy_dir: Directory to save visualizations to
        """
        log_divider("PLOTTING HISTORICAL REWARDS", char="-")
        logger.info("Creating historical rewards visualizations")
        
        if not time_series_data:
            logger.warning("No time series data available for historical rewards plots")
            return
        
        # Use output_dir if provided, otherwise fallback to results_dir
        output_dir = strategy_dir if strategy_dir is not None else self.results_dir
        
        # For each parameter combination, plot the historical rewards
        for (effort, strategy), data in time_series_data.items():
            # Get the model data
            model_data = data.get('model_data')
            if model_data is None:
                continue
                
            # Get a reference to the platform agent to access reward history
            model = data.get('model')
            if model is None:
                logger.warning(f"No model available for parameters: effort={effort:.2f}, strategy={strategy}")
                continue
                
            platform = model.get_platform()
            reward_history = platform.reward_history
                
            if not reward_history or all(not hist for hist in reward_history.values()):
                logger.warning(f"No reward history available for parameters: effort={effort:.2f}, strategy={strategy}")
                continue
                
            # Create a visualization showing rewards per quality over time
            plt.figure(figsize=(12, 8))
            
            # Map quality values to more readable labels
            quality_labels = {
                0: "No Contribution",
                1: "Low Quality", 
                2: "High Quality"
            }
            
            # Use colors that match other visualizations
            colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
            
            # Plot each quality's reward history
            for i, (quality, history) in enumerate(reward_history.items()):
                if history:  # Only plot if there's actual data
                    label = quality_labels.get(quality, f"Quality {quality}")
                    plt.plot(range(len(history)), history, 
                            color=colors[i], linewidth=2, 
                            label=f"{label} Rewards")
            
            plt.xlabel('Time Steps')
            plt.ylabel('Reward Amount')
            plt.title(f'Historical Rewards by Contribution Quality\nEffort Cost (γ)={effort:.2f}, Strategy={strategy.title()}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add annotations for final values
            for i, (quality, history) in enumerate(reward_history.items()):
                if history:
                    plt.annotate(f"{history[-1]:.2f}", 
                                xy=(len(history)-1, history[-1]),
                                xytext=(5, 0), textcoords='offset points',
                                color=colors[i], fontweight='bold')
            
            # Save the figure
            output_path = output_dir / f"{output_prefix}_historical_rewards_e{effort:.2f}_{strategy}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved historical rewards visualization to {output_path}")
            
            # Create a second visualization showing the ratio between high quality and low quality rewards
            if len(reward_history) >= 2 and 1 in reward_history and 2 in reward_history:
                high_quality_history = reward_history[2]
                low_quality_history = reward_history[1]
                
                # Make sure both histories have data and are the same length
                if high_quality_history and low_quality_history and len(high_quality_history) == len(low_quality_history):
                    plt.figure(figsize=(12, 6))
                    
                    # Calculate the ratio of high to low quality rewards
                    ratio_history = [high/low if low > 0 else np.nan for high, low in zip(high_quality_history, low_quality_history)]
                    
                    plt.plot(range(len(ratio_history)), ratio_history, 
                            color='purple', linewidth=2, label='High/Low Quality Reward Ratio')
                    
                    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, 
                                label='Equal Rewards (Ratio=1)')
                    
                    plt.xlabel('Time Steps')
                    plt.ylabel('Reward Ratio (High/Low)')
                    plt.title(f'High to Low Quality Reward Ratio Over Time\nEffort Cost (γ)={effort:.2f}, Strategy={strategy.title()}')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Add annotation for final ratio value
                    if not np.isnan(ratio_history[-1]):
                        plt.annotate(f"{ratio_history[-1]:.2f}x", 
                                    xy=(len(ratio_history)-1, ratio_history[-1]),
                                    xytext=(5, 0), textcoords='offset points',
                                    color='purple', fontweight='bold')
                    
                    # Save the figure
                    output_path = output_dir / f"{output_prefix}_reward_ratio_e{effort:.2f}_{strategy}.png"
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved reward ratio visualization to {output_path}")
                    
        logger.info("Historical rewards visualization completed")

    def plot_nash_equilibrium_analysis(self, results_df, time_series_data, output_prefix="nash_analysis", strategy_dir=None):
        """Create detailed visualizations focused on epsilon-Nash equilibrium properties."""
        log_divider("NASH EQUILIBRIUM ANALYSIS", char="-")
        logger.info("Creating ε-Nash equilibrium visualizations")
        
        # Use strategy_dir if provided, otherwise fallback to results_dir
        output_dir = strategy_dir if strategy_dir is not None else self.results_dir
        
        # Get epsilon value if available, otherwise use default
        epsilon = 0.05  # Default
        if 'epsilon_value' in results_df.columns:
            epsilon_values = results_df['epsilon_value'].unique()
            if len(epsilon_values) == 1:
                epsilon = epsilon_values[0]
        
        # 1. Create detailed Nash equilibrium reach time distribution
        if 'nash_equilibrium_step' in results_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Filter to only include runs that reached equilibrium
            nash_reached = results_df[results_df['epsilon_nash'] == True].copy()
            
            if not nash_reached.empty:
                # Plot histogram of steps to reach equilibrium
                bins = range(0, int(nash_reached['nash_equilibrium_step'].max()) + 5, 5)
                plt.hist(nash_reached['nash_equilibrium_step'], bins=bins, 
                         alpha=0.7, color='green', edgecolor='black')
                
                # Add vertical line for mean and median
                mean_steps = nash_reached['nash_equilibrium_step'].mean()
                median_steps = nash_reached['nash_equilibrium_step'].median()
                
                plt.axvline(mean_steps, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_steps:.1f} steps')
                plt.axvline(median_steps, color='blue', linestyle=':', linewidth=2,
                           label=f'Median: {median_steps:.1f} steps')
                
                plt.xlabel('Steps to Reach ε-Nash Equilibrium')
                plt.ylabel('Number of Simulations')
                plt.title(f'Distribution of Time to Reach ε-Nash Equilibrium (ε = {epsilon})')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add text with percentage of runs reaching equilibrium
                total_runs = len(results_df)
                nash_runs = len(nash_reached)
                percentage = (nash_runs / total_runs) * 100
                
                plt.figtext(0.7, 0.85, f"{nash_runs}/{total_runs} runs reached equilibrium\n({percentage:.1f}%)",
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
                
                output_path = output_dir / f"{output_prefix}_nash_reach_time_dist.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved Nash equilibrium reach time distribution to {output_path}")
            else:
                logger.warning("No simulations reached ε-Nash equilibrium, skipping histogram")
        
        # 2. Create a visualization of the relationship between min utility gain and participation rate
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot comparing min utility gain and participation
        sc = plt.scatter(results_df['participation_rate'], results_df['min_utility_gain'],
                       c=results_df['epsilon_nash'].astype(int), cmap='coolwarm',
                       alpha=0.7, s=100, edgecolor='k')
        
        # Add epsilon threshold line
        plt.axhline(y=epsilon, color='r', linestyle='--', 
                   label=f'ε threshold ({epsilon})')
        
        # Add annotations for points
        for i, row in results_df.iterrows():
            if row['epsilon_nash']:
                label = f"{row['nash_equilibrium_step']:.0f} steps"
            else:
                label = "No equilibrium"
                
            plt.annotate(label, 
                        (row['participation_rate'], row['min_utility_gain']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.xlabel('Participation Rate')
        plt.ylabel('Minimum Utility Gain')
        plt.title('Relationship Between Participation and Proximity to Nash Equilibrium')
        plt.colorbar(sc, label='Reached ε-Nash (1=Yes, 0=No)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_path = output_dir / f"{output_prefix}_nash_vs_participation.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Nash vs participation visualization to {output_path}")
        
        # 3. Create a detailed utility gain trajectory visualization
        if time_series_data:
            plt.figure(figsize=(15, 10))
            
            # Plot utility gain trajectories for each parameter combination
            for i, ((effort, strategy), data) in enumerate(time_series_data.items()):
                if 'utility_gain_history' in data and data['utility_gain_history']:
                    # Extract utility gain history
                    steps, gains = zip(*data['utility_gain_history'])
                    
                    # Get information about whether this run reached equilibrium
                    matching_rows = results_df[(results_df['mean_effort_cost'] == effort) & 
                                             (results_df['reward_strategy'] == strategy)]
                    
                    if not matching_rows.empty:
                        reached_equilibrium = matching_rows['epsilon_nash'].iloc[0]
                        line_style = '-' if reached_equilibrium else ':'
                        line_width = 2.5 if reached_equilibrium else 1.5
                        marker = 'o' if reached_equilibrium else None
                        marker_size = 5 if reached_equilibrium else 0
                        
                        # Plot the trajectory
                        plt.plot(steps, gains, line_style, linewidth=line_width,
                               marker=marker, markersize=marker_size,
                               label=f"Effort={effort:.2f}, Strategy={strategy} (Equil: {reached_equilibrium})")
            
            # Add epsilon threshold line
            plt.axhline(y=epsilon, color='r', linestyle='--', linewidth=2,
                       label=f'ε threshold ({epsilon})')
            
            plt.xlabel('Time Steps')
            plt.ylabel('Max Utility Gain (Lower = Closer to Equilibrium)')
            plt.title('Utility Gain Trajectories: Convergence to ε-Nash Equilibrium')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            output_path = output_dir / f"{output_prefix}_utility_gain_trajectories.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved utility gain trajectories to {output_path}")
        
        # 4. Create a "proximity to Nash" visualization
        plt.figure(figsize=(12, 8))
        
        # Calculate how close each simulation got to epsilon-Nash
        results_df['nash_proximity'] = (epsilon - results_df['min_utility_gain']) / epsilon
        # Cap the proximity at 1.0 for those that reached equilibrium
        results_df['nash_proximity'] = np.minimum(results_df['nash_proximity'], 1.0)
        
        # Sort by proximity for better visualization
        sorted_df = results_df.sort_values('nash_proximity', ascending=False)
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(sorted_df)), sorted_df['nash_proximity'], 
                       height=0.8, color='skyblue', edgecolor='navy')
        
        # Add a line at 0 (epsilon threshold)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2,
                   label=f'ε threshold ({epsilon})')
        
        # Color bars based on whether equilibrium was reached
        for i, reached in enumerate(sorted_df['epsilon_nash']):
            bars[i].set_color('green' if reached else 'skyblue')
        
        # Add labels for the effort cost and strategy
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            effort = row['mean_effort_cost']
            strategy = row['reward_strategy'] if 'reward_strategy' in row else 'unknown'
            plt.text(row['nash_proximity'] + 0.02, i, 
                    f"γ={effort:.2f}, {strategy}", 
                    va='center', fontsize=9)
            
            # Add annotation about how close to Nash
            prox_text = f"{row['nash_proximity']:.2f}"
            if row['epsilon_nash']:
                prox_text += " (Reached)"
            plt.text(max(0.02, row['nash_proximity'] - 0.3), i, prox_text, 
                    va='center', ha='center', color='white', fontweight='bold', fontsize=9)
        
        plt.yticks([])  # Hide y-axis labels since we've added our own
        plt.xlabel('Proximity to ε-Nash Equilibrium (1.0 = Reached, <0 = Far from equilibrium)')
        plt.title('How Close Each Simulation Got to ε-Nash Equilibrium')
        plt.grid(axis='x', alpha=0.3)
        
        output_path = output_dir / f"{output_prefix}_nash_proximity.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Nash proximity visualization to {output_path}")
        
        # 5. Create a "stability after equilibrium" visualization if any simulations reached equilibrium
        if 'epsilon_nash' in results_df.columns and any(results_df['epsilon_nash']):
            # Only possible if we have time series data
            if time_series_data:
                plt.figure(figsize=(15, 10))
                
                # Plot participation rate stability after reaching equilibrium
                ax1 = plt.subplot(2, 1, 1)
                
                # Track whether any data was actually plotted
                data_plotted = False
                
                for (effort, strategy), data in time_series_data.items():
                    # Check if this combination reached equilibrium
                    matching_rows = results_df[(results_df['mean_effort_cost'] == effort) & 
                                             (results_df['reward_strategy'] == strategy)]
                    
                    if not matching_rows.empty and matching_rows['epsilon_nash'].iloc[0]:
                        # Get the step at which equilibrium was reached
                        equil_step = matching_rows['nash_equilibrium_step'].iloc[0]
                        
                        # Get model data
                        model_data = data['model_data']
                        
                        # Plot participation rate before and after equilibrium
                        steps = model_data.index
                        participation = model_data['Participation Rate']
                        
                        # Color code before and after equilibrium
                        pre_steps = [s for s in steps if s < equil_step]
                        post_steps = [s for s in steps if s >= equil_step]
                        
                        pre_part = [participation[s] for s in pre_steps]
                        post_part = [participation[s] for s in post_steps]
                        
                        if pre_steps:
                            plt.plot(pre_steps, pre_part, 'b-', alpha=0.5, linewidth=1)
                        
                        if post_steps:
                            plt.plot(post_steps, post_part, 'g-', linewidth=2.5,
                                   label=f"After equilibrium (γ={effort:.2f}, {strategy})")
                            
                            # Add a vertical line at equilibrium point
                            plt.axvline(x=equil_step, color='r', linestyle='--', alpha=0.5)
                            
                            # Calculate stability metrics
                            post_std = np.std(post_part)
                            plt.annotate(f"σ={post_std:.3f}", 
                                        (post_steps[-1], post_part[-1]),
                                        xytext=(5, 0), textcoords='offset points')
                            
                            data_plotted = True
                
                if data_plotted:
                    # Increase font sizes for better readability
                    plt.title('Participation Rate Stability After Reaching ε-Nash Equilibrium', fontsize=16)
                    plt.xlabel('Time Steps', fontsize=14)
                    plt.ylabel('Participation Rate', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    # Increase legend font size
                    plt.legend(fontsize=12)
                    # Increase tick label font sizes
                    plt.tick_params(axis='both', which='major', labelsize=12)
                    
                    # Second subplot for quality
                    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                    
                    for (effort, strategy), data in time_series_data.items():
                        # Check if this combination reached equilibrium
                        matching_rows = results_df[(results_df['mean_effort_cost'] == effort) & 
                                                 (results_df['reward_strategy'] == strategy)]
                        
                        if not matching_rows.empty and matching_rows['epsilon_nash'].iloc[0]:
                            # Get the step at which equilibrium was reached
                            equil_step = matching_rows['nash_equilibrium_step'].iloc[0]
                            
                            # Get model data
                            model_data = data['model_data']
                            
                            # Plot average quality before and after equilibrium
                            steps = model_data.index
                            quality = model_data['Average Quality']
                            
                            # Color code before and after equilibrium
                            pre_steps = [s for s in steps if s < equil_step]
                            post_steps = [s for s in steps if s >= equil_step]
                            
                            pre_qual = [quality[s] for s in pre_steps]
                            post_qual = [quality[s] for s in post_steps]
                            
                            if pre_steps:
                                plt.plot(pre_steps, pre_qual, 'b-', alpha=0.5, linewidth=1)
                            
                            if post_steps:
                                plt.plot(post_steps, post_qual, 'g-', linewidth=2.5,
                                       label=f"After equilibrium (γ={effort:.2f}, {strategy})")
                                
                                # Calculate stability metrics
                                post_std = np.std(post_qual)
                                plt.annotate(f"σ={post_std:.3f}", 
                                            (post_steps[-1], post_qual[-1]),
                                            xytext=(5, 0), textcoords='offset points',
                                            fontsize=12)
                    
                    # Increase font sizes for better readability
                    plt.title('Average Quality Stability After Reaching ε-Nash Equilibrium', fontsize=16)
                    plt.xlabel('Time Steps', fontsize=14)
                    plt.ylabel('Average Quality', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    # Increase legend font size
                    plt.legend(fontsize=12)
                    # Increase tick label font sizes
                    plt.tick_params(axis='both', which='major', labelsize=12)
                    
                    # Adjust figure size for better readability
                    plt.gcf().set_size_inches(15, 12)
                    
                    # Adjust spacing between subplots to ensure labels don't overlap
                    plt.subplots_adjust(hspace=0.3)
                    
                    plt.tight_layout()
                    output_path = output_dir / f"{output_prefix}_post_equilibrium_stability.png"
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Increased DPI for higher resolution
                    plt.close()
                    logger.info(f"Saved post-equilibrium stability visualization to {output_path}")
                else:
                    logger.warning("No complete time series data found for simulations that reached equilibrium")
        
        logger.info("Nash equilibrium analysis visualization completed")

    def compare_nash_equilibrium(self, results_df, output_prefix="nash_comparison", output_dir=None):
        """
        Create comparative visualizations for Nash equilibrium across different reward strategies.
        
        This function creates specialized plots that focus on comparing how different
        reward strategies perform in terms of reaching epsilon-Nash equilibrium.
        """
        log_divider("NASH EQUILIBRIUM COMPARISON ACROSS STRATEGIES", char="-")
        logger.info("Creating comparative Nash equilibrium visualizations")
        
        # Use output_dir if provided, otherwise fallback to results_dir
        output_dir = output_dir if output_dir is not None else self.results_dir
        
        # Create data structures for analysis
        if 'reward_strategy' not in results_df.columns:
            logger.warning("No reward_strategy column found in results. Cannot compare across strategies.")
            return
        
        # Get epsilon value if available
        epsilon = 0.05  # Default
        if 'epsilon_value' in results_df.columns:
            epsilon_values = results_df['epsilon_value'].unique()
            if len(epsilon_values) == 1:
                epsilon = epsilon_values[0]
        
        # 1. Create a comparative bar chart of percentage of runs reaching Nash equilibrium
        strategies = results_df['reward_strategy'].unique()
        nash_percentages = []
        
        for strategy in strategies:
            strategy_runs = results_df[results_df['reward_strategy'] == strategy]
            total_runs = len(strategy_runs)
            nash_runs = strategy_runs['epsilon_nash'].sum()
            nash_percentages.append((nash_runs / total_runs) * 100 if total_runs > 0 else 0)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(strategies, nash_percentages, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on top of bars
        for bar, percentage in zip(bars, nash_percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Reward Strategy')
        plt.ylabel('Percentage of Runs Reaching ε-Nash Equilibrium')
        plt.title(f'Effectiveness of Reward Strategies in Reaching ε-Nash Equilibrium (ε = {epsilon})')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylim(0, 110)  # Leave room for labels
        
        output_path = output_dir / f"{output_prefix}_nash_percentage_by_strategy.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Nash equilibrium percentage by strategy to {output_path}")
        
        # 2. Create a box plot of steps to reach Nash equilibrium by strategy
        plt.figure(figsize=(12, 8))
        
        # Prepare data for box plot
        box_data = []
        strategy_labels = []
        
        for strategy in strategies:
            strategy_runs = results_df[(results_df['reward_strategy'] == strategy) & 
                                     (results_df['epsilon_nash'] == True)]
            
            if not strategy_runs.empty:
                box_data.append(strategy_runs['nash_equilibrium_step'])
                
                # Create label with mean steps and count
                mean_steps = strategy_runs['nash_equilibrium_step'].mean()
                count = len(strategy_runs)
                strategy_labels.append(f"{strategy}\n({count} runs, μ={mean_steps:.1f})")
            else:
                box_data.append([])
                strategy_labels.append(f"{strategy}\n(0 runs)")
        
        # Create box plot
        bp = plt.boxplot(box_data, patch_artist=True, notch=True, showfliers=True)
        
        # Customize box appearance
        for box in bp['boxes']:
            box.set(facecolor='skyblue', alpha=0.7)
        
        for whisker in bp['whiskers']:
            whisker.set(color='navy', linewidth=1.5)
        
        for cap in bp['caps']:
            cap.set(color='navy', linewidth=1.5)
        
        for median in bp['medians']:
            median.set(color='red', linewidth=2)
        
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='red', alpha=0.5, markersize=6)
        
        plt.xticks(range(1, len(strategy_labels) + 1), strategy_labels)
        plt.ylabel('Steps to Reach ε-Nash Equilibrium')
        plt.title(f'Distribution of Steps to Reach ε-Nash Equilibrium by Strategy (ε = {epsilon})')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add jittered scatter points for individual data points
        for i, data in enumerate(box_data):
            if len(data) > 0:  # Check if data is not empty
                # Add some random jitter for better visualization
                x = np.random.normal(i + 1, 0.04, size=len(data))
                plt.scatter(x, data, alpha=0.4, color='navy', s=30)
        
        output_path = output_dir / f"{output_prefix}_nash_steps_boxplot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Nash equilibrium steps box plot to {output_path}")
        
        # 3. Create a scatter plot comparing proximity to Nash equilibrium with platform utility
        plt.figure(figsize=(12, 8))
        
        # Calculate proximity to Nash
        results_df['nash_proximity'] = (epsilon - results_df['min_utility_gain']) / epsilon
        results_df['nash_proximity'] = np.minimum(results_df['nash_proximity'], 1.0)
        
        # Create unique markers and colors for each strategy
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        # Create scatter plot
        for i, strategy in enumerate(strategies):
            strategy_data = results_df[results_df['reward_strategy'] == strategy]
            plt.scatter(strategy_data['platform_utility'], strategy_data['nash_proximity'],
                       marker=markers[i % len(markers)], 
                       color=colors[i % len(colors)],
                       s=100, alpha=0.7, label=strategy, edgecolor='k')
        
        # Add a horizontal line at nash_proximity = 0 (epsilon threshold)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, 
                   label=f'ε threshold ({epsilon})')
        
        # Add annotation for each point showing the effort cost
        for i, row in results_df.iterrows():
            plt.annotate(f"γ={row['mean_effort_cost']:.2f}", 
                        (row['platform_utility'], row['nash_proximity']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.xlabel('Platform Utility')
        plt.ylabel('Proximity to ε-Nash Equilibrium')
        plt.title('Relationship Between Platform Utility and Nash Equilibrium')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Reward Strategy')
        
        output_path = output_dir / f"{output_prefix}_utility_vs_nash.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved platform utility vs Nash proximity to {output_path}")
        
        # 4. Create a stability comparison for strategies that reached equilibrium
        if 'epsilon_nash' in results_df.columns and any(results_df['epsilon_nash']):
            plt.figure(figsize=(15, 10))
            
            # We'll need two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
            
            # Track strategies with valid data
            valid_strategies = []
            
            # For each strategy, calculate post-equilibrium stability metrics
            for strategy in strategies:
                strategy_runs = results_df[(results_df['reward_strategy'] == strategy) & 
                                         (results_df['epsilon_nash'] == True)]
                
                if not strategy_runs.empty:
                    # Calculate stability metrics
                    participation_stability = strategy_runs['participation_stability'].mean()
                    quality_stability = strategy_runs['quality_stability'].mean()
                    
                    valid_strategies.append(strategy)
                    
                    # Add to the participation stability plot
                    ax1.bar(strategy, participation_stability, alpha=0.7, 
                           color='skyblue', edgecolor='navy')
                    ax1.text(strategy, participation_stability + 0.01, 
                            f"{participation_stability:.3f}", 
                            ha='center', va='bottom')
                    
                    # Add to the quality stability plot
                    ax2.bar(strategy, quality_stability, alpha=0.7,
                           color='lightgreen', edgecolor='darkgreen')
                    ax2.text(strategy, quality_stability + 0.01,
                            f"{quality_stability:.3f}",
                            ha='center', va='bottom')
            
            if valid_strategies:
                # Set titles and labels
                ax1.set_title('Participation Rate Stability After Reaching ε-Nash Equilibrium')
                ax1.set_ylabel('Stability (higher is better)')
                ax1.grid(axis='y', linestyle='--', alpha=0.3)
                
                ax2.set_title('Quality Stability After Reaching ε-Nash Equilibrium')
                ax2.set_xlabel('Reward Strategy')
                ax2.set_ylabel('Stability (higher is better)')
                ax2.grid(axis='y', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                
                output_path = output_dir / f"{output_prefix}_post_nash_stability.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved post-Nash stability comparison to {output_path}")
            else:
                logger.warning("No strategies had simulations that reached equilibrium, skipping stability comparison")
                plt.close()
        
        # 5. Create a radar/spider chart comparing overall Nash equilibrium metrics
        strategy_metrics = {}
        
        for strategy in strategies:
            strategy_runs = results_df[results_df['reward_strategy'] == strategy]
            
            # Calculate metrics
            nash_reach_pct = strategy_runs['epsilon_nash'].mean() * 100
            
            # For steps to reach, only consider runs that reached equilibrium
            nash_reached = strategy_runs[strategy_runs['epsilon_nash'] == True]
            if not nash_reached.empty:
                # Normalize steps to reach - lower is better, so invert
                max_steps = 100  # Theoretical maximum
                steps_to_reach = nash_reached['nash_equilibrium_step'].mean()
                steps_normalized = 100 * (1 - (steps_to_reach / max_steps))
            else:
                steps_normalized = 0
            
            # Proximity to Nash for runs that didn't reach equilibrium
            nash_not_reached = strategy_runs[strategy_runs['epsilon_nash'] == False]
            if not nash_not_reached.empty:
                # Higher proximity is better (closer to Nash)
                proximity = (epsilon - nash_not_reached['min_utility_gain'].mean()) / epsilon
                proximity_normalized = max(0, proximity * 100)
            else:
                proximity_normalized = 100 if not nash_reached.empty else 0
            
            # Post-equilibrium stability (if applicable)
            if not nash_reached.empty:
                participation_stability = nash_reached['participation_stability'].mean() * 100
                quality_stability = nash_reached['quality_stability'].mean() * 100
            else:
                participation_stability = 0
                quality_stability = 0
            
            strategy_metrics[strategy] = {
                'Nash Reach %': nash_reach_pct,
                'Quick Convergence': steps_normalized,
                'Nash Proximity': proximity_normalized,
                'Participation Stability': participation_stability,
                'Quality Stability': quality_stability
            }
        
        # Create the radar chart if we have data
        if strategy_metrics:
            plt.figure(figsize=(12, 10))
            
            # Prepare data for radar chart
            categories = ['Nash Reach %', 'Quick Convergence', 'Nash Proximity', 
                        'Participation Stability', 'Quality Stability']
            
            # Number of categories
            N = len(categories)
            
            # Create angle for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize the plot
            ax = plt.subplot(111, polar=True)
            
            # Set the first axis to be on top
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            
            # Draw y-axis labels
            ax.set_rlabel_position(0)
            plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
            
            # Plot each strategy
            for i, strategy in enumerate(strategies):
                values = [strategy_metrics[strategy][cat] for cat in categories]
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', 
                       label=strategy, color=colors[i % len(colors)])
                ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.title('Nash Equilibrium Performance Comparison', size=15, y=1.1)
            
            output_path = output_dir / f"{output_prefix}_nash_radar_chart.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Nash equilibrium radar chart to {output_path}")
            
        logger.info("Comparative Nash equilibrium analysis completed")

    def plot_nash_convergence_analysis(self, results_df, time_series_data, output_prefix="nash_convergence", output_dir=None):
        """Create specialized visualizations to analyze convergence to Nash equilibrium."""
        log_divider("NASH CONVERGENCE ANALYSIS", char="-")
        logger.info("Creating Nash equilibrium convergence visualizations")
        
        # Use output_dir if provided, otherwise fallback to results_dir
        output_dir = output_dir if output_dir is not None else self.results_dir
        
        # Ensure we have the necessary data
        if not time_series_data:
            logger.warning("No time series data available for convergence analysis.")
            return
            
        # Get epsilon value if available
        epsilon = 0.05  # Default
        if 'epsilon_value' in results_df.columns:
            epsilon_values = results_df['epsilon_value'].unique()
            if len(epsilon_values) == 1:
                epsilon = epsilon_values[0]
        
        # 1. Create a detailed convergence path visualization
        plt.figure(figsize=(15, 10))
        
        # Track which parameter combinations reached Nash equilibrium
        reached_nash = {}
        not_reached_nash = {}
        
        for (effort, strategy), data in time_series_data.items():
            if 'nash_analysis' in data and 'all_utility_gains' in data['nash_analysis']:
                # Get the utility gains and steps
                utility_gains = data['nash_analysis']['all_utility_gains']
                steps = data['nash_analysis']['all_steps_checked']
                reached = data['nash_analysis']['epsilon_nash_reached']
                
                if reached:
                    reached_nash[(effort, strategy)] = (steps, utility_gains)
                else:
                    not_reached_nash[(effort, strategy)] = (steps, utility_gains)
        
        # Plot convergence paths for runs that reached Nash
        for i, ((effort, strategy), (steps, gains)) in enumerate(reached_nash.items()):
            # Normalize utility gains by epsilon to make them more comparable
            normalized_gains = [gain / epsilon for gain in gains]
            
            # Plot as a solid line with marker for the equilibrium point
            equilibrium_idx = None
            for j, gain in enumerate(normalized_gains):
                if gain <= 1.0:  # At or below epsilon threshold
                    equilibrium_idx = j
                    break
            
            if equilibrium_idx is not None:
                # Plot pre-equilibrium trajectory
                plt.plot(steps[:equilibrium_idx+1], normalized_gains[:equilibrium_idx+1], 
                       '-', linewidth=2, 
                       label=f"γ={effort:.2f}, {strategy} (Reached)",
                       color=plt.cm.tab10(i % 10))
                
                # Mark the equilibrium point
                plt.scatter([steps[equilibrium_idx]], [normalized_gains[equilibrium_idx]], 
                          color='black', s=100, zorder=5, marker='*')
                
                # Plot post-equilibrium trajectory in different style
                if equilibrium_idx < len(steps) - 1:
                    plt.plot(steps[equilibrium_idx:], normalized_gains[equilibrium_idx:], 
                           '--', linewidth=1.5, alpha=0.7,
                           color=plt.cm.tab10(i % 10))
        
        # Plot convergence paths for runs that didn't reach Nash
        for i, ((effort, strategy), (steps, gains)) in enumerate(not_reached_nash.items()):
            # Normalize utility gains by epsilon
            normalized_gains = [gain / epsilon for gain in gains]
            
            # Plot as dotted line
            plt.plot(steps, normalized_gains, ':', linewidth=1.5, alpha=0.5,
                   label=f"γ={effort:.2f}, {strategy} (Not Reached)",
                   color=plt.cm.Paired(i % 10))
        
        # Add a horizontal line at the epsilon threshold (1.0 when normalized)
        plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
                   label=f'ε threshold')
        
        # Add horizontal lines for 2x and 3x epsilon thresholds
        plt.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'2ε threshold')
        plt.axhline(y=3.0, color='yellow', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'3ε threshold')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Utility Gain / ε (lower is better)')
        plt.title('Convergence Paths to ε-Nash Equilibrium')
        plt.grid(True, alpha=0.3)
        
        # Add legend with smaller font size and place outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Set y-axis limits with some padding
        max_gain = 5.0  # Default in case calculations fail
        try:
            all_gains = []
            for gains_tuple in reached_nash.values():
                normalized_gains = [gain / epsilon for gain in gains_tuple[1]]
                all_gains.extend(normalized_gains)
            
            for gains_tuple in not_reached_nash.values():
                normalized_gains = [gain / epsilon for gain in gains_tuple[1]]
                all_gains.extend(normalized_gains)
                
            if all_gains:
                max_gain = max(all_gains) * 1.1
        except:
            logger.warning("Error calculating y-axis limits, using default value")
        
        plt.ylim(0, min(10, max_gain))  # Cap at 10x epsilon for readability
        
        plt.tight_layout()
        output_path = output_dir / f"{output_prefix}_convergence_paths.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Nash convergence paths to {output_path}")
        
        # 3. Create a visualization of convergence stability
        plt.figure(figsize=(15, 10))
        
        # Plot volatility in utility gain over time for each run
        fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # First subplot - raw utility gain trajectories
        ax1 = axs[0]
        
        # Plot the raw utility gains
        for i, ((effort, strategy), (steps, gains)) in enumerate(reached_nash.items()):
            # Normalize by epsilon
            normalized_gains = [gain / epsilon for gain in gains]
            
            # Plot the normalized gains
            ax1.plot(steps, normalized_gains, '-', linewidth=2,
                    label=f"γ={effort:.2f}, {strategy}", 
                    color=plt.cm.tab10(i % 10))
            
            # Mark the equilibrium threshold
            ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        
        ax1.set_ylabel('Utility Gain / ε')
        ax1.set_title('Raw Utility Gain Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Second subplot - volatility in utility gain
        ax2 = axs[1]
        
        # For each run that reached Nash, calculate the step-by-step changes
        for i, ((effort, strategy), (steps, gains)) in enumerate(reached_nash.items()):
            if len(gains) > 1:
                # Calculate step-by-step changes
                changes = [abs(gains[j] - gains[j-1]) for j in range(1, len(gains))]
                change_steps = [steps[j] for j in range(1, len(gains))]
                
                # Normalize by epsilon
                normalized_changes = [change / epsilon for change in changes]
                
                # Plot the volatility
                ax2.plot(change_steps, normalized_changes, '-', linewidth=2,
                        label=f"γ={effort:.2f}, {strategy}",
                        color=plt.cm.tab10(i % 10))
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Change in Utility Gain / ε')
        ax2.set_title('Volatility in Utility Gain (Higher = More Unstable)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        output_path = output_dir / f"{output_prefix}_convergence_stability.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved convergence stability visualization to {output_path}")
        
        # 4. Create a phase diagram showing progression of participation vs quality
        plt.figure(figsize=(12, 10))
        
        for (effort, strategy), data in time_series_data.items():
            # Extract the model data
            model_data = data['model_data']
            
            # Check if this run reached Nash
            reached = False
            nash_step = None
            
            if 'nash_analysis' in data:
                reached = data['nash_analysis']['epsilon_nash_reached']
                nash_step = data['nash_analysis']['nash_equilibrium_step']
            
            # Create a scatter plot of participation vs quality
            # Use color gradient to show progression over time
            if 'Participation Rate' in model_data.columns and 'Average Quality' in model_data.columns:
                participation = model_data['Participation Rate']
                quality = model_data['Average Quality']
                
                # Create an array of time steps
                time_steps = np.array(model_data.index)
                
                # Plot with color gradient based on time
                scatter = plt.scatter(participation, quality, 
                                    c=time_steps, cmap='viridis', 
                                    s=50, alpha=0.8, 
                                    edgecolor='k', linewidth=0.5,
                                    label=f"γ={effort:.2f}, {strategy}")
                
                # Plot a line showing the trajectory
                plt.plot(participation, quality, '-', alpha=0.4,
                        color=plt.cm.Set2(hash(strategy) % 8))
                
                # Mark the starting point
                plt.scatter(participation.iloc[0], quality.iloc[0], 
                          s=100, marker='o', edgecolor='k', facecolor='none',
                          linewidth=2)
                
                # Mark the equilibrium point if reached
                if reached and nash_step is not None:
                    try:
                        # Find the closest index
                        closest_idx = np.argmin(np.abs(np.array(model_data.index) - nash_step))
                        plt.scatter(participation.iloc[closest_idx], quality.iloc[closest_idx], 
                                  s=150, marker='*', edgecolor='k', facecolor='yellow',
                                  linewidth=1.5, zorder=10)
                    except:
                        # If there's an issue finding the exact index, skip marking it
                        pass
                
                # Mark the final point
                plt.scatter(participation.iloc[-1], quality.iloc[-1], 
                          s=100, marker='s', edgecolor='k', facecolor='red',
                          linewidth=1.5, zorder=5)
                
                # Add trajectory direction arrows
                num_arrows = 5
                if len(participation) > num_arrows + 1:
                    indices = np.linspace(0, len(participation)-1, num_arrows, dtype=int)
                    for i in range(len(indices)-1):
                        idx = indices[i]
                        next_idx = indices[i+1]
                        plt.annotate("", 
                                    xy=(participation.iloc[next_idx], quality.iloc[next_idx]),
                                    xytext=(participation.iloc[idx], quality.iloc[idx]),
                                    arrowprops=dict(arrowstyle="->", color=plt.cm.Set2(hash(strategy) % 8), 
                                                  lw=1.5, alpha=0.7))
        
        plt.xlabel('Participation Rate')
        plt.ylabel('Average Quality')
        plt.title('Phase Diagram: Quality vs Participation Trajectory')
        plt.grid(True, alpha=0.3)
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time Steps')
        
        # Add legend for the markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='k',
                  markersize=10, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markeredgecolor='k',
                  markersize=12, label='Nash Equilibrium'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markeredgecolor='k',
                  markersize=10, label='End')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        output_path = output_dir / f"{output_prefix}_phase_diagram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved phase diagram to {output_path}")
        
        logger.info("Nash convergence analysis completed")


if __name__ == "__main__":
    logger.info("Starting simulation runner")
    runner = SimulationRunner()
    results, time_series = runner.run_experiment() 
    logger.info("Simulation complete")