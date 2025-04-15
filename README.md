# Monetary-Reputation Incentives Model

## Project Overview

This project implements a sophisticated agent-based model to study the complex interaction between monetary and reputation-based incentives in online contribution systems. The simulation explores how various incentive structures affect contribution quality, participation rates, and overall platform utility across diverse contributor populations.

The model aims to address several key research questions:
1. How do monetary incentives interact with reputation-based incentives in motivating quality contributions?
2. What is the optimal balance between these incentive types for maximizing platform utility?
3. How do different contributor types (intrinsically vs. extrinsically motivated) respond to various incentive structures?
4. What are the stability and growth characteristics of different incentive configurations over time?

## Theoretical Foundation

The model is grounded in behavioral economics and motivation theory, incorporating:
- **Utility Maximization Framework**: Agents make decisions based on expected utility calculations
- **Motivation Crowding Theory**: External incentives can either crowd out or reinforce intrinsic motivation
- **Reputation Economics**: Reputation serves as both a signal of quality and a form of non-monetary incentive
- **Heterogeneous Agent Modeling**: Contributors differ in their motivational orientation and cost structures

## Current Implementation

The model has been fully implemented and includes:

### Core Components
- **Agent Classes**: Implemented `ContributorAgent` and `PlatformAgent` classes in `agents.py`
- **Model Framework**: Built the `IncentiveModel` class using Mesa in `model.py`
- **Reward Strategies**: Implemented multiple reward distribution strategies in `reward_distributor.py`:
  - Weighted distribution (higher quality gets disproportionately higher rewards)
  - Linear distribution (rewards proportional to quality)
  - Threshold distribution (rewards only for contributions above a quality threshold)
- **Simulation Runner**: Created comprehensive simulation framework in `simulation.py` with parameter sweeps and visualization

### Agent Attributes and Behavior
- **Contributor Types**: Three distinct profiles with different sensitivities to monetary vs. reputation incentives
  - Extrinsically motivated (30%)
  - Balanced (50%)
  - Intrinsically motivated (20%)
- **Skill Levels**: Varying skill levels affecting contribution quality probabilities
- **Decision Process**: Sophisticated utility calculation for each possible action based on:
  - Expected monetary rewards
  - Expected reputation gains
  - Intrinsic satisfaction
  - Effort costs

### Data Collection and Analysis
- **Model Metrics**: Comprehensive collection of metrics at each step:
  - Average contribution quality
  - Participation rates (overall and by contributor type)
  - High/medium/low quality rates
  - Platform utility
  - Reputation inequality (Gini coefficient)
- **Stability and Growth**: Analysis of metric variation and trends over time

### Visualization Suite
Extensive visualization capabilities have been implemented, generating:
- **Time Series Analysis**: Participation rates, quality measures, and platform utility over time
- **Parameter Sweeps**: Effects of effort costs on key metrics across reward strategies
- **Comparative Analysis**: Effectiveness of different reward strategies
- **Contribution Breakdowns**: Distribution of contribution quality levels
- **Economic Analysis**: Platform benefits, costs, and utility
- **Participant Behavior**: Analysis by contributor type

## Results

Simulations have been run with multiple parameter combinations and reward strategies:

1. **Reward Strategy Comparison**: Analyzed weighted, linear, and threshold-based reward distribution
2. **Effort Cost Variation**: Tested different mean effort costs (0.5, 1.25, 2.0)
3. **Contributor Behavior Analysis**: Examined how different agent types respond to incentive structures
4. **Platform Economics**: Evaluated benefit-cost trade-offs and overall platform utility

Key results have been saved in the `results` directory as CSV files and visualizations.

## Usage

The model can be run by:

```python
# Import the simulation runner
from simulation import SimulationRunner

# Create a runner with specified results directory
runner = SimulationRunner(results_dir="results")

# Run a full experiment with default parameters
runner.run_experiment(output_prefix="experiment")
```

## Next Steps

Potential enhancements and extensions include:
- Expanding the parameter space for more comprehensive analysis
- Implementing adaptive reward strategies that evolve over time
- Adding network effects and social influence between contributors
- Modeling strategic behavior and gaming of the incentive system
- Running longer-term simulations to analyze system stability