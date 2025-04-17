"""
Monetary-Reputation Incentives Model: Game-Theoretic Framework
===============================================================

Formal Game Description:
------------------------
This model implements a Bayesian game with incomplete information where heterogeneous
contributors interact with a platform that distributes monetary rewards.

1. Players:
   - N contributor agents with heterogeneous types (extrinsic, intrinsic)
   - 1 platform agent that distributes rewards

2. Types:
   - Contributor types θ ∈ {Extrinsic, Intrinsic}
   - Skill levels s ∈ {Low, High}
   - Type distribution follows fixed probabilities (e.g., 50% extrinsic, 50% intrinsic)
   - Individual parameters (monetary/reputation sensitivity, effort cost) vary within types

3. Actions:
   - Contributors choose effort level e ∈ {No Effort, Low Effort, High Effort}
   - Effort + skill determine probabilistic contribution quality q ∈ {No Contribution, Low, High}
   - Platform distributes rewards according to a reward strategy R(q)

4. Payoffs:
   - Contributor utility: U(e,q,θ) = α(θ)·M(q) + β(θ)·Rep(q) - γ·Cost(e)
     where α(θ) is monetary sensitivity, β(θ) is reputation sensitivity,
     and γ is individual effort cost
   - Platform utility: V(q,R) = Benefit(q) - Cost(R)

5. Information:
   - Contributors know their own type but not others' types
   - Contributors observe historical reward distributions
   - Platform observes contribution qualities but not agent types or effort

Equilibrium Concept:
-------------------
The model implements a form of Bayesian Nash Equilibrium with learning dynamics.
Contributors choose effort levels to maximize expected utility given:
1. Their beliefs about reward distribution (formed through fictitious play)
2. Their type-dependent sensitivities to monetary and reputation incentives
3. Their skill-dependent quality probabilities

The equilibrium emerges when:
- Each contributor chooses optimal effort given their beliefs and type
- The distribution of contribution qualities stabilizes
- Reward distribution follows a consistent pattern

Theoretical Foundation:
----------------------
The model integrates several theoretical frameworks:
1. Expected utility theory: Agents maximize expected utility given probabilistic outcomes
2. Fictitious play: Agents form beliefs based on empirical distribution of past rewards
3. Motivation crowding theory: External incentives interact with intrinsic motivation
4. Signaling theory: Contribution quality signals agent skill and commitment
5. Heterogeneous agent modeling: Diverse agent types create strategic complexity
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np

from agents import PlatformAgent, ContributorAgent, ContributionQuality, ContributorType, ParticipationDecision

def compute_average_quality(model):
    """Calculate average contribution quality."""
    contributions = [agent.contribution_history[-1] 
                    for agent in model.agents 
                    if isinstance(agent, ContributorAgent) and agent.contribution_history]
    if contributions:
        average_quality = np.mean(contributions)
    else:
        average_quality = 0

    return average_quality

def compute_participation_rate(model):
    """Calculate participation rate based on actual decisions."""
    decisions = [agent.participation_decision_history[-1] 
                for agent in model.agents 
                if isinstance(agent, ContributorAgent) and agent.participation_decision_history]
    
    if decisions:   
        participation_rate = sum(d == ParticipationDecision.CONTRIBUTE.value for d in decisions) / len(decisions)
    else:
        participation_rate = 0

    return participation_rate

def compute_high_quality_rate(model):
    """Calculate rate of high quality contributions."""
    contributions = [agent.contribution_history[-1] 
                    for agent in model.agents 
                    if isinstance(agent, ContributorAgent) and agent.contribution_history]
    
    if contributions:   
        high_quality_rate = sum(c == ContributionQuality.HIGH_QUALITY.value for c in contributions) / len(contributions)
    else:
        high_quality_rate = 0

    return high_quality_rate

def compute_low_quality_rate(model):
    """Compute the proportion of active contributors submitting low quality contributions."""
    contributors = [agent for agent in model.agents if isinstance(agent, ContributorAgent)]
    if not contributors:
        return 0
    
    low_quality_contributors = 0
    for agent in contributors:
        if agent.contribution_history and agent.contribution_history[-1] == ContributionQuality.LOW_QUALITY.value:
            low_quality_contributors += 1
    
    return low_quality_contributors / len(contributors)

def compute_no_contribution_rate(model):
    """Compute the proportion of contributors who chose not to contribute."""
    contributors = [agent for agent in model.agents if isinstance(agent, ContributorAgent)]
    if not contributors:
        return 0
    
    non_contributors = 0
    for agent in contributors:
        if agent.contribution_history and agent.contribution_history[-1] == ContributionQuality.NO_CONTRIBUTION.value:
            non_contributors += 1
    
    return non_contributors / len(contributors)

def compute_extrinsic_participation(model):
    """Calculate participation rate for extrinsic contributors based on decisions."""
    type_agents = [agent for agent in model.agents 
                  if isinstance(agent, ContributorAgent) and agent.contributor_type == ContributorType.EXTRINSIC]
    
    if not type_agents:
        return 0
        
    decisions = [agent.participation_decision_history[-1] 
                for agent in type_agents if agent.participation_decision_history]
    
    if decisions:
        participation_rate = sum(d == ParticipationDecision.CONTRIBUTE.value for d in decisions) / len(decisions)
    else:
        participation_rate = 0
        
    return participation_rate

def compute_intrinsic_participation(model):
    """Calculate participation rate for intrinsic contributors based on decisions."""
    type_agents = [agent for agent in model.agents 
                  if isinstance(agent, ContributorAgent) and agent.contributor_type == ContributorType.INTRINSIC]
    
    if not type_agents:
        return 0
        
    decisions = [agent.participation_decision_history[-1] 
                for agent in type_agents if agent.participation_decision_history]
    
    if decisions:
        participation_rate = sum(d == ParticipationDecision.CONTRIBUTE.value for d in decisions) / len(decisions)
    else:
        participation_rate = 0
        
    return participation_rate

def compute_gini_coefficient(model):
    """Calculate Gini coefficient for reputation distribution as inequality measure."""
    reputations = [agent.current_reputation for agent in model.agents 
                  if isinstance(agent, ContributorAgent)]
    
    if not reputations or all(x == 0 for x in reputations):
        return 0
    
    reputations = sorted(reputations)
    n = len(reputations)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * reputations) / (n * np.sum(reputations))) - (n + 1) / n

def compute_platform_utility(model):
    """Calculate platform's utility (benefit minus cost)."""
    platform = model.get_platform()
    return platform.compute_step_utility()

class IncentiveModel(Model):
    """Model class for the monetary-reputation incentive system."""
    
    def __init__(self,
                num_contributors,
                reward_pool,
                mean_effort_cost, 
                effort_cost_std,
                extrinsic_ratio,
                intrinsic_ratio,
                low_skill_ratio,
                reward_strategy,
                phi,
                delta,
                lambda_param,
                reward_learning_rate,
                use_ewa):
        """
        Initialize the incentive model.
        
        Args:
            num_contributors: Number of contributor agents
            reward_pool: Total monetary rewards to distribute per step
            mean_effort_cost: Mean effort cost for contributors
            effort_cost_std: Standard deviation of effort cost
            extrinsic_ratio: Proportion of extrinsically motivated contributors
            intrinsic_ratio: Proportion of intrinsically motivated contributors
            low_skill_ratio: Proportion of low-skill contributors
            reward_strategy: Strategy for distributing rewards
            phi: EWA parameter - decay factor for experience (0 to 1)
            delta: EWA parameter - weight on foregone payoffs (0 to 1)
            lambda_param: EWA parameter - softmax sharpness parameter (1/temperature)
                          Low values (0.5-1) create more exploration, high values (>5) create exploitation
            reward_learning_rate: Learning rate for updating reward estimates (0 to 1)
                                 Controls how quickly agents adapt to new reward information
            use_ewa: Whether to use Experience-Weighted Attraction learning (True) or
                    direct utility maximization (False)
        """
        
        super().__init__()
        
        self.num_contributors = num_contributors
        self.is_epsilon_nash = False  # Track if we've reached epsilon-Nash equilibrium
        self.use_ewa = use_ewa  # Store whether we're using EWA learning
        
        # Store EWA parameters
        self.phi = phi
        self.delta = delta
        self.lambda_param = lambda_param
        self.reward_learning_rate = reward_learning_rate

        # Create platform agent with specified reward strategy
        self.platform = PlatformAgent(self, reward_pool, reward_strategy)
        
        type_distribution = []
        type_distribution.extend([ContributorType.EXTRINSIC] * int(num_contributors * extrinsic_ratio))
        type_distribution.extend([ContributorType.INTRINSIC] * int(num_contributors * intrinsic_ratio))

        # Create contributor agents with varying effort costs
        for contributor_type in type_distribution:
            effort_cost = max(0.1, np.random.normal(mean_effort_cost, effort_cost_std))
            # Initial reputation is drawn from beta distribution (most start low, few start high)
            initial_reputation = np.random.beta(1.5, 5.0)
            _ = ContributorAgent(
                self, 
                effort_cost, 
                initial_reputation, 
                contributor_type,
                skill_level=None,
                phi=phi,
                delta=delta,
                lambda_param=lambda_param,
                reward_learning_rate=reward_learning_rate,
                use_ewa=use_ewa
            )
            
        # Set up data collection
        model_reporters = {
            "Average Quality": compute_average_quality,
            "Participation Rate": compute_participation_rate,
            "High Quality Rate": compute_high_quality_rate,
            "Low Quality Rate": compute_low_quality_rate,
            "No Contribution Rate": compute_no_contribution_rate,
            "Extrinsic Participation": compute_extrinsic_participation,
            "Intrinsic Participation": compute_intrinsic_participation,
            "Reputation Inequality (Gini)": compute_gini_coefficient,
            "Platform Utility": compute_platform_utility,
            "Platform Step Benefit": lambda m: m.platform.current_step_benefit if hasattr(m, 'platform') else 0,
            "Platform Step Cost": lambda m: m.platform.current_step_cost if hasattr(m, 'platform') else 0,
            "Reward Strategy": lambda m: m.platform.get_reward_strategy_info()['name'],
            "Is Epsilon Nash": lambda m: m.is_epsilon_nash,
            "Decision Method": lambda m: "EWA Learning" if m.use_ewa else "Direct Utility Max"
        }
        
        agent_reporters = {
            "Reputation": "current_reputation",
            "Contributor Type": lambda agent: agent.contributor_type.name if hasattr(agent, 'contributor_type') else None,
            "Last Contribution": lambda agent: agent.contribution_history[-1] if isinstance(agent, ContributorAgent) and agent.contribution_history else None,
            "Last Decision": lambda agent: agent.participation_decision_history[-1] if isinstance(agent, ContributorAgent) and agent.participation_decision_history else None,
            "Skill Level": lambda agent: agent.skill_level.name if hasattr(agent, 'skill_level') else None,
        }
        
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )
        
    def get_platform(self):
        """Get the platform agent."""
        return self.platform
    
    def is_epsilon_nash_equilibrium(self, epsilon=0.05, sample_size=30):
        """
        Test if the model has reached an ε-Nash equilibrium.
        
        This tests whether a random sample of contributor agents could improve
        their expected utility by more than ε by unilaterally changing their
        participation decision.
        
        Args:
            epsilon: Maximum allowed utility improvement for ε-Nash equilibrium
            sample_size: Number of random agents to sample for the test
            
        Returns:
            tuple: (reached_equilibrium, max_utility_gain, agents_checked)
                - reached_equilibrium: True if the model is in an ε-Nash equilibrium, False otherwise
                - max_utility_gain: The maximum utility gain observed across all agents
                - agents_checked: Number of agents with valid decisions that were checked
        """
        contributor_agents = [agent for agent in self.agents if isinstance(agent, ContributorAgent)]
        
        # If we have fewer contributors than the sample size, use all of them
        if len(contributor_agents) <= sample_size:
            sample = contributor_agents
        else:
            # Take a random sample of the specified size
            sample = np.random.choice(contributor_agents, size=sample_size, replace=False)
        
        max_utility_gain = 0.0
        agents_checked = 0
        
        for agent in sample:
            # Get current participation decision
            if not agent.participation_decision_history:
                # Skip agents that haven't made a decision yet
                continue
            
            agents_checked += 1
            current_decision = agent.participation_decision_history[-1]
            
            # Calculate utility for current decision
            current_utility = agent.compute_expected_utility(current_decision)
            
            # Calculate utility for alternative decision
            alternative_decision = 1 - current_decision  # Toggle between 0 and 1
            alternative_utility = agent.compute_expected_utility(alternative_decision)
            
            # Calculate utility gain
            utility_gain = max(0, alternative_utility - current_utility)
            max_utility_gain = max(max_utility_gain, utility_gain)
            
            # Check if agent could improve utility by more than epsilon
            if utility_gain > epsilon:
                # We found an agent that could improve utility beyond epsilon
                self.is_epsilon_nash = False
                return False, utility_gain, agents_checked
        
        # No agent in the sample could improve utility by more than epsilon
        self.is_epsilon_nash = True
        return True, max_utility_gain, agents_checked
    
    def step(self):
        """Execute one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("step")