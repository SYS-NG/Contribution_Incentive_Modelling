"""
Monetary-Reputation Incentives Model: Game-Theoretic Framework
===============================================================

Formal Game Description:
------------------------
This model implements a Bayesian game with incomplete information where heterogeneous
contributors interact with a platform that distributes monetary rewards.

1. Players:
   - N contributor agents with heterogeneous types (extrinsic, balanced, intrinsic)
   - 1 platform agent that distributes rewards

2. Types:
   - Contributor types θ ∈ {Extrinsic, Balanced, Intrinsic}
   - Skill levels s ∈ {Low, Medium, High}
   - Type distribution follows fixed probabilities (e.g., 30% extrinsic, 50% balanced, 20% intrinsic)
   - Individual parameters (monetary/reputation sensitivity, effort cost) vary within types

3. Actions:
   - Contributors choose effort level e ∈ {No Effort, Low Effort, High Effort}
   - Effort + skill determine probabilistic contribution quality q ∈ {No Contribution, Low, Medium, High}
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

from agents import PlatformAgent, ContributorAgent, ContributionQuality, ContributorType

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
    """Calculate participation rate."""
    contributions = [agent.contribution_history[-1] 
                    for agent in model.agents 
                    if isinstance(agent, ContributorAgent) and agent.contribution_history]
    
    if contributions:   
        participation_rate = sum(c > ContributionQuality.NO_CONTRIBUTION.value for c in contributions) / len(contributions)
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

def compute_med_quality_rate(model):
    """Compute the proportion of active contributors submitting medium quality contributions."""
    contributors = [agent for agent in model.agents if isinstance(agent, ContributorAgent)]
    if not contributors:
        return 0
    
    med_quality_contributors = 0
    for agent in contributors:
        if agent.contribution_history and agent.contribution_history[-1] == ContributionQuality.MED_QUALITY.value:
            med_quality_contributors += 1
    
    return med_quality_contributors / len(contributors)

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

def compute_contribution_by_type(model, contributor_type):
    """Calculate contribution rate for a specific contributor type."""
    type_agents = [agent for agent in model.agents 
                  if isinstance(agent, ContributorAgent) and agent.contributor_type == contributor_type]
    
    if not type_agents:
        return 0
        
    contributions = [agent.contribution_history[-1] for agent in type_agents if agent.contribution_history]
    
    if contributions:
        participation_rate = sum(c > ContributionQuality.NO_CONTRIBUTION.value for c in contributions) / len(contributions)
    else:
        participation_rate = 0
        
    return participation_rate

def compute_extrinsic_participation(model):
    return compute_contribution_by_type(model, ContributorType.EXTRINSIC)

def compute_balanced_participation(model):
    return compute_contribution_by_type(model, ContributorType.BALANCED)

def compute_intrinsic_participation(model):
    return compute_contribution_by_type(model, ContributorType.INTRINSIC)

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
                num_contributors  = 100,
                reward_pool       = 100.0,
                mean_effort_cost  = 1.0, 
                effort_cost_std   = 0.2,
                extrinsic_ratio   = 0.3,
                balanced_ratio    = 0.5,
                intrinsic_ratio   = 0.2,
                reward_strategy   = 'weighted'):
        
        super().__init__()
        
        self.num_contributors = num_contributors

        # Create platform agent with specified reward strategy
        self.platform = PlatformAgent(self, reward_pool, reward_strategy)
        
        type_distribution = []
        type_distribution.extend([ContributorType.EXTRINSIC] * int(num_contributors * extrinsic_ratio))
        type_distribution.extend([ContributorType.BALANCED] * int(num_contributors * balanced_ratio))
        type_distribution.extend([ContributorType.INTRINSIC] * int(num_contributors * intrinsic_ratio))

        # Create contributor agents with varying effort costs
        for contributor_type in type_distribution:
            effort_cost = max(0.1, np.random.normal(mean_effort_cost, effort_cost_std))
            # Initial reputation is drawn from beta distribution (most start low, few start high)
            initial_reputation = np.random.beta(1.5, 5.0)
            _ = ContributorAgent(self, effort_cost, initial_reputation, contributor_type)
            
        # Set up data collection
        model_reporters = {
            "Average Quality": compute_average_quality,
            "Participation Rate": compute_participation_rate,
            "High Quality Rate": compute_high_quality_rate,
            "Med Quality Rate": compute_med_quality_rate,
            "Low Quality Rate": compute_low_quality_rate,
            "No Contribution Rate": compute_no_contribution_rate,
            "Extrinsic Participation": compute_extrinsic_participation,
            "Balanced Participation": compute_balanced_participation,
            "Intrinsic Participation": compute_intrinsic_participation,
            "Reputation Inequality (Gini)": compute_gini_coefficient,
            "Platform Utility": compute_platform_utility,
            "Platform Step Benefit": lambda m: m.platform.current_step_benefit if hasattr(m, 'platform') else 0,
            "Platform Step Cost": lambda m: m.platform.current_step_cost if hasattr(m, 'platform') else 0,
            "Reward Strategy": lambda m: m.platform.get_reward_strategy_info()['name']
        }
        
        agent_reporters = {
            "Reputation": "current_reputation",
            "Contributor Type": lambda agent: agent.contributor_type.name if hasattr(agent, 'contributor_type') else None,
            "Last Contribution": lambda agent: agent.contribution_history[-1] if isinstance(agent, ContributorAgent) and agent.contribution_history else None,
            "Skill Level": lambda agent: agent.skill_level.name if hasattr(agent, 'skill_level') else None,
        }
        
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )
        
    def get_platform(self):
        """Get the platform agent."""
        return self.platform
    
    def step(self):
        """Execute one step of the model."""
        self.datacollector.collect(self)
        self.agents.do("step")