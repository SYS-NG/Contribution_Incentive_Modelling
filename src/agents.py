from mesa import Agent, Model
from enum import Enum
import numpy as np
from reward_distributor import RewardDistributionStrategy, create_strategy

class EffortLevel(Enum):
    NO_EFFORT   = 0
    LOW_EFFORT  = 1
    HIGH_EFFORT = 2

class ContributorType(Enum):
    EXTRINSIC = 0  # Primarily motivated by monetary rewards
    BALANCED  = 1  # Balanced motivation between money and reputation
    INTRINSIC = 2  # Primarily motivated by reputation/intrinsic factors

class SkillLevel(Enum):
    LOW    = 0
    MEDIUM = 1
    HIGH   = 2

class ContributionQuality(Enum):
    NO_CONTRIBUTION = 0
    LOW_QUALITY     = 1
    MED_QUALITY     = 2
    HIGH_QUALITY    = 3

class PlatformAgent(Agent):
    """Platform agent that sets incentive policies."""
    
    def __init__(self, model: Model, reward_pool: int, reward_strategy: str):
        super().__init__(model)
        self.reward_pool     = reward_pool        # Fixed monetary pool per step
        self.total_cost      = 0
        self.total_benefit   = 0
        self.reward_strategy = create_strategy(reward_strategy)
            
        # Track rewards given for each quality over time
        self.reward_history    = {
            ContributionQuality.NO_CONTRIBUTION.value: [],
            ContributionQuality.LOW_QUALITY.value: [],
            ContributionQuality.MED_QUALITY.value: [],
            ContributionQuality.HIGH_QUALITY.value: []
        }
        # Track per-step utility
        self.step_utility = []
        self.current_step_benefit = 0
        self.current_step_cost = 0
            
    def compute_platform_utility(self):
        """Compute platform's utility based on contributions and costs."""
        return self.total_benefit - self.total_cost
    
    def compute_step_utility(self):
        """Compute platform's utility for the current step only."""
        return self.current_step_benefit - self.current_step_cost
    
    def step(self):
        """Platform's step function"""
        
        # Reset step metrics
        self.current_step_benefit = 0
        self.current_step_cost = 0
        
        # Count contributions of each quality
        contribution_counts = {
            ContributionQuality.NO_CONTRIBUTION.value: 0,
            ContributionQuality.LOW_QUALITY.value:     0,
            ContributionQuality.MED_QUALITY.value:     0,
            ContributionQuality.HIGH_QUALITY.value:    0
        }
        
        contributors_with_contributions = []
        
        # Count contributions and calculate benefits
        for agent in self.model.agents:
            if isinstance(agent, ContributorAgent) and agent.contribution_history:
                contribution = agent.contribution_history[-1]
                contribution_counts[contribution] += 1
                if contribution > ContributionQuality.NO_CONTRIBUTION.value:
                    contributors_with_contributions.append(agent)
                
                contribution_value = self.calculate_contribution_value(contribution)
                self.total_benefit += contribution_value
                self.current_step_benefit += contribution_value
        
        # Distribute rewards from the pool based on contribution quality using the strategy
        if contributors_with_contributions:
            # Use the reward distribution strategy to calculate rewards per quality level
            reward_per_quality = self.reward_strategy.distribute_rewards(
                contribution_counts, 
                self.reward_pool
            )
            
            # Store in history for future reference
            for quality, reward in reward_per_quality.items():
                if quality in self.reward_history:
                    self.reward_history[quality].append(reward)
            
            # Distribute rewards to contributors
            for agent in contributors_with_contributions:
                contribution = agent.contribution_history[-1]
                reward = reward_per_quality.get(contribution, 0)
                agent.receive_reward(reward)
                self.total_cost += reward
                self.current_step_cost += reward
        
        # Calculate and store step utility
        step_utility = self.compute_step_utility()
        self.step_utility.append(step_utility)
        
        return {
            'utility': step_utility,
            'contributions': contribution_counts,
            'benefit': self.current_step_benefit,
            'cost': self.current_step_cost
        }
    
    def calculate_reward_cost(self, contribution_quality):
        """Get the current reward for a contribution quality."""
        if not self.reward_history[contribution_quality]:
            # Initial estimate if no history exists
            return 0
        return self.reward_history[contribution_quality][-1]
    
    def calculate_contribution_value(self, contribution_quality):
        """Calculate the value gained from a contribution."""
        if contribution_quality == ContributionQuality.NO_CONTRIBUTION.value:
            return 0
        elif contribution_quality == ContributionQuality.LOW_QUALITY.value:
            return 0.5
        elif contribution_quality == ContributionQuality.MED_QUALITY.value:
            return 1.2
        else:  # HIGH_QUALITY
            return 2.5
            
    def get_historical_rewards(self, contribution_quality):
        """Get historical rewards for a specific contribution quality."""
        return self.reward_history.get(contribution_quality, [])
    
    def get_reward_strategy_info(self):
        """Get information about the current reward strategy."""
        return {
            'name': self.reward_strategy.get_name(),
            'description': self.reward_strategy.get_description()
        }

class ContributorAgent(Agent):
    """
    Contributor agent that makes strategic decisions on effort level and contribution quality.
    
    This class implements a boundedly rational agent in a game-theoretic framework who:
    1. Has heterogeneous preferences based on contributor type (extrinsic, balanced, intrinsic)
    2. Makes effort decisions to maximize expected utility
    3. Has probabilistic contribution quality based on skill level and effort
    4. Forms beliefs about rewards through simplified fictitious play
    5. Adapts strategy based on observed reward history
    
    The agent's decision process models a form of Bayesian Nash Equilibrium where each agent:
    - Chooses the effort level that maximizes expected utility given their beliefs
    - Has type-dependent sensitivities to monetary rewards vs. reputation gains
    - Forms beliefs about reward distribution through observing past outcomes
    - Has incomplete information about other agents' types and decisions
    
    The heterogeneity in the agent population creates strategic diversity:
    - Extrinsic agents (30%): Primarily motivated by monetary rewards
    - Balanced agents (50%): Equal sensitivity to monetary and reputation
    - Intrinsic agents (20%): Primarily motivated by reputation/intrinsic factors
    
    This heterogeneity combined with varying skill levels and adaptive learning
    leads to complex equilibrium dynamics in the model.
    """
    
    def __init__(self, model, effort_cost, initial_reputation, contributor_type=None, skill_level=None):
        """
        Initialize a contributor agent with heterogeneous attributes.
        
        Args:
            model: The model instance this agent belongs to
            effort_cost: Individual cost parameter for high-quality contributions
            initial_reputation: Starting reputation value
            contributor_type: Type determining monetary vs. reputation sensitivity
                             (None for random assignment based on distribution)
            skill_level: Skill level affecting contribution quality probabilities
                        (None for random assignment based on distribution)
        """
        super().__init__(model)
        self.effort_cost          = effort_cost  # γ: cost of high-quality contributions
        self.current_reputation   = initial_reputation
        self.contribution_history = []
        self.cumulative_rewards   = 0
        
        if contributor_type is None:
            self.contributor_type = np.random.choice(list(ContributorType), p=[0.3, 0.5, 0.2])
        else:
            self.contributor_type = contributor_type
            
        if skill_level is None:
            self.skill_level = np.random.choice(list(SkillLevel), p=[0.3, 0.5, 0.2])
        else:
            self.skill_level = skill_level
            
        if self.contributor_type == ContributorType.EXTRINSIC:
            self.monetary_sensitivity   = np.random.uniform(0.8, 1.0)
            self.reputation_sensitivity = np.random.uniform(0.1, 0.4)
        elif self.contributor_type == ContributorType.BALANCED:
            self.monetary_sensitivity   = np.random.uniform(0.4, 0.8)
            self.reputation_sensitivity = np.random.uniform(0.4, 0.8)
        else:  # INTRINSIC
            self.monetary_sensitivity   = np.random.uniform(0.1, 0.4)
            self.reputation_sensitivity = np.random.uniform(0.8, 1.0)
    
    def calculate_reputation_change(self, contribution_quality):
        """Calculate reputation change based on contribution quality."""
        if contribution_quality == ContributionQuality.HIGH_QUALITY.value:
            return 0.08
        elif contribution_quality == ContributionQuality.MED_QUALITY.value:
            return 0.05
        elif contribution_quality == ContributionQuality.LOW_QUALITY.value:
            return 0.02
        else:  # NO_CONTRIBUTION
            return 0
    
    def estimate_monetary_reward(self, contribution_quality):
        """
        Estimate expected monetary reward based on contribution quality and historical reward distribution.
        
        This method implements a simplified fictitious play approach where agents form beliefs
        about expected rewards based on the observed history of rewards for each quality level.
        Fictitious play is a learning process in game theory where players choose best responses
        based on the empirical distribution of other players' past actions.
        
        In this context:
        1. The agent observes historical rewards for each quality level
        2. Forms beliefs about the expected reward for a given quality
        3. Uses weighted averaging that gives more importance to recent observations
           to account for potential changes in the reward distribution strategy
        
        For heterogeneous agents, this adaptive learning is crucial as:
        - Different agent types (extrinsic, balanced, intrinsic) weigh monetary rewards differently
        - Agents with different skill levels have different quality probability distributions
        - The platform's reward strategy may distribute rewards differently as the mix of
          contribution qualities changes over time
        
        Args:
            contribution_quality: The quality level to estimate reward for
            
        Returns:
            Estimated monetary reward for the given contribution quality
        """
        # No reward for no contribution
        if contribution_quality == ContributionQuality.NO_CONTRIBUTION.value:
            return 0
        
        platform = self.model.get_platform()
        historical_rewards = platform.get_historical_rewards(contribution_quality)

        # Fictitious play implementation
        if historical_rewards:
            # Calculate weighted average with more weight on recent observations
            # This balances between:
            # - Learning from historical patterns (stability)
            # - Adapting to recent changes (responsiveness)
            
            if len(historical_rewards) >= 5:
                # With longer history, use weighted recency-biased average
                weights = np.linspace(0.5, 1.0, min(5, len(historical_rewards)))
                weights = weights / np.sum(weights)  # Normalize weights
                
                # Calculate weighted average of recent rewards
                recent_rewards = historical_rewards[-5:]
                expected_reward = np.sum(np.array(recent_rewards) * weights)
                
                # Add small exploration bonus if reward varies significantly
                # (encourages exploring when reward distribution is unstable)
                reward_std = np.std(recent_rewards)
                if reward_std > 0.5:
                    exploration_bonus = 0.1 * reward_std
                    # Adjust based on agent type - extrinsic agents explore more for monetary gain
                    if self.contributor_type == ContributorType.EXTRINSIC:
                        expected_reward += exploration_bonus
            else:
                # With limited history, use simple average
                expected_reward = sum(historical_rewards) / len(historical_rewards)
        else:
            # Fall back to initial estimate if no history
            # Higher quality should reasonably expect higher reward
            expected_reward = contribution_quality
            
        return expected_reward
    
    def calculate_effort_cost(self, effort_level):
        """Calculate effort cost based on effort level."""
        if effort_level == EffortLevel.NO_EFFORT.value:
            return 0
        elif effort_level == EffortLevel.HIGH_EFFORT.value:
            return self.effort_cost
        else:  # LOW_EFFORT
            return self.effort_cost * 0.3
        
    def compute_expected_utility(self, effort_level):
        """
        Compute expected utility for a given effort level with non-linear effects.
        
        This is the core game-theoretic decision function that implements utility maximization
        for heterogeneous agents. Each agent computes the expected utility for each possible
        effort level and chooses the one that maximizes their utility.
        
        The utility function incorporates:
        1. Expected monetary rewards: Weighted by monetary sensitivity (varies by agent type)
        2. Expected reputation gains: Weighted by reputation sensitivity (varies by agent type)
        3. Effort costs: Based on effort level and individual cost parameter
        
        For each possible effort level, the agent:
        - Estimates contribution quality probabilities (based on skill and effort)
        - Calculates expected monetary reward for each possible quality
        - Calculates expected reputation change for each possible quality
        - Computes the weighted expectation across all possible quality outcomes
        - Subtracts the effort cost
        
        The heterogeneity of agents (different types, skills, sensitivities) creates
        strategic diversity in the population, leading to complex equilibrium dynamics
        where different agents may find different strategies optimal.
        
        Args:
            effort_level: The effort level to calculate utility for
            
        Returns:
            Expected utility value for the given effort level
        """
        platform = self.model.get_platform()
        
        if effort_level == EffortLevel.NO_EFFORT.value:
            # Consider the opportunity cost of not contributing (potential loss of reputation)
            reputation_change = self.calculate_reputation_change(ContributionQuality.NO_CONTRIBUTION.value)
            reputation_utility = reputation_change * self.reputation_sensitivity
            return reputation_utility
        
        # Estimate the expected quality based on effort and skill
        expected_quality_probs = self.quality_probabilities(effort_level)
        
        # Calculate expected monetary utility using historical rewards
        expected_monetary_utility = 0
        for quality, prob in expected_quality_probs.items():
            monetary_reward = self.estimate_monetary_reward(quality)
            expected_monetary_utility += prob * monetary_reward * self.monetary_sensitivity
        
        expected_reputation_utility = 0
        for quality, prob in expected_quality_probs.items():
            reputation_change = self.calculate_reputation_change(quality)
            expected_reputation_utility += prob * reputation_change * self.reputation_sensitivity
        
        # Cost based on effort level
        cost = self.calculate_effort_cost(effort_level)
            
        return (expected_monetary_utility + 
                expected_reputation_utility -
                cost)
    
    def quality_probabilities(self, effort_level):
        """
        Estimate probabilities of different quality outcomes based on effort and skill.
        
        This method implements the stochastic mapping from effort decisions to contribution
        quality outcomes, which is a critical component of the game's information structure.
        The probabilistic nature creates:
        
        1. Strategic uncertainty: Agents cannot perfectly predict their contribution quality
        2. Skill differentiation: Higher skill agents have better quality distributions
        3. Effort-quality relationship: Higher effort increases probability of better quality
        
        This probability distribution forms the basis for expected utility calculations in
        the game-theoretic decision model. It creates a strategic trade-off where:
        - High-skill agents have stronger incentives for high effort (better returns)
        - Low-skill agents may find high effort less worthwhile (lower quality improvement)
        - Different agent types will have different optimal effort strategies based on
          their skill level and sensitivity parameters
        
        Args:
            effort_level: The effort level for which to calculate quality probabilities
            
        Returns:
            Dictionary mapping contribution quality levels to probabilities
        """
        if effort_level == EffortLevel.NO_EFFORT.value:
            return {
                ContributionQuality.HIGH_QUALITY.value: 0.0,
                ContributionQuality.MED_QUALITY.value: 0.0,
                ContributionQuality.LOW_QUALITY.value: 0.0,
                ContributionQuality.NO_CONTRIBUTION.value: 1.0
            }
        
        if self.skill_level == SkillLevel.LOW:
            if effort_level == EffortLevel.HIGH_EFFORT.value:
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.05,
                    ContributionQuality.MED_QUALITY.value:     0.15,
                    ContributionQuality.LOW_QUALITY.value:     0.6,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.2
                }
            else:  # LOW_EFFORT
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.0,
                    ContributionQuality.MED_QUALITY.value:     0.05,
                    ContributionQuality.LOW_QUALITY.value:     0.7,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.25
                }
            
        elif self.skill_level == SkillLevel.MEDIUM:
            if effort_level == EffortLevel.HIGH_EFFORT.value:
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.5,
                    ContributionQuality.MED_QUALITY.value:     0.35,
                    ContributionQuality.LOW_QUALITY.value:     0.1,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.05
                }
            else:  # LOW_EFFORT
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.1,
                    ContributionQuality.MED_QUALITY.value:     0.3,
                    ContributionQuality.LOW_QUALITY.value:     0.5,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.1
                }
            
        else:  # HIGH skill
            if effort_level == EffortLevel.HIGH_EFFORT.value:
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.8,
                    ContributionQuality.MED_QUALITY.value:     0.15,
                    ContributionQuality.LOW_QUALITY.value:     0.04,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.01
                }
            else:  # LOW_EFFORT
                return {
                    ContributionQuality.HIGH_QUALITY.value:    0.3,
                    ContributionQuality.MED_QUALITY.value:     0.4,
                    ContributionQuality.LOW_QUALITY.value:     0.25,
                    ContributionQuality.NO_CONTRIBUTION.value: 0.05
                }
    
    def decide_effort_level(self):
        """Decide effort level based on expected utility."""
        utilities = {level.value: self.compute_expected_utility(level.value) for level in list(EffortLevel)}
        
        # Choose the effort level with highest utility
        chosen_level = max(utilities.items(), key=lambda x: x[1])[0]
        return chosen_level
    
    def determine_quality(self, effort_level):
        """Determine the actual quality based on effort level and skill level with randomization."""
        # Get probabilities from quality_probabilities method
        probs = self.quality_probabilities(effort_level)
        
        # Use numpy.random.choice with the probabilities
        return np.random.choice(
            list(probs.keys()),
            p=list(probs.values())
        )
        
    def step(self):
        """Contributor's step function."""

        effort_level = self.decide_effort_level()
        actual_quality = self.determine_quality(effort_level)
        self.contribution_history.append(actual_quality)
        
        return actual_quality
    
    def receive_reward(self, reward):
        """Receive monetary reward."""
        self.cumulative_rewards += reward
        
    def update_reputation(self, reputation_change):
        """Update agent's reputation."""
        self.current_reputation += reputation_change 