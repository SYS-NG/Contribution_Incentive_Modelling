from mesa import Agent, Model
from enum import Enum
import numpy as np
from reward_distributor import RewardDistributionStrategy, create_strategy

class ParticipationDecision(Enum):
    ABSTAIN    = 0
    CONTRIBUTE = 1

class ContributorType(Enum):
    EXTRINSIC = 0  # Primarily motivated by monetary rewards
    INTRINSIC = 1  # Primarily motivated by reputation/intrinsic factors

class SkillLevel(Enum):
    LOW  = 0
    HIGH = 1

class ContributionQuality(Enum):
    NO_CONTRIBUTION = 0
    LOW_QUALITY     = 1
    HIGH_QUALITY    = 2

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
        
        # Prepare contributor data for reputation-aware strategies
        contributors_data = []
        for agent in contributors_with_contributions:
            if isinstance(agent, ContributorAgent) and agent.contribution_history:
                contributors_data.append(
                    (agent.unique_id, agent.contribution_history[-1], agent.current_reputation)
                )

        # Use the reward distribution strategy to calculate rewards per quality level
        reward_per_quality = self.reward_strategy.distribute_rewards(
            contribution_counts, 
            self.reward_pool,
            contributors=contributors_data
        )

        # Check if strategy returned agent-specific rewards
        agent_specific_rewards = getattr(self.reward_strategy, 'agent_rewards', None)

        # Store in history for future reference
        for quality, reward in reward_per_quality.items():
            if quality in self.reward_history:
                self.reward_history[quality].append(reward)

        # Distribute rewards to contributors
        for agent in contributors_with_contributions:
            contribution = agent.contribution_history[-1]
            
            # Use agent-specific reward if available, otherwise use quality-based reward
            if agent_specific_rewards and agent.unique_id in agent_specific_rewards:
                reward = agent_specific_rewards[agent.unique_id]
            else:
                reward = reward_per_quality.get(contribution, 0)
            
            agent.receive_reward(reward)
            self.total_cost += reward
            self.current_step_cost += reward
        
        # Calculate and store step utility
        step_utility = self.compute_step_utility()
        self.step_utility.append(step_utility)
        
        # After rewards are distributed, have each contributor finalize their step
        for agent in self.model.agents:
            if isinstance(agent, ContributorAgent):
                agent.finalize_step()
        
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
            return 1
        else:  # HIGH_QUALITY
            return 3
            
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
    Contributor agent that makes strategic decisions on participation and contribution quality.
    
    This class implements a boundedly rational agent in a game-theoretic framework who:
    1. Has heterogeneous preferences based on contributor type (extrinsic, intrinsic)
    2. Makes participation decisions to maximize expected utility
    3. Has probabilistic contribution quality based on skill level
    4. Forms beliefs about rewards through simplified fictitious play
    5. Adapts strategy based on observed reward history
    
    The agent's decision process models a form of Bayesian Nash Equilibrium where each agent:
    - Chooses whether to contribute based on maximizing expected utility given their beliefs
    - Has type-dependent sensitivities to monetary rewards vs. reputation gains
    - Forms beliefs about reward distribution through observing past outcomes
    - Has incomplete information about other agents' types and decisions
    
    The heterogeneity in the agent population creates strategic diversity:
    - Extrinsic agents (50%): Primarily motivated by monetary rewards
    - Intrinsic agents (50%): Primarily motivated by reputation/intrinsic factors
    
    This heterogeneity combined with varying skill levels and adaptive learning
    leads to complex equilibrium dynamics in the model.
    """
    
    def __init__(self, model, effort_cost, initial_reputation, contributor_type, skill_level,
                 phi, delta, lambda_param, reward_learning_rate, use_ewa):
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
            phi: EWA parameter - decay factor for experience (0 to 1)
                 Controls how quickly previous experience depreciates
            delta: EWA parameter - weight on foregone payoffs (0 to 1)
                   Higher values place more emphasis on what could have been earned
            lambda_param: EWA parameter - softmax sharpness parameter
                          Low values (0.5-1) create more exploration (choice probabilities closer)
                          High values (>5) create exploitation (winner-take-all probabilities)
            reward_learning_rate: Learning rate for updating reward estimates (0 to 1)
                                 Controls how quickly agents adapt to new reward information
                                 Higher values adapt faster but with more volatility
            use_ewa: Whether to use Experience-Weighted Attraction learning (True) or
                    direct utility maximization (False)
        """
        super().__init__(model)
        self.effort_cost          = effort_cost  # γ: cost of high-quality contributions
        self.current_reputation   = initial_reputation
        self.contribution_history = []
        self.participation_decision_history = []  # Track actual participation decisions
        self.cumulative_rewards   = 0
        self.use_ewa              = use_ewa  # Whether to use EWA learning or direct utility max
        
        # EWA learning parameters
        self.phi = phi                    # Experience decay factor
        self.delta = delta                # Weight on foregone payoffs
        self.lambda_param = lambda_param  # Softmax sharpness parameter
        self.reward_learning_rate = reward_learning_rate  # Learning rate for reward estimates
        
        # Initialize EWA attractions and experience
        self.attractions = {
            ParticipationDecision.ABSTAIN.value: 0.0,  
            ParticipationDecision.CONTRIBUTE.value: 0.0
        }
        self.experience = 1.0  # Initial experience weight
        
        # Keep track of reward estimates for each quality level
        self.reward_estimates = {
            ContributionQuality.NO_CONTRIBUTION.value: 0,
            ContributionQuality.LOW_QUALITY.value: 1.0,
            ContributionQuality.HIGH_QUALITY.value: 2.0
        }
        
        # Track the latest decision and reward for EWA updates
        self.last_decision = None
        self.last_reward = 0
        self.last_utilities = {}  # Store utilities from last decision
        
        if contributor_type is None:
            self.contributor_type = np.random.choice(list(ContributorType), p=[0.5, 0.5])
        else:
            self.contributor_type = contributor_type
            
        if skill_level is None:
            self.skill_level = np.random.choice(list(SkillLevel), p=[0.5, 0.5])
        else:
            self.skill_level = skill_level
            
        if self.contributor_type == ContributorType.EXTRINSIC:
            self.monetary_sensitivity   = np.random.uniform(0.7, 1.0)
        else:  # INTRINSIC
            self.monetary_sensitivity   = np.random.uniform(0, 0.3)
        
        self.reputation_sensitivity = 1 - self.monetary_sensitivity
    
    def calculate_reputation_change(self, contribution_quality):
        """Calculate reputation change based on contribution quality."""
        if contribution_quality == ContributionQuality.HIGH_QUALITY.value:
            return 2
        elif contribution_quality == ContributionQuality.LOW_QUALITY.value:
            return 1
        else:  # NO_CONTRIBUTION
            return 0
    
    def estimate_monetary_reward(self, contribution_quality):
        """
        Estimate expected monetary reward based on contribution quality using exponential recency-weighting.
        
        This method implements exponential recency-weighted averaging where more recent
        rewards have exponentially higher influence than older rewards. This is controlled
        by the reward_learning_rate parameter:
        1. The agent retrieves the latest observed reward for the quality level
        2. Updates its estimate using: new_estimate = (1 - L) * old_estimate + L * new_reward
           where L is the reward_learning_rate
        3. Higher learning rate means faster adaptation to recent rewards, but more volatility
        4. Lower learning rate means more stable estimates but slower adaptation
        
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

        # If there's reward history for this quality level
        if historical_rewards:
            # Get the most recent observed reward
            latest_reward = historical_rewards[-1]
            
            # Update estimate using exponential recency-weighting formula
            current_estimate = self.reward_estimates.get(contribution_quality, contribution_quality)
            updated_estimate = (1 - self.reward_learning_rate) * current_estimate + self.reward_learning_rate * latest_reward
            
            # Store the updated estimate
            self.reward_estimates[contribution_quality] = updated_estimate
                    
            return updated_estimate
        else:
            # If no history, use the initial estimate (same as quality level)
            self.reward_estimates[contribution_quality] = contribution_quality
            return contribution_quality
    
    def calculate_effort_cost(self, participation_decision):
        """Calculate effort cost based on participation decision."""
        if participation_decision == ParticipationDecision.ABSTAIN.value:
            return 0
        else:  # CONTRIBUTE
            return self.effort_cost
        
    def compute_expected_utility(self, participation_decision):
        """
        Compute expected utility for a given participation decision.
        
        This is the core game-theoretic decision function that implements utility maximization
        for heterogeneous agents. Each agent computes the expected utility for each possible
        participation decision and chooses the one that maximizes their utility.
        
        The utility function incorporates:
        1. Expected monetary rewards: Weighted by monetary sensitivity (varies by agent type)
        2. Expected reputation gains: Weighted by reputation sensitivity (varies by agent type)
        3. Effort costs: Based on participation decision and individual cost parameter
        
        For each possible participation decision, the agent:
        - Estimates contribution quality probabilities (based on skill)
        - Calculates expected monetary reward for each possible quality
        - Calculates expected reputation change for each possible quality
        - Computes the weighted expectation across all possible quality outcomes
        - Subtracts the effort cost
        
        The heterogeneity of agents (different types, skills, sensitivities) creates
        strategic diversity in the population, leading to complex equilibrium dynamics
        where different agents may find different strategies optimal.
        
        Args:
            participation_decision: The participation decision to calculate utility for
            
        Returns:
            Expected utility value for the given participation decision
        """
        platform = self.model.get_platform()
        
        if participation_decision == ParticipationDecision.ABSTAIN.value:
            # Consider the opportunity cost of not contributing (potential loss of reputation)
            reputation_change = self.calculate_reputation_change(ContributionQuality.NO_CONTRIBUTION.value)
            reputation_utility = reputation_change * self.reputation_sensitivity
            return reputation_utility
        
        # Estimate the expected quality based on skill
        expected_quality_probs = self.quality_probabilities(participation_decision)
        
        # Calculate expected monetary utility using historical rewards
        expected_monetary_utility = 0
        for quality, prob in expected_quality_probs.items():
            monetary_reward = self.estimate_monetary_reward(quality)
            expected_monetary_utility += prob * monetary_reward * self.monetary_sensitivity
        
        expected_reputation_utility = 0
        for quality, prob in expected_quality_probs.items():
            reputation_change = self.calculate_reputation_change(quality)
            expected_reputation_utility += prob * reputation_change * self.reputation_sensitivity
        
        # Cost based on participation
        cost = self.calculate_effort_cost(participation_decision)
            
        return (expected_monetary_utility + 
                expected_reputation_utility -
                cost)
    
    def quality_probabilities(self, participation_decision):
        """
        Estimate probabilities of different quality outcomes based on participation and skill.
        
        This method implements the stochastic mapping from participation decisions to contribution
        quality outcomes, which is a critical component of the game's information structure.
        The probabilistic nature creates:
        
        1. Strategic uncertainty: Agents cannot perfectly predict their contribution quality
        2. Skill differentiation: Higher skill agents have better quality distributions
        
        This probability distribution forms the basis for expected utility calculations in
        the game-theoretic decision model. It creates a strategic trade-off where:
        - High-skill agents have stronger incentives to contribute (better returns)
        - Low-skill agents may find contributing less worthwhile (lower quality outcomes)
        - Different agent types will have different optimal strategies based on
          their skill level and sensitivity parameters
        
        Args:
            participation_decision: The participation decision for which to calculate quality probabilities
            
        Returns:
            Dictionary mapping contribution quality levels to probabilities
        """
        if participation_decision == ParticipationDecision.ABSTAIN.value:
            return {
                ContributionQuality.HIGH_QUALITY.value: 0.0,
                ContributionQuality.LOW_QUALITY.value: 0.0,
                ContributionQuality.NO_CONTRIBUTION.value: 1.0
            }
        
        if self.skill_level == SkillLevel.LOW:
            return {
                ContributionQuality.HIGH_QUALITY.value:    0.2,
                ContributionQuality.LOW_QUALITY.value:     0.6,
                ContributionQuality.NO_CONTRIBUTION.value: 0.2
            }
        else:  # HIGH skill
            return {
                ContributionQuality.HIGH_QUALITY.value:    0.7,
                ContributionQuality.LOW_QUALITY.value:     0.28,
                ContributionQuality.NO_CONTRIBUTION.value: 0.02
            }
    
    def decide_participation(self):
        """
        Decide whether to participate based on expected utility maximization or EWA learning.
        
        If use_ewa is True, uses Experience-Weighted Attraction (EWA) learning with softmax decision rule.
        If use_ewa is False, directly maximizes expected utility.
        
        Returns:
            The chosen participation decision (0 for abstain, 1 for contribute)
        """
        # Compute utilities for each possible decision
        # We'll save these for the EWA update in finalize_step
        self.last_utilities = {
            decision.value: self.compute_expected_utility(decision.value) 
            for decision in list(ParticipationDecision)
        }
        
        if self.use_ewa:
            # EWA LEARNING APPROACH
            # Get the current attraction values
            attraction_values = list(self.attractions.values())
            
            # Apply softmax with lambda parameter to get probabilities
            # Higher lambda = more exploitation, lower = more exploration
            exp_attractions = np.exp(np.array(attraction_values) * self.lambda_param)
            probabilities = exp_attractions / np.sum(exp_attractions)
            
            # Choose decision based on probabilities
            decisions = list(self.attractions.keys())
            chosen_decision = np.random.choice(decisions, p=probabilities)
        else:
            # DIRECT UTILITY MAXIMIZATION APPROACH
            # Simply choose the decision with the highest expected utility
            chosen_decision = max(self.last_utilities, key=self.last_utilities.get)
        
        # Save the chosen decision for EWA update
        self.last_decision = chosen_decision
        
        return chosen_decision
    
    def determine_quality(self, participation_decision):
        """Determine the actual quality based on participation decision and skill level with randomization."""
        # Get probabilities from quality_probabilities method
        probs = self.quality_probabilities(participation_decision)
        
        # Use numpy.random.choice with the probabilities
        return np.random.choice(
            list(probs.keys()),
            p=list(probs.values())
        )
        
    def step(self):
        """Contributor's step function."""
        participation_decision = self.decide_participation()
        self.participation_decision_history.append(participation_decision)  # Store participation decision
        actual_quality = self.determine_quality(participation_decision)
        self.contribution_history.append(actual_quality)
        
        return actual_quality
    
    def receive_reward(self, reward):
        """Receive monetary reward."""
        self.cumulative_rewards += reward
        self.last_reward = reward  # Store for EWA update
    
    def finalize_step(self):
        """
        Update attractions after rewards have been distributed.
        This implements the Experience-Weighted Attraction (EWA) learning update.
        
        EWA combines reinforcement learning and belief-based learning in a single framework:
        1. phi: Controls how much previous experience is discounted
        2. delta: Weight placed on foregone payoffs (counterfactual reasoning)
        3. lambda: Controls the sensitivity to attractions (exploration vs. exploitation)
        """
        if self.last_decision is None:
            # Skip if there's no decision to update
            return
            
        # Only update EWA attractions if we're using EWA learning
        if self.use_ewa:
            # Update experience
            self.experience = self.phi * (1 - self.delta) * self.experience + 1
            
            # For each possible action, update the attraction
            for action, utility in self.last_utilities.items():
                # Calculate the actual utility for the chosen action
                # For the chosen action, we use the actual reward received
                if action == self.last_decision:
                    # For the chosen action, use actual reward
                    if action == ParticipationDecision.CONTRIBUTE.value:
                        # Adjust utility with actual reward instead of expected reward
                        # First compute base utility without monetary reward
                        reputation_utility = 0
                        for quality, prob in self.quality_probabilities(action).items():
                            reputation_change = self.calculate_reputation_change(quality)
                            reputation_utility += prob * reputation_change * self.reputation_sensitivity
                        
                        # Use actual reward instead of expected
                        actual_utility = (self.last_reward * self.monetary_sensitivity + 
                                        reputation_utility - 
                                        self.calculate_effort_cost(action))
                    else:
                        # For ABSTAIN, use the calculated utility
                        actual_utility = utility
                    
                    # Update attraction for chosen action
                    self.attractions[action] = (
                        (self.phi * self.experience * self.attractions[action] + actual_utility) / 
                        self.experience
                    )
                else:
                    # For unchosen actions, update based on foregone payoffs
                    # Delta controls weight on unchosen actions (counterfactual reasoning)
                    self.attractions[action] = (
                        (self.phi * self.experience * self.attractions[action] + self.delta * utility) / 
                        self.experience
                    )
        
        # Reset for next step
        self.last_decision = None
        self.last_reward = 0
    
    def update_reputation(self, reputation_change):
        """Update agent's reputation."""
        self.current_reputation += reputation_change 