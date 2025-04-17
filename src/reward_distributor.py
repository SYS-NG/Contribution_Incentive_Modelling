from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np

class RewardDistributionStrategy(ABC):
    """
    Abstract base class for reward distribution strategies.
    
    This class defines the interface for all reward distribution strategies.
    Each strategy must implement the distribute_rewards method, which takes
    contribution counts and a reward pool and returns a mapping of
    contribution quality to reward amounts.
    """
    
    @abstractmethod
    def distribute_rewards(self, 
                         contribution_counts: Dict[int, int], 
                         reward_pool: float,
                         **kwargs) -> Dict[int, float]:
        """
        Distribute rewards based on contribution counts and available reward pool.
        
        Args:
            contribution_counts: Dictionary mapping contribution quality levels to counts
            reward_pool: Total available rewards to distribute
            **kwargs: Additional parameters specific to the distribution strategy
            
        Returns:
            Dictionary mapping contribution quality levels to reward amounts
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the distribution strategy."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of how the distribution strategy works."""
        pass


class WeightedDistributionStrategy(RewardDistributionStrategy):
    """
    Default weighted distribution strategy used in the original model.
    
    Distributes rewards based on fixed weights for each quality level,
    with higher quality contributions receiving disproportionately higher rewards.
    """
    
    def __init__(self, weights: Optional[Dict[int, float]] = None):
        """
        Initialize with quality weights.
        
        Args:
            weights: Optional dictionary mapping quality levels to weights.
                     If None, default weights will be used.
        """
        # Default weights from the original model
        self.weights = weights or {
            0: 0,    # No contribution
            1: 1,    # Low quality
            2: 5     # High quality
        }
    
    def distribute_rewards(self, 
                         contribution_counts: Dict[int, int], 
                         reward_pool: float,
                         **kwargs) -> Dict[int, float]:
        """
        Distribute rewards using a weighted approach.
        
        Args:
            contribution_counts: Dictionary mapping contribution quality levels to counts
            reward_pool: Total available rewards to distribute
            
        Returns:
            Dictionary mapping contribution quality levels to reward amounts per contribution
        """
        # Calculate total weighting
        total_weight = 0
        for quality, count in contribution_counts.items():
            total_weight += self.weights.get(quality, 0) * count
        
        # Calculate reward per quality level
        reward_per_quality = {}
        for quality in contribution_counts:
            if total_weight > 0:
                reward_per_quality[quality] = (self.weights.get(quality, 0) / total_weight) * reward_pool
            else:
                reward_per_quality[quality] = 0
                
        return reward_per_quality
    
    def get_name(self) -> str:
        return "Weighted Distribution"
    
    def get_description(self) -> str:
        return "Distributes rewards based on fixed weights, with higher quality receiving higher rewards."


class LinearDistributionStrategy(RewardDistributionStrategy):
    """
    Linear reward distribution strategy.
    
    Distributes rewards in direct proportion to contribution quality level.
    """
    
    def distribute_rewards(self, 
                         contribution_counts: Dict[int, int], 
                         reward_pool: float,
                         **kwargs) -> Dict[int, float]:
        """
        Distribute rewards using a linear approach based on quality level.
        
        Args:
            contribution_counts: Dictionary mapping contribution quality levels to counts
            reward_pool: Total available rewards to distribute
            
        Returns:
            Dictionary mapping contribution quality levels to reward amounts per contribution
        """
        # Calculate total quality points
        total_quality_points = 0
        for quality, count in contribution_counts.items():
            total_quality_points += quality * count
        
        # Calculate reward per quality level - directly proportional to quality
        reward_per_quality = {}
        for quality in contribution_counts:
            if total_quality_points > 0 and quality > 0:
                reward_per_quality[quality] = (quality / total_quality_points) * reward_pool
            else:
                reward_per_quality[quality] = 0
                
        return reward_per_quality
    
    def get_name(self) -> str:
        return "Linear Distribution"
    
    def get_description(self) -> str:
        return "Distributes rewards in direct proportion to contribution quality level."


class ThresholdDistributionStrategy(RewardDistributionStrategy):
    """
    Threshold-based reward distribution strategy.
    
    Distributes rewards only to contributions that meet or exceed a quality threshold.
    """
    
    def __init__(self, threshold: int = 2):
        """
        Initialize with threshold quality level.
        
        Args:
            threshold: Minimum quality level required to receive rewards (default: 2 - high quality)
        """
        self.threshold = threshold
    
    def distribute_rewards(self, 
                         contribution_counts: Dict[int, int], 
                         reward_pool: float,
                         **kwargs) -> Dict[int, float]:
        """
        Distribute rewards using a threshold approach.
        
        Args:
            contribution_counts: Dictionary mapping contribution quality levels to counts
            reward_pool: Total available rewards to distribute
            
        Returns:
            Dictionary mapping contribution quality levels to reward amounts per contribution
        """
        # Calculate total eligible contributions
        eligible_contributions = 0
        for quality, count in contribution_counts.items():
            if quality >= self.threshold:
                eligible_contributions += count
        
        # Calculate reward per quality level
        reward_per_quality = {}
        for quality in contribution_counts:
            if eligible_contributions > 0 and quality >= self.threshold:
                # Equal distribution among all eligible contributions
                reward_per_quality[quality] = reward_pool / eligible_contributions
            else:
                reward_per_quality[quality] = 0
                
        return reward_per_quality
    
    def get_name(self) -> str:
        return f"Threshold Distribution (min: {self.threshold})"
    
    def get_description(self) -> str:
        return f"Distributes rewards only to contributions that meet or exceed a threshold quality of {self.threshold}."

class ReputationWeightedStrategy(RewardDistributionStrategy):
    """
    Distributes rewards based on both contribution quality and contributor reputation.
    
    This strategy balances quality-based rewards with reputation influence,
    creating a virtuous cycle where high-reputation contributors receive 
    relatively higher rewards for the same quality contributions.
    """
    
    def __init__(self, quality_weight: float = 0.7, reputation_weight: float = 0.3):
        """
        Initialize with weights for quality vs. reputation importance.
        
        Args:
            quality_weight: Weight given to contribution quality (0 to 1)
            reputation_weight: Weight given to contributor reputation (0 to 1)
                              Note: quality_weight + reputation_weight should equal 1.0
        """
        self.quality_weight = quality_weight
        self.reputation_weight = reputation_weight
        
        # Quality weights similar to the weighted distribution
        self.quality_weights = {
            0: 0,    # No contribution
            1: 1,    # Low quality
            2: 5     # High quality
        }
    
    def distribute_rewards(self, 
                         contribution_counts: Dict[int, int], 
                         reward_pool: float,
                         **kwargs) -> Dict[int, float]:
        """
        Distribute rewards using both quality and reputation.
        
        Args:
            contribution_counts: Dictionary mapping contribution quality levels to counts
            reward_pool: Total available rewards to distribute
            kwargs:
                contributors: List of contributors with their qualities and reputations
                
        Returns:
            Dictionary mapping contribution quality levels to reward amounts per contribution
        """
        # If no contributor data provided, fall back to quality-only distribution
        if 'contributors' not in kwargs:
            # Calculate total weighting based on quality only
            total_weight = 0
            for quality, count in contribution_counts.items():
                total_weight += self.quality_weights.get(quality, 0) * count
            
            # Calculate reward per quality level
            reward_per_quality = {}
            for quality in contribution_counts:
                if total_weight > 0:
                    reward_per_quality[quality] = (self.quality_weights.get(quality, 0) / total_weight) * reward_pool
                else:
                    reward_per_quality[quality] = 0
            
            return reward_per_quality
        
        # Get contributors data (list of tuples: (agent_id, quality, reputation))
        contributors = kwargs['contributors']
        
        # Quality base calculation - similar to weighted distribution
        quality_weights = {agent_id: self.quality_weights.get(quality, 0) 
                          for agent_id, quality, _ in contributors}
        
        # Calculate reputation-adjusted weights
        adjusted_weights = {}
        total_adjusted_weight = 0
        
        for agent_id, quality, reputation in contributors:
            # Skip no-contribution entries
            if quality == 0:
                adjusted_weights[agent_id] = 0
                continue
                
            # Calculate combined weight
            quality_component = self.quality_weight * self.quality_weights.get(quality, 0)
            reputation_component = self.reputation_weight * max(1, reputation)  # Minimum reputation impact of 1
            
            adjusted_weights[agent_id] = quality_component + reputation_component
            total_adjusted_weight += adjusted_weights[agent_id]
        
        # Calculate per-quality rewards based on the agents of each quality level
        reward_per_quality = {0: 0, 1: 0, 2: 0}  # Initialize all qualities
        contribution_counts_by_quality = {0: 0, 1: 0, 2: 0}  # Track how many of each quality
        
        # Calculate per-agent rewards first
        agent_rewards = {}
        for agent_id, quality, _ in contributors:
            if total_adjusted_weight > 0 and quality > 0:
                agent_reward = (adjusted_weights[agent_id] / total_adjusted_weight) * reward_pool
                agent_rewards[agent_id] = agent_reward
                
                # Add to the quality's total
                reward_per_quality[quality] += agent_reward
                contribution_counts_by_quality[quality] += 1
            else:
                agent_rewards[agent_id] = 0
        
        # Convert to per-contribution rewards for each quality
        for quality in reward_per_quality:
            if contribution_counts_by_quality[quality] > 0:
                reward_per_quality[quality] /= contribution_counts_by_quality[quality]
        
        # Store agent-specific rewards for potential use
        kwargs['agent_rewards'] = agent_rewards
                
        return reward_per_quality
    
    def get_name(self) -> str:
        return f"Reputation-Weighted (Q:{self.quality_weight:.1f}, R:{self.reputation_weight:.1f})"
    
    def get_description(self) -> str:
        return "Distributes rewards based on both contribution quality and contributor reputation."

# Factory function to create distribution strategies
def create_strategy(strategy_type: str, **kwargs) -> RewardDistributionStrategy:
    """
    Factory function to create a reward distribution strategy.
    
    Args:
        strategy_type: Type of strategy to create ('weighted', 'linear', 'threshold')
        **kwargs: Additional parameters for the strategy
        
    Returns:
        An instance of the specified RewardDistributionStrategy
        
    Raises:
        ValueError: If the strategy type is not recognized
    """
    if strategy_type.lower() == 'weighted':
        weights = kwargs.get('weights', None)
        return WeightedDistributionStrategy(weights)
    elif strategy_type.lower() == 'linear':
        return LinearDistributionStrategy()
    elif strategy_type.lower() == 'threshold':
        threshold = kwargs.get('threshold', 2)
        return ThresholdDistributionStrategy(threshold)
    elif strategy_type.lower() == 'reputation_weighted':
        quality_weight = kwargs.get('quality_weight', 0.7)
        reputation_weight = kwargs.get('reputation_weight', 0.3)
        return ReputationWeightedStrategy(quality_weight, reputation_weight)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}") 