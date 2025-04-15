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
            2: 3,    # Medium quality
            3: 7     # High quality
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
            threshold: Minimum quality level required to receive rewards (default: 2 - medium quality)
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

class ReputationFocusedDistributionStrategy(RewardDistributionStrategy):
    """
    Threshold-based reward distribution strategy.
    
    Distributes rewards only to contributions that meet or exceed a quality threshold.
    """
    
    def __init__(self, threshold: int = 2):
        """
        Initialize with threshold quality level.
        """
    
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
        # TODO: reputation focused strat
        pass
    
    def get_name(self) -> str:
        return f"Reputation Focused Distribution"
    
    def get_description(self) -> str:
        return f"Distributes rewards with heavier emphasis of high reputation individuals"

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
    elif strategy_type.lower() == 'rep_focused':
        return ReputationFocusedDistributionStrategy()
    elif strategy_type.lower() == 'threshold':
        threshold = kwargs.get('threshold', 2)
        return ThresholdDistributionStrategy(threshold)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}") 