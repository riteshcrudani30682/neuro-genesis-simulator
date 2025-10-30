#!/usr/bin/env python3
"""
Test script for reinforcement learning functionality
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuro_genesis_sim import get_state, choose_action, compute_reward, QNetwork, device, INP_DIM

def test_rl_components():
    """Test reinforcement learning components"""
    print("Testing reinforcement learning components...")
    
    # Test Q-network with correct input dimension
    q_network = QNetwork(INP_DIM, action_dim=4)  # Use the actual input dimension
    q_network = q_network.to(device)
    
    # Test state representation with correct dimensions
    state = np.zeros(INP_DIM, dtype=np.float32)
    print(f"State shape: {state.shape}")
    
    # Test action selection
    action = choose_action(state, epsilon=0.0)  # No exploration
    print(f"Chosen action: {action}")
    assert 0 <= action <= 3, "Action should be between 0 and 3"
    
    # Test reward computation
    reward = compute_reward(None, 0.5, 0.7)  # Positive change
    print(f"Reward for positive change: {reward}")
    
    reward = compute_reward(None, 0.7, 0.5)  # Negative change
    print(f"Reward for negative change: {reward}")
    
    # Test optimal range reward
    reward = compute_reward(None, 0.4, 0.6)  # In optimal range
    print(f"Reward for optimal range: {reward}")
    
    print("Reinforcement learning test passed!")

if __name__ == "__main__":
    test_rl_components()