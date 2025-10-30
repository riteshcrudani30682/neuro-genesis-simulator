#!/usr/bin/env python3
"""
Test script for reinforcement learning integration
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuro_genesis_sim import get_state, choose_action, compute_reward, update_q_network, QNetwork, device, INP_DIM

def test_rl_integration():
    """Test full reinforcement learning integration"""
    print("Testing reinforcement learning integration...")
    
    # Test Q-network
    q_network = QNetwork(INP_DIM, action_dim=4)
    q_network = q_network.to(device)
    
    # Create a simple state
    state = np.random.rand(INP_DIM).astype(np.float32)
    next_state = np.random.rand(INP_DIM).astype(np.float32)
    
    # Test action selection
    action = choose_action(state, epsilon=0.0)  # No exploration
    print(f"Chosen action: {action}")
    assert 0 <= action <= 3, "Action should be between 0 and 3"
    
    # Test reward computation
    reward = compute_reward(None, 0.3, 0.7)  # Positive change
    print(f"Reward: {reward}")
    assert reward > 0, "Reward should be positive for positive activation change"
    
    # Test Q-network update
    loss = update_q_network(state, action, reward, next_state)
    print(f"Loss after Q-network update: {loss}")
    assert loss >= 0, "Loss should be non-negative"
    
    print("Reinforcement learning integration test passed!")

if __name__ == "__main__":
    test_rl_integration()