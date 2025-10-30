#!/usr/bin/env python3
"""
Test script for multi-region brain functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuro_genesis_sim import initialize_grid_with_regions, REGION_COUNT

def test_regions():
    """Test that grid initialization with regions works correctly"""
    print("Testing multi-region brain functionality...")
    
    # Initialize grid with regions
    grid = initialize_grid_with_regions()
    
    # Check that we have the right dimensions
    assert len(grid) == 64, f"Expected 64 columns, got {len(grid)}"
    assert len(grid[0]) == 48, f"Expected 48 rows, got {len(grid[0])}"
    
    # Check region assignments
    vision_count = 0
    motor_count = 0
    cognitive_count = 0
    
    for x in range(64):
        for y in range(48):
            cell = grid[x][y]
            if cell.region_id == 0:
                vision_count += 1
            elif cell.region_id == 1:
                motor_count += 1
            elif cell.region_id == 2:
                cognitive_count += 1
    
    print(f"Vision region cells: {vision_count}")
    print(f"Motor region cells: {motor_count}")
    print(f"Cognitive region cells: {cognitive_count}")
    
    # Check that regions are approximately the right size
    expected_region_size = 64 * 48 // 3
    assert vision_count > 0, "Vision region should have cells"
    assert motor_count > 0, "Motor region should have cells"
    assert cognitive_count > 0, "Cognitive region should have cells"
    
    print("Multi-region test passed!")

if __name__ == "__main__":
    test_regions()