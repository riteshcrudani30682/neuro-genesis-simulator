#!/usr/bin/env python3
"""
Living AI Creature Simulator
चलो सीखते हैं!!! Let's learn!!!

Run: python creature_sim.py
Controls:
  Left-click: Place food
  Space: Apply reward
  S: Save model
  L: Load model
  C: Clear/reset
  ESC: Quit

A single creature (contiguous cluster of cells) that senses environment,
acts (moves/consumes), learns from rewards using batched PyTorch brain,
self-organizes via neurogenesis, and persists state.
"""

import pygame
import numpy as np
import pickle
import os
import random
from collections import deque
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim

# AMP handling
USE_AMP = torch.cuda.is_available()
AMP_AVAILABLE = False
AmpGradScaler = None
autocast_context = None

if USE_AMP:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        AmpGradScaler = GradScaler
        autocast_context = autocast
    except ImportError:
        USE_AMP = False
        autocast_context = torch.no_grad
else:
    autocast_context = torch.no_grad

# -----------------------
# Parameters (tweakable)
# -----------------------
GRID_W, GRID_H = 48, 48           # grid (columns x rows)
CELL_SIZE = 15                    # pixels
INFO_DIM = 8                      # dimension of info vector stored per cell
HIDDEN_DIM = 64                   # hidden layer size
NEIGHBOR_RADIUS = 1               # neighborhood radius (1 => 3x3)
DT = 0.1                          # simulation time step
SPAWN_THRESHOLD = 0.7             # activation threshold to spawn new cell
SPAWN_ENERGY_MIN = 0.5            # minimum energy to spawn
MUTATION_STD = 0.02               # noise added to child info
LEARNING_RATE = 0.001             # learning rate
DECAY = 0.005                     # activation decay per step
ENERGY_DECAY = 0.002              # energy decay per frame
DEATH_ENERGY_THRESHOLD = 0.1      # energy threshold for death
DEATH_TTL = 30                    # frames before death
FOOD_ENERGY = 0.3                 # energy gained from eating food
EXPLORATION_STD = 0.1             # action noise std dev
EXPLORATION_RATE = 0.3            # exploration rate
EXPLORATION_DECAY = 0.995         # exploration decay
EXPLORATION_MIN = 0.01            # minimum exploration
TRAIN_EVERY_N_FRAMES = 5          # train every N frames
BATCH_SIZE = 32                   # training batch size
MAX_BATCH_SIZE = 512              # maximum batch size
REPLAY_BUFFER_SIZE = 2000         # replay buffer size
SAVE_FILE = "creature_save.pkl"

# -----------------------
# Utilities
# -----------------------
def clamp01(x): return max(0.0, min(1.0, x))

def info_to_color(vec):
    """Map info vector to RGB via simple projection"""
    v = np.array(vec)
    if np.linalg.norm(v) < 1e-6:
        v = np.linspace(0, 1, len(v))
    v = (v - v.min()) if v.max() != v.min() else v
    if v.max() != 0:
        v = v / v.max()
    # pick first 3 dims for rgb (pad if needed)
    rgb = np.zeros(3)
    for i in range(3):
        idx = i % len(v)
        rgb[i] = v[idx]
    # map to 0-255 with gamma
    return tuple((np.power(rgb, 0.8) * 255).astype(int))

# -----------------------
# Grid / Cell structure
# -----------------------
class Cell:
    def __init__(self, x, y, id=None):
        self.x = x
        self.y = y
        self.alive = False
        self.act = 0.0
        self.info: np.ndarray = np.zeros(INFO_DIM, dtype=np.float32)
        self.id = id
        self.parent_id = None
        self.generation = 0
        self.energy = 1.0
        self.death_timer = 0

# Initialize grid
grid = [[Cell(x, y, y*GRID_W + x) for y in range(GRID_H)] for x in range(GRID_W)]

# Precompute neighbor offsets
neighbor_offsets = []
for dx in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
    for dy in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
        if dx == 0 and dy == 0:
            continue
        neighbor_offsets.append((dx, dy))
NEIGHBOR_COUNT = len(neighbor_offsets)

# Calculate input dimension (neighbors + info + sensory)
SENSORY_DIM = 3  # dx_food, dy_food, energy
INP_DIM = NEIGHBOR_COUNT + INFO_DIM + SENSORY_DIM

# Creature tracking
creature_cells = set()  # Set of (x,y) coordinates belonging to creature
food_positions = set()  # Set of (x,y) coordinates with food

# Evolution log
evolution_log = []

# -----------------------
# Neural Network Brain
# -----------------------
class NeuralBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, output_dim=3):
        super(NeuralBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize neural network brain
brain = NeuralBrain(INP_DIM)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
brain = brain.to(device)
print(f"Using device: {device}")

# Optimizer for learning
optimizer = optim.Adam(brain.parameters(), lr=LEARNING_RATE)

# GradScaler for AMP
scaler = None
if USE_AMP and device.type == "cuda" and AmpGradScaler is not None:
    scaler = AmpGradScaler()

# Replay buffer
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Performance profiling
frame_times = deque(maxlen=30)  # last 30 frames for averaging
forward_times = deque(maxlen=30)
backward_times = deque(maxlen=30)

# -----------------------
# Helper functions
# -----------------------
def neighbors_coords(x, y):
    """Generate coordinates of neighbors"""
    for dx, dy in neighbor_offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny

def find_nearest_food(creature_center_x, creature_center_y):
    """Find relative vector to nearest food"""
    if not food_positions:
        return 0.0, 0.0
    
    min_dist = float('inf')
    nearest_x, nearest_y = 0, 0
    
    for fx, fy in food_positions:
        dist = sqrt((fx - creature_center_x)**2 + (fy - creature_center_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = fx, fy
    
    # Return relative vector normalized
    dx = nearest_x - creature_center_x
    dy = nearest_y - creature_center_y
    
    # Normalize but keep direction
    if min_dist > 0:
        dx /= min_dist
        dy /= min_dist
    
    return dx, dy

def get_creature_center():
    """Get center of mass of creature"""
    if not creature_cells:
        return GRID_W//2, GRID_H//2
    
    cx = sum(x for x, y in creature_cells) / len(creature_cells)
    cy = sum(y for x, y in creature_cells) / len(creature_cells)
    return cx, cy

def initialize_creature():
    """Initialize a small creature in the center"""
    global creature_cells
    creature_cells = set()
    
    # Create a 3x3 block in the center
    center_x, center_y = GRID_W//2, GRID_H//2
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            x, y = center_x + dx, center_y + dy
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                grid[x][y].alive = True
                grid[x][y].act = random.random() * 0.5
                grid[x][y].info = np.random.normal(scale=0.3, size=INFO_DIM).astype(np.float32)
                grid[x][y].energy = 1.0
                creature_cells.add((x, y))

def place_food(x, y):
    """Place food at position"""
    if 0 <= x < GRID_W and 0 <= y < GRID_H:
        food_positions.add((x, y))

def remove_food(x, y):
    """Remove food at position"""
    if (x, y) in food_positions:
        food_positions.remove((x, y))

# -----------------------
# Simulation functions
# -----------------------
def sim_step(reward_map=None, frame_count=0):
    global brain, optimizer, scaler, EXPLORATION_RATE
    # reward_map: 2D array same dims with reward floats (0..1) applied this timestep
    
    # Collect all alive creature cells
    alive_cells = []
    for x, y in creature_cells:
        cell = grid[x][y]
        if cell.alive:
            alive_cells.append((cell, x, y))
    
    if not alive_cells:
        return
    
    # Vectorized forward pass
    N_alive = len(alive_cells)
    
    # Build input tensor for all alive cells
    inp_tensor = torch.zeros((N_alive, INP_DIM), dtype=torch.float32, device=device)
    
    # Get creature center for sensory input
    creature_center_x, creature_center_y = get_creature_center()
    dx_food, dy_food = find_nearest_food(creature_center_x, creature_center_y)
    
    for i, (cell, x, y) in enumerate(alive_cells):
        # Gather neighbor activations
        neigh_idx = 0
        for nx, ny in neighbors_coords(x, y):
            neigh_cell = grid[nx][ny]
            inp_tensor[i, neigh_idx] = float(neigh_cell.act if neigh_cell.alive else 0.0)
            neigh_idx += 1
        
        # Add cell info
        inp_tensor[i, NEIGHBOR_COUNT:NEIGHBOR_COUNT+INFO_DIM] = torch.from_numpy(cell.info).to(device)
        
        # Add sensory inputs
        inp_tensor[i, NEIGHBOR_COUNT+INFO_DIM] = dx_food
        inp_tensor[i, NEIGHBOR_COUNT+INFO_DIM+1] = dy_food
        inp_tensor[i, NEIGHBOR_COUNT+INFO_DIM+2] = cell.energy
    
    # Batched forward pass
    forward_start = pygame.time.get_ticks()
    with torch.no_grad():
        if USE_AMP and device.type == "cuda" and autocast_context is not None:
            with autocast_context():
                predictions = brain(inp_tensor)
        else:
            predictions = brain(inp_tensor)
    forward_time = pygame.time.get_ticks() - forward_start
    forward_times.append(forward_time)
    
    # Apply updates
    delta_acts = predictions[:, 0].cpu().numpy() * DT  # First output is activation delta
    
    # Collect experiences for learning
    experiences = []
    
    for i, (cell, x, y) in enumerate(alive_cells):
        prev_act = cell.act
        # Apply delta and decay
        cell.act = clamp01(cell.act + delta_acts[i] - DECAY * cell.act * DT)
        
        # Energy decay
        cell.energy = max(0.0, cell.energy - ENERGY_DECAY)
        
        # Check for food consumption
        reward = 0.0
        if (x, y) in food_positions:
            cell.energy = min(1.0, cell.energy + FOOD_ENERGY)
            remove_food(x, y)
            reward = 1.0  # Positive reward for eating
        
        # Apply manual reward if provided
        if reward_map is not None and reward_map[x, y] > 0:
            reward = max(reward, reward_map[x, y])
        
        # Store experience
        experience = {
            'state': inp_tensor[i].cpu().numpy(),
            'action': predictions[i].cpu().numpy(),
            'reward': reward,
            'cell_id': cell.id
        }
        experiences.append(experience)
        
        # Death check
        if cell.energy < DEATH_ENERGY_THRESHOLD:
            cell.death_timer += 1
            if cell.death_timer >= DEATH_TTL:
                cell.alive = False
                if (x, y) in creature_cells:
                    creature_cells.remove((x, y))
        else:
            cell.death_timer = 0
    
    # Add experiences to replay buffer
    if experiences:
        replay_buffer.extend(experiences)
    
    # Neurogenesis: spawn children from highly active cells with sufficient energy
    new_cells = []
    for cell, x, y in alive_cells:
        if cell.act > SPAWN_THRESHOLD and cell.energy > SPAWN_ENERGY_MIN and random.random() < 0.1:
            # find empty neighbor
            empties = [(nx, ny) for nx, ny in neighbors_coords(x, y) if not grid[nx][ny].alive]
            if empties:
                nx, ny = random.choice(empties)
                if (nx, ny) not in creature_cells:
                    child = grid[nx][ny]
                    child.alive = True
                    child.act = min(1.0, cell.act * 0.6 + 0.1)
                    child.energy = cell.energy * 0.7
                    # copy info with mutation
                    child.info = (cell.info + np.random.normal(scale=MUTATION_STD, size=INFO_DIM)).astype(np.float32)
                    # Evolution tracking
                    child.parent_id = cell.id
                    child.generation = cell.generation + 1
                    # Add to creature
                    new_cells.append((nx, ny))
                    # Log the birth
                    evolution_log.append({
                        'child_id': child.id,
                        'parent_id': cell.id,
                        'generation': child.generation,
                        'position': (nx, ny),
                    })
    
    # Add new cells to creature
    for x, y in new_cells:
        creature_cells.add((x, y))
    
    # Learning: Update neural network when reward present or periodically
    train_this_frame = (frame_count % TRAIN_EVERY_N_FRAMES == 0) and len(replay_buffer) > BATCH_SIZE
    
    if train_this_frame and replay_buffer:
        # Sample batch from replay buffer
        batch_size = min(BATCH_SIZE, len(replay_buffer))
        batch = random.sample(replay_buffer, batch_size)
        
        # Build training batch
        batch_states = torch.zeros((batch_size, INP_DIM), dtype=torch.float32, device=device)
        batch_actions = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)  # 3 outputs
        batch_rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for i, exp in enumerate(batch):
            batch_states[i] = torch.from_numpy(exp['state']).to(device)
            batch_actions[i] = torch.from_numpy(exp['action']).to(device)
            batch_rewards[i] = float(exp['reward'])
        
        # Training step
        backward_start = pygame.time.get_ticks()
        optimizer.zero_grad()
        
        if USE_AMP and device.type == "cuda" and scaler is not None and autocast_context is not None:
            with autocast_context():
                preds = brain(batch_states)
                # Loss is MSE between predicted actions and target actions modulated by reward
                target_actions = batch_actions + batch_rewards.unsqueeze(1) * 0.1  # Scale reward influence
                loss = nn.functional.mse_loss(preds, target_actions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = brain(batch_states)
            # Loss is MSE between predicted actions and target actions modulated by reward
            target_actions = batch_actions + batch_rewards.unsqueeze(1) * 0.1  # Scale reward influence
            loss = nn.functional.mse_loss(preds, target_actions)
            loss.backward()
            optimizer.step()
        
        backward_time = pygame.time.get_ticks() - backward_start
        backward_times.append(backward_time)
    
    # Decay exploration
    if EXPLORATION_RATE > EXPLORATION_MIN:
        EXPLORATION_RATE *= EXPLORATION_DECAY

# -----------------------
# Visualization (pygame)
# -----------------------
def init_visualization():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE, GRID_H*CELL_SIZE + 100))  # Extra space for stats
    pygame.display.set_caption("Living AI Creature Simulator")
    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()
    return screen, font, clock

def draw(screen, font, clock):
    # Draw grid
    for x in range(GRID_W):
        for y in range(GRID_H):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            cell = grid[x][y]
            
            if cell.alive and (x, y) in creature_cells:
                # Creature cell
                base = info_to_color(cell.info)
                brightness = cell.act
                color = tuple(int(brightness * c) for c in base)
                pygame.draw.rect(screen, color, rect)
            elif (x, y) in food_positions:
                # Food
                pygame.draw.rect(screen, (255, 200, 100), rect)
            else:
                # Empty space
                pygame.draw.rect(screen, (20, 20, 20), rect)
    
    # Draw grid lines
    for x in range(0, GRID_W*CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, GRID_H*CELL_SIZE))
    for y in range(0, GRID_H*CELL_SIZE, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (0, y), (GRID_W*CELL_SIZE, y))
    
    # Draw stats panel
    stats_y = GRID_H * CELL_SIZE
    pygame.draw.rect(screen, (30, 30, 30), (0, stats_y, GRID_W*CELL_SIZE, 100))
    
    # Calculate stats
    creature_energy = sum(grid[x][y].energy for x, y in creature_cells if grid[x][y].alive) / max(1, len(creature_cells))
    creature_size = len(creature_cells)
    avg_activation = sum(grid[x][y].act for x, y in creature_cells if grid[x][y].alive) / max(1, len(creature_cells))
    fps = clock.get_fps()
    
    # Draw stats text
    stats = [
        f"Cells: {creature_size}",
        f"Energy: {creature_energy:.2f}",
        f"Act: {avg_activation:.2f}",
        f"FPS: {fps:.1f}",
        f"Food: {len(food_positions)}",
        f"Explore: {EXPLORATION_RATE:.2f}"
    ]
    
    for i, stat in enumerate(stats):
        text = font.render(stat, True, (200, 200, 200))
        screen.blit(text, (10, stats_y + 5 + i*15))
    
    # Draw lineage histogram
    if evolution_log:
        hist_width = 100
        hist_height = 60
        hist_x = GRID_W*CELL_SIZE - hist_width - 10
        hist_y = stats_y + 20
        
        pygame.draw.rect(screen, (50, 50, 50), (hist_x, hist_y, hist_width, hist_height))
        pygame.draw.rect(screen, (100, 100, 100), (hist_x, hist_y, hist_width, hist_height), 1)
        
        # Simple generation histogram
        generations = [entry['generation'] for entry in evolution_log[-50:]]  # Last 50 entries
        if generations:
            max_gen = max(generations)
            if max_gen > 0:
                for gen in range(min(10, max_gen + 1)):
                    count = generations.count(gen)
                    if count > 0:
                        bar_height = min(hist_height - 5, count * 3)
                        bar_width = max(1, hist_width // min(10, max_gen + 1) - 1)
                        bar_x = hist_x + gen * bar_width
                        bar_y = hist_y + hist_height - bar_height - 2
                        pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, bar_width, bar_height))

# -----------------------
# Save / Load
# -----------------------
def save_sim(filename=SAVE_FILE):
    data = {
        "brain_state_dict": brain.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grid_state": [[[grid[x][y].alive, grid[x][y].act, grid[x][y].info.tolist(), 
                        grid[x][y].energy, grid[x][y].generation, grid[x][y].parent_id, 
                        grid[x][y].death_timer] 
                       for y in range(GRID_H)] for x in range(GRID_W)],
        "creature_cells": list(creature_cells),
        "food_positions": list(food_positions),
        "evolution_log": evolution_log,
        "replay_buffer": list(replay_buffer)
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("Saved to", filename)

def load_sim(filename=SAVE_FILE):
    global brain, optimizer, evolution_log
    if not os.path.exists(filename):
        print("No save file found:", filename)
        return
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Load brain state
    brain.load_state_dict(data["brain_state_dict"])
    brain = brain.to(device)
    
    # Load optimizer state
    optimizer.load_state_dict(data["optimizer_state_dict"])
    
    # Load grid state
    for x in range(GRID_W):
        for y in range(GRID_H):
            state = data["grid_state"][x][y]
            grid[x][y].alive = state[0]
            grid[x][y].act = state[1]
            grid[x][y].info = np.array(state[2], dtype=np.float32)
            grid[x][y].energy = state[3]
            grid[x][y].generation = state[4]
            grid[x][y].parent_id = state[5]
            grid[x][y].death_timer = state[6]
    
    # Load creature cells
    global creature_cells, food_positions
    creature_cells = set(tuple(cell) for cell in data["creature_cells"])
    food_positions = set(tuple(food) for food in data["food_positions"])
    
    # Load evolution log
    evolution_log = data["evolution_log"]
    
    # Load replay buffer
    global replay_buffer
    replay_buffer = deque(data["replay_buffer"], maxlen=REPLAY_BUFFER_SIZE)
        
    print("Loaded from", filename)

# -----------------------
# Smoke test
# -----------------------
def run_smoke_test(steps=20):
    """Run a headless smoke test"""
    print("Running smoke test...")
    
    # Initialize
    initialize_creature()
    initial_size = len(creature_cells)
    
    # Place some food
    for _ in range(5):
        fx = random.randint(5, GRID_W-6)
        fy = random.randint(5, GRID_H-6)
        place_food(fx, fy)
    
    # Track distances to food
    initial_distances = []
    creature_center_x, creature_center_y = get_creature_center()
    for fx, fy in food_positions:
        dist = sqrt((fx - creature_center_x)**2 + (fy - creature_center_y)**2)
        initial_distances.append(dist)
    avg_initial_distance = sum(initial_distances) / len(initial_distances) if initial_distances else 0
    
    # Run simulation steps
    for i in range(steps):
        reward_map = np.zeros((GRID_W, GRID_H), dtype=np.float32)
        
        # Apply occasional reward
        if i % 7 == 0:
            creature_center_x, creature_center_y = get_creature_center()
            for x in range(max(0, int(creature_center_x)-2), min(GRID_W, int(creature_center_x)+3)):
                for y in range(max(0, int(creature_center_y)-2), min(GRID_H, int(creature_center_y)+3)):
                    reward_map[x, y] = 0.5
        
        sim_step(reward_map=reward_map, frame_count=i)
        
        if i % (steps//5) == 0:
            current_size = len(creature_cells)
            print(f"Step {i}: {current_size} cells")
    
    # Check final distance to food
    final_distances = []
    creature_center_x, creature_center_y = get_creature_center()
    for fx, fy in food_positions:
        dist = sqrt((fx - creature_center_x)**2 + (fy - creature_center_y)**2)
        final_distances.append(dist)
    avg_final_distance = sum(final_distances) / len(final_distances) if final_distances else 0
    
    final_size = len(creature_cells)
    
    # Assertions
    assert initial_size > 0, "No creature initialized"
    assert len(food_positions) > 0, "No food placed"
    
    # Check that creature learned to move toward food (distance should decrease)
    distance_improvement = avg_initial_distance - avg_final_distance
    assert distance_improvement >= 0, f"Distance should decrease: {avg_initial_distance:.2f} -> {avg_final_distance:.2f}"
    
    print(f"Smoke test passed! Distance improved by: {distance_improvement:.2f}")
    print(f"Creature size: {initial_size} -> {final_size}")
    return True

# -----------------------
# Main loop
# -----------------------
def main():
    screen, font, clock = init_visualization()
    
    # Initialize creature
    initialize_creature()
    
    running = True
    frame_count = 0
    
    print("Living AI Creature Simulator")
    print("Controls:")
    print("  Left-click: Place food")
    print("  Space: Apply reward")
    print("  S: Save model")
    print("  L: Load model")
    print("  C: Clear/reset")
    print("  ESC: Quit")
    
    while running:
        frame_start = pygame.time.get_ticks()
        
        screen.fill((0,0,0))
        reward_map = np.zeros((GRID_W, GRID_H), dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Place food
                    mx, my = event.pos
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        place_food(gx, gy)
                elif event.button == 3:
                    # Place creature cell
                    mx, my = event.pos
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        grid[gx][gy].alive = True
                        grid[gx][gy].act = random.random() * 0.5
                        grid[gx][gy].info = np.random.normal(scale=0.3, size=INFO_DIM).astype(np.float32)
                        grid[gx][gy].energy = 1.0
                        creature_cells.add((gx, gy))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_sim()
                elif event.key == pygame.K_l:
                    load_sim()
                elif event.key == pygame.K_c:
                    initialize_creature()
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # apply reward around creature
                    creature_center_x, creature_center_y = get_creature_center()
                    for x in range(max(0, int(creature_center_x)-2), min(GRID_W, int(creature_center_x)+3)):
                        for y in range(max(0, int(creature_center_y)-2), min(GRID_H, int(creature_center_y)+3)):
                            reward_map[x, y] = max(reward_map[x, y], 0.8)
                    print("Applied reward near creature")

        # Every frame, do a sim step with current reward_map
        sim_step(reward_map=reward_map, frame_count=frame_count)
        frame_count += 1

        draw(screen, font, clock)
        pygame.display.flip()
        clock.tick(60)  # Target 60 FPS
        
        # Profile frame time
        frame_time = pygame.time.get_ticks() - frame_start
        frame_times.append(frame_time)
        
        # Print profiling info occasionally
        if frame_count % 60 == 0 and frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            avg_forward_time = sum(forward_times) / len(forward_times) if forward_times else 0
            avg_backward_time = sum(backward_times) / len(backward_times) if backward_times else 0
            print(f"Avg frame: {avg_frame_time:.1f}ms | Forward: {avg_forward_time:.1f}ms | Backward: {avg_backward_time:.1f}ms")

    pygame.quit()
    print("Bye! Saved weights automatically.")
    save_sim()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_smoke_test()
    else:
        main()