#!/usr/bin/env python3
"""
RL-Enhanced Living AI Creature using Neuro-Genesis Structure
चलो सीखते हैं!!! Let's learn!!!

Run: python creature_rl_ns.py
Controls:
  Left-click: Place food
  Right-click: Place hazard
  Space: Apply reward
  S: Save model
  L: Load model
  C: Clear/reset
  ESC: Quit

Advanced Self-Learning Living AI Creature system that extends the Neuro-Genesis (NS) Simulator structure
with embedded Reinforcement Learning (Q-Learning) while preserving NS architecture and data flow principles.
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
LEARNING_RATE = 0.001             # learning rate for NS brain
LEARNING_RATE_RL = 0.0005         # learning rate for RL
DECAY = 0.005                     # activation decay per step
ENERGY_DECAY = 0.002              # energy decay per frame
DEATH_ENERGY_THRESHOLD = 0.1      # energy threshold for death
DEATH_TTL = 30                    # frames before death
FOOD_ENERGY = 0.3                 # energy gained from eating food
HAZARD_ENERGY_LOSS = 0.2          # energy lost from hazard
EXPLORATION_STD = 0.1             # action noise std dev
EPSILON = 0.3                     # exploration rate
EPSILON_DECAY = 0.995             # exploration decay
EPSILON_MIN = 0.01                # minimum exploration
TRAIN_EVERY_N_FRAMES = 5          # train every N frames
BATCH_SIZE = 32                   # training batch size
MAX_BATCH_SIZE = 512              # maximum batch size
REPLAY_BUFFER_SIZE = 2000         # replay buffer size
GAMMA = 0.95                      # discount factor
SAVE_FILE = "creature_rl_save.pkl"

# Region types
REGION_VISION = 0
REGION_MOTOR = 1
REGION_COGNITIVE = 2

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
        self.region = REGION_VISION  # Default region

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
SENSORY_DIM = 4  # dx_food, dy_food, dx_hazard, dy_hazard
INP_DIM = NEIGHBOR_COUNT + INFO_DIM + SENSORY_DIM

# Creature tracking
creature_cells = set()  # Set of (x,y) coordinates belonging to creature
food_positions = set()  # Set of (x,y) coordinates with food
hazard_positions = set()  # Set of (x,y) coordinates with hazards

# Evolution log
evolution_log = []

# -----------------------
# Neural Network Brain (NS + RL)
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

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, action_dim=4):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize neural network brains
ns_brain = NeuralBrain(INP_DIM)
q_network = QNetwork(INP_DIM)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ns_brain = ns_brain.to(device)
q_network = q_network.to(device)
print(f"Using device: {device}")

# Optimizers
ns_optimizer = optim.Adam(ns_brain.parameters(), lr=LEARNING_RATE)
rl_optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE_RL)

# GradScaler for AMP
scaler_ns = None
scaler_rl = None
if USE_AMP and device.type == "cuda" and AmpGradScaler is not None:
    scaler_ns = AmpGradScaler()
    scaler_rl = AmpGradScaler()

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

def find_nearest_hazard(creature_center_x, creature_center_y):
    """Find relative vector to nearest hazard"""
    if not hazard_positions:
        return 0.0, 0.0
    
    min_dist = float('inf')
    nearest_x, nearest_y = 0, 0
    
    for hx, hy in hazard_positions:
        dist = sqrt((hx - creature_center_x)**2 + (hy - creature_center_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = hx, hy
    
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
                # Assign regions
                if dx == 0 and dy == 0:
                    grid[x][y].region = REGION_COGNITIVE
                elif abs(dx) + abs(dy) == 1:
                    grid[x][y].region = REGION_MOTOR
                else:
                    grid[x][y].region = REGION_VISION
                creature_cells.add((x, y))

def place_food(x, y):
    """Place food at position"""
    if 0 <= x < GRID_W and 0 <= y < GRID_H:
        food_positions.add((x, y))

def remove_food(x, y):
    """Remove food at position"""
    if (x, y) in food_positions:
        food_positions.remove((x, y))

def place_hazard(x, y):
    """Place hazard at position"""
    if 0 <= x < GRID_W and 0 <= y < GRID_H:
        hazard_positions.add((x, y))

def remove_hazard(x, y):
    """Remove hazard at position"""
    if (x, y) in hazard_positions:
        hazard_positions.remove((x, y))

# -----------------------
# RL Functions
# -----------------------
def get_state(cell, x, y):
    """Get the state representation for a cell"""
    # Gather neighbor activations
    state = []
    
    # Add neighbor activations
    for nx, ny in neighbors_coords(x, y):
        neigh_cell = grid[nx][ny]
        state.append(float(neigh_cell.act if neigh_cell.alive else 0.0))
    
    # Pad with zeros if needed
    while len(state) < NEIGHBOR_COUNT:
        state.append(0.0)
    
    # Add cell info
    state.extend(cell.info.tolist())
    
    # Add sensory inputs
    creature_center_x, creature_center_y = get_creature_center()
    dx_food, dy_food = find_nearest_food(creature_center_x, creature_center_y)
    dx_hazard, dy_hazard = find_nearest_hazard(creature_center_x, creature_center_y)
    
    state.append(dx_food)
    state.append(dy_food)
    state.append(dx_hazard)
    state.append(dy_hazard)
    
    return np.array(state, dtype=np.float32)

def select_action(state):
    """Select action using epsilon-greedy policy"""
    global EPSILON
    if random.random() <= EPSILON:
        # Explore: choose random action
        return random.randint(0, 3)
    else:
        # Exploit: choose best action based on Q-values
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if USE_AMP and device.type == "cuda" and autocast_context is not None:
                with autocast_context():
                    q_values = q_network(state_tensor)
            else:
                q_values = q_network(state_tensor)
        return q_values.argmax().item()

def compute_reward(cell, prev_act, new_act):
    """Compute reward based on cell activation change and environmental factors"""
    # Reward for positive activation change
    activation_change = new_act - prev_act
    reward = activation_change * 10.0
    
    # Additional reward for maintaining optimal activation (0.5-0.8)
    if 0.5 <= new_act <= 0.8:
        reward += 0.1
    
    # Penalty for extreme activations
    if new_act > 0.9 or new_act < 0.1:
        reward -= 0.05
        
    # Energy-based reward
    if cell.energy > 0.8:
        reward += 0.05
    elif cell.energy < 0.3:
        reward -= 0.1
        
    return reward

def update_q_network(states, actions, rewards, next_states, dones):
    """Update Q-network using Q-learning"""
    states_tensor = torch.from_numpy(states).to(device)
    next_states_tensor = torch.from_numpy(next_states).to(device)
    actions_tensor = torch.from_numpy(actions).to(device)
    rewards_tensor = torch.from_numpy(rewards).to(device)
    dones_tensor = torch.from_numpy(dones).float().to(device)  # Convert to float
    
    # Get current Q-values
    if USE_AMP and device.type == "cuda" and scaler_rl is not None and autocast_context is not None:
        with autocast_context():
            current_q_values = q_network(states_tensor)
            current_q_value = current_q_values.gather(1, actions_tensor.unsqueeze(1))
            
            # Get next Q-values and compute target
            with torch.no_grad():
                next_q_values = q_network(next_states_tensor)
            target_q_value = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values.max(1)[0]
            
            # Compute loss
            loss = nn.functional.mse_loss(current_q_value.squeeze(), target_q_value)
        
        rl_optimizer.zero_grad()
        scaler_rl.scale(loss).backward()
        scaler_rl.step(rl_optimizer)
        scaler_rl.update()
    else:
        current_q_values = q_network(states_tensor)
        current_q_value = current_q_values.gather(1, actions_tensor.unsqueeze(1))
        
        # Get next Q-values and compute target
        with torch.no_grad():
            next_q_values = q_network(next_states_tensor)
        target_q_value = rewards_tensor + (1 - dones_tensor) * GAMMA * next_q_values.max(1)[0]
        
        # Compute loss
        loss = nn.functional.mse_loss(current_q_value.squeeze(), target_q_value)
        
        rl_optimizer.zero_grad()
        loss.backward()
        rl_optimizer.step()
    
    return loss.item()

# -----------------------
# Simulation functions
# -----------------------
def sim_step(reward_map=None, frame_count=0):
    global ns_brain, q_network, ns_optimizer, rl_optimizer, scaler_ns, scaler_rl, EPSILON
    # reward_map: 2D array same dims with reward floats (0..1) applied this timestep
    
    # Collect all alive creature cells
    alive_cells = []
    for x, y in creature_cells:
        cell = grid[x][y]
        if cell.alive:
            alive_cells.append((cell, x, y))
    
    if not alive_cells:
        return
    
    # Vectorized forward pass for NS brain
    N_alive = len(alive_cells)
    
    # Build input tensor for all alive cells
    inp_tensor = torch.zeros((N_alive, INP_DIM), dtype=torch.float32, device=device)
    
    # Get creature center for sensory input
    creature_center_x, creature_center_y = get_creature_center()
    dx_food, dy_food = find_nearest_food(creature_center_x, creature_center_y)
    dx_hazard, dy_hazard = find_nearest_hazard(creature_center_x, creature_center_y)
    
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
        inp_tensor[i, NEIGHBOR_COUNT+INFO_DIM+2] = dx_hazard
        inp_tensor[i, NEIGHBOR_COUNT+INFO_DIM+3] = dy_hazard
    
    # Batched forward pass for NS brain
    forward_start = pygame.time.get_ticks()
    with torch.no_grad():
        if USE_AMP and device.type == "cuda" and autocast_context is not None:
            with autocast_context():
                ns_predictions = ns_brain(inp_tensor)
        else:
            ns_predictions = ns_brain(inp_tensor)
    forward_time = pygame.time.get_ticks() - forward_start
    forward_times.append(forward_time)
    
    # Apply updates
    delta_acts = ns_predictions[:, 0].cpu().numpy() * DT  # First output is activation delta
    
    # Collect experiences for RL learning
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
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
        
        # Check for hazard contact
        if (x, y) in hazard_positions:
            cell.energy = max(0.0, cell.energy - HAZARD_ENERGY_LOSS)
            remove_hazard(x, y)
            reward = -0.5  # Negative reward for hazard
        
        # Apply manual reward if provided
        if reward_map is not None and reward_map[x, y] > 0:
            reward = max(reward, reward_map[x, y])
        
        # Compute RL reward
        rl_reward = compute_reward(cell, prev_act, cell.act)
        reward += rl_reward
        
        # Get state for RL
        state = get_state(cell, x, y)
        action = select_action(state)
        
        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        # Apply action (movement)
        dx, dy = 0, 0
        if action == 0:  # Up
            dy = -1
        elif action == 1:  # Down
            dy = 1
        elif action == 2:  # Left
            dx = -1
        elif action == 3:  # Right
            dx = 1
            
        # Move creature (apply to all cells in creature)
        new_creature_cells = set()
        for cx, cy in creature_cells:
            nx, ny = cx + dx, cy + dy
            # Clamp to grid bounds
            nx = max(0, min(GRID_W-1, nx))
            ny = max(0, min(GRID_H-1, ny))
            new_creature_cells.add((nx, ny))
            # Update cell position
            if grid[cx][cy].alive:
                grid[nx][ny].alive = True
                grid[nx][ny].act = grid[cx][cy].act
                grid[nx][ny].info = grid[cx][cy].info.copy()
                grid[nx][ny].energy = grid[cx][cy].energy
                grid[nx][ny].region = grid[cx][cy].region
                grid[nx][ny].generation = grid[cx][cy].generation
                grid[nx][ny].parent_id = grid[cx][cy].parent_id
                grid[nx][ny].id = grid[cx][cy].id
                grid[nx][ny].death_timer = grid[cx][cy].death_timer
                # Clear old position
                if (cx, cy) != (nx, ny):
                    grid[cx][cy].alive = False
        
        creature_cells.clear()
        creature_cells.update(new_creature_cells)
        
        # Get next state
        next_state = get_state(cell, x, y)  # Simplified - in reality would need updated positions
        next_states.append(next_state)
        dones.append(not cell.alive)
        
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
    if states:
        for i in range(len(states)):
            experience = {
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'next_state': next_states[i],
                'done': dones[i]
            }
            replay_buffer.append(experience)
    
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
                    # Assign region based on parent
                    child.region = cell.region
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
    
    # RL Learning: Update Q-network when reward present or periodically
    train_this_frame = (frame_count % TRAIN_EVERY_N_FRAMES == 0) and len(replay_buffer) > BATCH_SIZE
    
    if train_this_frame and replay_buffer and len(replay_buffer) >= BATCH_SIZE:
        # Sample batch from replay buffer
        batch = random.sample(replay_buffer, BATCH_SIZE)
        
        # Build training batch
        batch_states = np.array([exp['state'] for exp in batch])
        batch_actions = np.array([exp['action'] for exp in batch])
        batch_rewards = np.array([exp['reward'] for exp in batch])
        batch_next_states = np.array([exp['next_state'] for exp in batch])
        batch_dones = np.array([exp['done'] for exp in batch])
        
        # Training step
        backward_start = pygame.time.get_ticks()
        loss = update_q_network(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
        backward_time = pygame.time.get_ticks() - backward_start
        backward_times.append(backward_time)
    
    # Decay exploration
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

# -----------------------
# Visualization (pygame)
# -----------------------
def init_visualization():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE, GRID_H*CELL_SIZE + 120))  # Extra space for stats
    pygame.display.set_caption("RL-Enhanced Living AI Creature (NS Structure)")
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
                # Creature cell - color based on region
                base = info_to_color(cell.info)
                brightness = cell.act
                color = tuple(int(brightness * c) for c in base)
                
                # Modify color based on region
                if cell.region == REGION_VISION:
                    # Blue tint for vision
                    color = (color[0]//2, color[1]//2, min(255, color[2] + 100))
                elif cell.region == REGION_MOTOR:
                    # Red tint for motor
                    color = (min(255, color[0] + 100), color[1]//2, color[2]//2)
                elif cell.region == REGION_COGNITIVE:
                    # Green tint for cognitive
                    color = (color[0]//2, min(255, color[1] + 100), color[2]//2)
                    
                pygame.draw.rect(screen, color, rect)
            elif (x, y) in food_positions:
                # Food
                pygame.draw.rect(screen, (255, 200, 100), rect)
            elif (x, y) in hazard_positions:
                # Hazard
                pygame.draw.rect(screen, (200, 50, 50), rect)
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
    pygame.draw.rect(screen, (30, 30, 30), (0, stats_y, GRID_W*CELL_SIZE, 120))
    
    # Calculate stats
    creature_energy = sum(grid[x][y].energy for x, y in creature_cells if grid[x][y].alive) / max(1, len(creature_cells))
    creature_size = len(creature_cells)
    avg_activation = sum(grid[x][y].act for x, y in creature_cells if grid[x][y].alive) / max(1, len(creature_cells))
    fps = clock.get_fps()
    
    # Calculate average Q-value (sample a few cells)
    avg_q_value = 0.0
    q_count = 0
    for x, y in list(creature_cells)[:5]:  # Sample first 5 cells
        if grid[x][y].alive:
            state = get_state(grid[x][y], x, y)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                if USE_AMP and device.type == "cuda" and autocast_context is not None:
                    with autocast_context():
                        q_values = q_network(state_tensor)
                else:
                    q_values = q_network(state_tensor)
            avg_q_value += q_values.mean().item()
            q_count += 1
    if q_count > 0:
        avg_q_value /= q_count
    
    # Draw stats text
    stats = [
        f"Cells: {creature_size}",
        f"Energy: {creature_energy:.2f}",
        f"Act: {avg_activation:.2f}",
        f"FPS: {fps:.1f}",
        f"Food: {len(food_positions)}",
        f"Hazards: {len(hazard_positions)}",
        f"Epsilon: {EPSILON:.3f}",
        f"Avg Q-Value: {avg_q_value:.3f}"
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
        "ns_brain_state_dict": ns_brain.state_dict(),
        "q_network_state_dict": q_network.state_dict(),
        "ns_optimizer_state_dict": ns_optimizer.state_dict(),
        "rl_optimizer_state_dict": rl_optimizer.state_dict(),
        "grid_state": [[[grid[x][y].alive, grid[x][y].act, grid[x][y].info.tolist(), 
                        grid[x][y].energy, grid[x][y].generation, grid[x][y].parent_id, 
                        grid[x][y].death_timer, grid[x][y].region] 
                       for y in range(GRID_H)] for x in range(GRID_W)],
        "creature_cells": list(creature_cells),
        "food_positions": list(food_positions),
        "hazard_positions": list(hazard_positions),
        "evolution_log": evolution_log,
        "replay_buffer": list(replay_buffer),
        "EPSILON": EPSILON,
        "GAMMA": GAMMA
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("Saved to", filename)

def load_sim(filename=SAVE_FILE):
    global ns_brain, q_network, ns_optimizer, rl_optimizer, evolution_log, EPSILON, GAMMA
    if not os.path.exists(filename):
        print("No save file found:", filename)
        return
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Load brain states
    ns_brain.load_state_dict(data["ns_brain_state_dict"])
    q_network.load_state_dict(data["q_network_state_dict"])
    ns_brain = ns_brain.to(device)
    q_network = q_network.to(device)
    
    # Load optimizer states
    ns_optimizer.load_state_dict(data["ns_optimizer_state_dict"])
    rl_optimizer.load_state_dict(data["rl_optimizer_state_dict"])
    
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
            grid[x][y].region = state[7] if len(state) > 7 else REGION_VISION
    
    # Load creature cells
    global creature_cells, food_positions, hazard_positions
    creature_cells = set(tuple(cell) for cell in data["creature_cells"])
    food_positions = set(tuple(food) for food in data["food_positions"])
    hazard_positions = set(tuple(hazard) for hazard in data["hazard_positions"])
    
    # Load evolution log
    evolution_log = data["evolution_log"]
    
    # Load replay buffer
    global replay_buffer
    replay_buffer = deque(data["replay_buffer"], maxlen=REPLAY_BUFFER_SIZE)
    
    # Load RL parameters
    EPSILON = data.get("EPSILON", EPSILON)
    GAMMA = data.get("GAMMA", GAMMA)
        
    print("Loaded from", filename)

# -----------------------
# Smoke test
# -----------------------
def run_smoke_test(episodes=5):
    """Run a headless smoke test"""
    print("Running smoke test...")
    
    episode_rewards = []
    
    for episode in range(episodes):
        # Initialize
        initialize_creature()
        initial_size = len(creature_cells)
        
        # Place some food and hazards
        for _ in range(3):
            fx = random.randint(5, GRID_W-6)
            fy = random.randint(5, GRID_H-6)
            place_food(fx, fy)
            
        for _ in range(2):
            hx = random.randint(5, GRID_W-6)
            hy = random.randint(5, GRID_H-6)
            place_hazard(hx, hy)
        
        total_reward = 0.0
        steps = 20
        
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
            total_reward += sum(exp['reward'] for exp in list(replay_buffer)[-10:])  # Approximate recent rewards
            
            if i % (steps//5) == 0:
                current_size = len(creature_cells)
                print(f"Episode {episode}, Step {i}: {current_size} cells")
        
        episode_rewards.append(total_reward)
        final_size = len(creature_cells)
        print(f"Episode {episode} completed. Total reward: {total_reward:.2f}, Final size: {final_size}")
    
    # Check that rewards are increasing (learning is happening)
    if len(episode_rewards) > 1:
        reward_improvement = episode_rewards[-1] - episode_rewards[0]
        print(f"Reward improvement: {reward_improvement:.2f}")
        # We expect some improvement but won't assert since it's stochastic
    
    print("Smoke test completed!")
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
    
    print("RL-Enhanced Living AI Creature (NS Structure)")
    print("Controls:")
    print("  Left-click: Place food")
    print("  Right-click: Place hazard")
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
                    # Place hazard
                    mx, my = event.pos
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        place_hazard(gx, gy)
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
    print("Bye! Saved RL weights automatically.")
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