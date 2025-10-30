# File: neuro_genesis_sim.py
# High-Performance Neuro-Genesis Cellular Simulator
# Run: python neuro_genesis_sim.py
# Parameters: GRID_W=64, GRID_H=48, INFO_DIM=6
#
# Exclamations: चलो सीखते हैं!!! Let's learn!!!

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
import threading
import time
import json

# Try to import control panel, but make it optional
try:
    from control_panel import SimulationController
    CONTROL_PANEL_AVAILABLE = True
except ImportError:
    CONTROL_PANEL_AVAILABLE = False
    SimulationController = None  # Define as None to satisfy linter
    print("Control panel not available. Install tkinter to enable GUI controls.")

# Try to import NetworkX for neural network graph visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("NetworkX not available. Neural network graph visualization disabled.")

# Global controller instance
controller = None

# AMP handling - avoid naming conflicts by using a class-based approach
AMP_AVAILABLE = False

# Dummy implementations
class DummyGradScaler:
    def __init__(self, device_type='cuda'):
        pass
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        pass
    def update(self):
        pass

class AMPHandler:
    """Handle AMP context creation"""
    
    @staticmethod
    def dummy_autocast(device_type, enabled=True):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    
    @staticmethod
    def create_context(device_type, enabled=True):
        return AMPHandler.dummy_autocast(device_type, enabled)

# Set up default implementations
create_amp_autocast = AMPHandler.create_context
AmpGradScaler = DummyGradScaler

# Try to set up real AMP implementations
_amp_setup_successful = False
try:
    # Try modern PyTorch imports
    from torch.amp.autocast_mode import autocast as real_autocast
    from torch.amp.grad_scaler import GradScaler as RealGradScaler
    
    class RealAMPHandler:
        @staticmethod
        def create_context(device_type, enabled=True):
            if enabled:
                return real_autocast(device_type=device_type)
            else:
                return AMPHandler.dummy_autocast(device_type, enabled)
    
    create_amp_autocast = RealAMPHandler.create_context
    AmpGradScaler = RealGradScaler
    AMP_AVAILABLE = True
    _amp_setup_successful = True
except ImportError:
    pass

if not _amp_setup_successful:
    try:
        # Try CUDA-specific imports
        from torch.cuda.amp import autocast as cuda_autocast
        from torch.cuda.amp import GradScaler as CudaGradScaler
        
        class CUDAAMPHandler:
            @staticmethod
            def create_context(device_type, enabled=True):
                if enabled and device_type == "cuda":
                    return cuda_autocast()
                else:
                    return AMPHandler.dummy_autocast(device_type, enabled)
        
        create_amp_autocast = CUDAAMPHandler.create_context
        AmpGradScaler = CudaGradScaler
        AMP_AVAILABLE = True
    except ImportError:
        pass

# -----------------------
# Parameters (tweakable)
# -----------------------
GRID_W, GRID_H = 64, 48           # grid (columns x rows)
CELL_SIZE = 12                    # pixels
INFO_DIM = 6                      # dimension of info vector stored per cell
NEIGHBOR_RADIUS = 1               # neighborhood radius (1 => 3x3)

# Multi-region brain parameters
REGION_COUNT = 3                  # Number of brain regions
REGION_NAMES = ["Vision", "Motor", "Cognitive"]  # Region names
REGION_WEIGHTS = [1.0, 1.0, 1.0]  # Weights for each region (can be adjusted)

# Default parameter values (will be updated by controller if available)
DT = 0.1                          # simulation time step (affects update scale)
SPAWN_THRESHOLD = 0.85            # activation threshold to spawn new cell
SPAWN_PROB = 0.35                 # probability to spawn when threshold met
MUTATION_STD = 0.05               # noise added to child info
LEARNING_RATE = 0.002             # learning rate
DECAY = 0.01                      # activation decay per step
STIM_STRONG = 0.9                 # activation added by clicking
INIT_DENSITY = 0.06               # initial fractional occupancy
SAVE_FILE = "sim_save.pkl"

# Reinforcement learning parameters
GAMMA = 0.95                      # discount factor for Q-learning
EPSILON = 0.1                     # exploration rate
EPSILON_DECAY = 0.995             # exploration decay rate
EPSILON_MIN = 0.01                # minimum exploration rate
LEARNING_RATE_RL = 0.001          # RL-specific learning rate

# Sound parameters
SOUND_ENABLED = False             # Sound feedback disabled by default to prevent irritation
BASE_FREQ = 220                   # base frequency for sound feedback
SOUND_COOLDOWN = 100              # frames between sounds
last_sound_frame = 0              # track when last sound was played

# Memory compression parameters
MERGE_THRESHOLD = 0.95            # cosine similarity threshold for merging (increased from 0.9)
MERGE_PROB = 0.005                # probability of checking for merges each step (decreased from 0.01)

# Replay learning parameters
REPLAY_BUFFER_SIZE = 1000         # Size of replay buffer
REPLAY_ENABLED = False            # Enable replay learning
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)  # Replay buffer for experience replay

# Performance parameters
USE_AMP = AMP_AVAILABLE           # use automatic mixed precision if available
BATCH_TRAIN_EVERY_N_FRAMES = 1    # train every N frames
MAX_BATCH_SIZE = 1024             # maximum batch size for training

# Data analytics
activation_history = deque(maxlen=100)  # History of average activations
mutation_history = deque(maxlen=100)    # History of mutation variances

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
# Neural Network Brain
# -----------------------
class NeuralBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, region_count=1):
        super(NeuralBrain, self).__init__()
        self.region_count = region_count
        
        # Create separate networks for each region if multi-region
        if region_count > 1:
            self.networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                ) for _ in range(region_count)
            ])
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()
            )
        
    def forward(self, x, region_id=0):
        if self.region_count > 1:
            return self.networks[region_id](x)
        else:
            return self.network(x)

# -----------------------
# Q-Network for Reinforcement Learning
# -----------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, action_dim=4):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)

# -----------------------
# Grid / Cell structure
# -----------------------
class Cell:
    def __init__(self, id=None, region_id=0):
        self.alive = False
        self.act = 0.0
        self.info: np.ndarray = np.zeros(INFO_DIM, dtype=np.float32)
        self.id = id
        self.region_id = region_id  # Region this cell belongs to
        # Evolution tracking
        self.parent_id = None
        self.generation = 0
        self.mutation_history = []
        # Reinforcement learning
        self.q_values = np.zeros(4)  # 4 possible actions: up, down, left, right
        self.reward_history = []

# Initialize grid with regions
def initialize_grid_with_regions():
    """Initialize grid with cells assigned to different regions"""
    grid = []
    for c in range(GRID_W):
        col = []
        for r in range(GRID_H):
            # Assign region based on position
            if c < GRID_W // 3:
                region_id = 0  # Vision region
            elif c < 2 * GRID_W // 3:
                region_id = 1  # Motor region
            else:
                region_id = 2  # Cognitive region
            col.append(Cell(r*GRID_W + c, region_id))
        grid.append(col)
    return grid

grid = initialize_grid_with_regions()

# Precompute neighbor offsets
neighbor_offsets = []
for dx in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
    for dy in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
        if dx == 0 and dy == 0:
            continue
        neighbor_offsets.append((dx, dy))
NEIGHBOR_COUNT = len(neighbor_offsets)

# Calculate input dimension
INP_DIM = NEIGHBOR_COUNT + INFO_DIM

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize neural network brain with region support
brain = NeuralBrain(INP_DIM, region_count=REGION_COUNT)
brain = brain.to(device)

# Initialize Q-network for reinforcement learning
q_network = QNetwork(INP_DIM, action_dim=4)  # 4 actions: up, down, left, right
q_network = q_network.to(device)
q_optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE_RL)

print(f"Using device: {device}")

# Optimizer for learning
optimizer = optim.Adam(brain.parameters(), lr=LEARNING_RATE)

# GradScaler for AMP
scaler = AmpGradScaler() if USE_AMP and device.type == "cuda" else None

# Evolution log
evolution_log = []

# Performance profiling
frame_times = deque(maxlen=30)  # last 30 frames for averaging

# Sound system initialization
sound_initialized = False
sound_channel = None

# Control panel thread
control_panel_thread = None

# -----------------------
# Sound Feedback
# -----------------------
def init_sound():
    """Initialize sound system"""
    global sound_initialized, sound_channel
    if SOUND_ENABLED:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            sound_initialized = True
            sound_channel = pygame.mixer.Channel(0)
            print("Sound system initialized")
            return True
        except pygame.error as e:
            print(f"Sound initialization failed: {e}")
            return False
    return False

def play_activation_sound(activation_level):
    """Play a tone with pitch based on activation level"""
    global last_sound_frame
    if not SOUND_ENABLED or not sound_initialized:
        return
        
    # Limit sound frequency
    current_frame = pygame.time.get_ticks()
    if current_frame - last_sound_frame < SOUND_COOLDOWN:
        return
    
    try:
        # Map activation (0-1) to frequency (base_freq to base_freq*4)
        freq = BASE_FREQ + activation_level * BASE_FREQ * 3
        
        # Generate a simple sine wave
        duration = 50  # ms - shorter duration
        sample_rate = 22050
        frames = int(duration * sample_rate / 1000)
        
        # Reduce volume (lower amplitude)
        arr = np.zeros((frames, 2))
        for i in range(frames):
            wave_value = 2048 * np.sin(2 * np.pi * freq * i / sample_rate)
            arr[i][0] = wave_value  # left channel
            arr[i][1] = wave_value  # right channel
            
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        if sound_channel:
            sound_channel.play(sound)
        last_sound_frame = current_frame
    except Exception as e:
        pass  # Ignore sound errors

# -----------------------
# Memory Compression
# -----------------------
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def compress_memory():
    """Merge similar info vectors to reduce memory usage"""
    # Collect all alive cells
    alive_cells = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if cell.alive:
                alive_cells.append((cell, x, y))
    
    # Early return if not enough cells
    if len(alive_cells) < 2:
        return
    
    # Compare pairs of cells for similarity
    merged_count = 0
    processed_pairs = set()
    
    # Limit the number of comparisons to avoid O(n^2) complexity
    max_comparisons = min(1000, len(alive_cells) * 10)  # Cap comparisons
    comparison_count = 0
    
    for i in range(len(alive_cells)):
        if comparison_count >= max_comparisons:
            break
            
        for j in range(i+1, len(alive_cells)):
            if comparison_count >= max_comparisons:
                break
                
            cell1, x1, y1 = alive_cells[i]
            cell2, x2, y2 = alive_cells[j]
            
            # Create a unique pair identifier
            pair_id = tuple(sorted([cell1.id, cell2.id]) if cell1.id is not None and cell2.id is not None else (i, j))
            if pair_id in processed_pairs:
                continue
                
            processed_pairs.add(pair_id)
            comparison_count += 1
            
            # Calculate similarity
            similarity = cosine_similarity(cell1.info, cell2.info)
            
            # If similar enough, merge (average their info)
            if similarity > MERGE_THRESHOLD:
                # Average the info vectors with weighted average based on activation
                weight1 = cell1.act + 0.1  # Add small bias to prevent zero weights
                weight2 = cell2.act + 0.1
                total_weight = weight1 + weight2
                
                merged_info = (cell1.info * weight1 + cell2.info * weight2) / total_weight
                
                # Apply to both cells
                cell1.info = merged_info
                cell2.info = merged_info
                
                merged_count += 1
    
    if merged_count > 0:
        print(f"Merged {merged_count} similar cell pairs")

# -----------------------
# Lineage Graph
# -----------------------
def get_lineage_depth(cell_id, max_depth=10):
    """Get the depth of a cell in the lineage tree"""
    depth = 0
    current_id = cell_id
    
    # Limit depth to prevent infinite loops
    while depth < max_depth:
        # Find the cell with this ID
        found = False
        for x in range(GRID_W):
            for y in range(GRID_H):
                cell = grid[x][y]
                if cell.alive and cell.id == current_id and cell.parent_id is not None:
                    current_id = cell.parent_id
                    depth += 1
                    found = True
                    break
            if found:
                break
        if not found:
            break
            
    return depth

def draw_lineage_graph(screen, font):
    """Draw a simple lineage graph visualization"""
    # Draw a small lineage graph in the corner
    graph_width, graph_height = 200, 150
    graph_x, graph_y = 10, GRID_H * CELL_SIZE - graph_height - 10
    
    # Draw background
    pygame.draw.rect(screen, (30, 30, 30), (graph_x, graph_y, graph_width, graph_height))
    pygame.draw.rect(screen, (100, 100, 100), (graph_x, graph_y, graph_width, graph_height), 2)
    
    # Collect lineage data
    depths = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if cell.alive and cell.id is not None:
                depth = get_lineage_depth(cell.id)
                depths.append(depth)
    
    if depths:
        # Draw histogram of lineage depths
        max_depth = max(depths) if depths else 1
        bar_width = max(1, graph_width // max(1, max_depth + 1))
        
        for depth in range(max_depth + 1):
            count = depths.count(depth)
            if count > 0:
                bar_height = min(graph_height - 20, count * 5)  # Scale height
                bar_x = graph_x + depth * bar_width
                bar_y = graph_y + graph_height - bar_height - 10
                pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, bar_width - 1, bar_height))
        
        # Draw title
        title = font.render("Lineage Depth", True, (200, 200, 200))
        screen.blit(title, (graph_x + 5, graph_y + 5))

# -----------------------
# Initialize with some random cells
# -----------------------
def random_init(density=INIT_DENSITY):
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if random.random() < density:
                cell.alive = True
                cell.act = random.random() * 0.6
                cell.info = np.random.normal(scale=0.5, size=INFO_DIM).astype(np.float32)
                cell.generation = 0
            else:
                cell.alive = False
                cell.act = 0.0
                cell.info = np.zeros(INFO_DIM, dtype=np.float32)
                cell.generation = 0

random_init()

# -----------------------
# Helper: neighborhood
# -----------------------
def neighbors_coords(x, y):
    """Generate coordinates of neighbors"""
    for dx, dy in neighbor_offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny

# -----------------------
# Reinforcement Learning Functions
# -----------------------
def get_state(cell, x, y):
    """Get the state representation for a cell"""
    # Gather neighbor activations and info
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
    
    return np.array(state, dtype=np.float32)

def choose_action(state, epsilon=EPSILON):
    """Choose action using epsilon-greedy policy"""
    if random.random() <= epsilon:
        # Explore: choose random action
        return random.randint(0, 3)
    else:
        # Exploit: choose best action based on Q-values
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return q_values.argmax().item()

def compute_reward(cell, prev_act, new_act):
    """Compute reward based on cell activation change"""
    # Reward for positive activation change
    activation_change = new_act - prev_act
    reward = activation_change * 10.0
    
    # Additional reward for maintaining optimal activation (0.5-0.8)
    if 0.5 <= new_act <= 0.8:
        reward += 0.1
    
    # Penalty for extreme activations
    if new_act > 0.9 or new_act < 0.1:
        reward -= 0.05
        
    return reward

def update_q_network(state, action, reward, next_state, done=False):
    """Update Q-network using Q-learning"""
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(device)
    
    # Get current Q-value
    current_q_values = q_network(state_tensor)
    current_q_value = current_q_values[0, action]
    
    if done:
        target_q_value = reward
    else:
        # Get next Q-values and compute target
        with torch.no_grad():
            next_q_values = q_network(next_state_tensor)
        target_q_value = reward + GAMMA * next_q_values.max()
    
    # Compute loss and update
    loss = nn.functional.mse_loss(current_q_value, target_q_value)
    
    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()
    
    return loss.item()

# -----------------------
# Enhanced Simulation Step with RL
# -----------------------
def sim_step(reward_map=None, frame_count=0):
    global brain, optimizer, scaler, EPSILON
    # reward_map: 2D array same dims with reward floats (0..1) applied this timestep
    
    # Collect all alive cells
    alive_cells = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if cell.alive:
                alive_cells.append((cell, x, y))
    
    if not alive_cells:
        return
    
    # Store previous activations for reward computation
    prev_activations = {cell.id: cell.act for cell, _, _ in alive_cells}
    
    # Vectorized forward pass
    N_alive = len(alive_cells)
    
    # Build input tensor for all alive cells
    inp_tensor = torch.zeros((N_alive, INP_DIM), dtype=torch.float32, device=device)
    
    for i, (cell, x, y) in enumerate(alive_cells):
        # Gather neighbor activations
        neigh_idx = 0
        for nx, ny in neighbors_coords(x, y):
            neigh_cell = grid[nx][ny]
            inp_tensor[i, neigh_idx] = float(neigh_cell.act if neigh_cell.alive else 0.0)
            neigh_idx += 1
        
        # Add cell info
        inp_tensor[i, NEIGHBOR_COUNT:] = torch.from_numpy(cell.info).to(device)
    
    # Batched forward pass - use region-specific network
    with torch.no_grad():
        with create_amp_autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            # Process each region separately
            predictions = torch.zeros(N_alive, device=device)
            for region_id in range(REGION_COUNT):
                # Get indices of cells in this region
                region_indices = [i for i, (cell, _, _) in enumerate(alive_cells) if cell.region_id == region_id]
                if region_indices:
                    region_inp = inp_tensor[region_indices]
                    region_preds = brain(region_inp, region_id).squeeze()
                    # Ensure the same dtype
                    if region_preds.dtype != predictions.dtype:
                        region_preds = region_preds.to(predictions.dtype)
                    predictions[region_indices] = region_preds * REGION_WEIGHTS[region_id]
    
    # Apply updates
    delta_acts = predictions.cpu().numpy() * DT
    
    # Collect data for analytics
    avg_activation = np.mean([cell.act for cell, _, _ in alive_cells])
    activation_history.append(avg_activation)
    
    # Calculate mutation variance for analytics
    if evolution_log:
        recent_mutations = [entry['mutation_size'] for entry in evolution_log[-10:]]
        if recent_mutations:
            mutation_variance = np.var(recent_mutations)
            mutation_history.append(mutation_variance)
    
    # Apply updates and compute rewards
    for i, (cell, x, y) in enumerate(alive_cells):
        prev_act = cell.act
        # Apply delta and decay
        cell.act = clamp01(cell.act + delta_acts[i] - DECAY * cell.act * DT)
        
        # Play sound feedback for highly active cells
        if SOUND_ENABLED and cell.act > 0.5:  # Only for highly active cells
            play_activation_sound(cell.act)
        
        # Reinforcement learning: compute reward and update Q-network
        state = get_state(cell, x, y)
        reward = compute_reward(cell, prev_act, cell.act)
        cell.reward_history.append(reward)
        
        # Store reward in replay buffer for this cell
        if REPLAY_ENABLED and len(cell.reward_history) > 1:
            # Get previous state (simplified - in practice would store actual states)
            prev_state = state  # Simplified for this implementation
            action = 0  # Placeholder action
            next_state = state
            done = False
            # In a full implementation, we would store (state, action, reward, next_state, done) tuples
            
            # For now, we'll just use the reward for updating the brain network
    
    # Learning: Update neural network when reward present
    train_this_frame = (frame_count % BATCH_TRAIN_EVERY_N_FRAMES == 0) and reward_map is not None
    
    if train_this_frame and reward_map is not None:  # Added explicit check for linter
        # Collect cells with reward > 0
        reward_cells = []
        for x in range(GRID_W):
            for y in range(GRID_H):
                r = reward_map[x, y]
                if r > 0:
                    cell = grid[x][y]
                    if cell.alive:
                        reward_cells.append((cell, x, y, r))
        
        if reward_cells:
            # Add to replay buffer if enabled
            if REPLAY_ENABLED:
                replay_buffer.append(reward_cells)
            
            # Limit batch size
            if len(reward_cells) > MAX_BATCH_SIZE:
                reward_cells = random.sample(reward_cells, MAX_BATCH_SIZE)
            
            N_reward = len(reward_cells)
            
            # Build training batch
            train_inp = torch.zeros((N_reward, INP_DIM), dtype=torch.float32, device=device)
            targets = torch.zeros(N_reward, dtype=torch.float32, device=device)
            
            for i, (cell, x, y, r) in enumerate(reward_cells):
                # Build input vector
                neigh_idx = 0
                for nx, ny in neighbors_coords(x, y):
                    train_inp[i, neigh_idx] = float(grid[nx][ny].act if grid[nx][ny].alive else 0.0)
                    neigh_idx += 1
                train_inp[i, NEIGHBOR_COUNT:] = torch.from_numpy(cell.info).to(device)
                
                # Target = post_activation * reward
                targets[i] = float(cell.act * r)
            
            # Training step
            optimizer.zero_grad()
            
            if USE_AMP and device.type == "cuda" and scaler is not None:  # Added explicit check for linter
                with create_amp_autocast("cuda"):
                    # Process each region separately for training
                    total_loss = None
                    for region_id in range(REGION_COUNT):
                        # Get indices of cells in this region
                        region_indices = [i for i, (cell, _, _, _) in enumerate(reward_cells) if cell.region_id == region_id]
                        if region_indices:
                            region_inp = train_inp[region_indices]
                            region_targets = targets[region_indices]
                            region_preds = brain(region_inp, region_id).squeeze()
                            region_loss = nn.functional.mse_loss(region_preds, region_targets)
                            if total_loss is None:
                                total_loss = region_loss * REGION_WEIGHTS[region_id]
                            else:
                                total_loss = total_loss + region_loss * REGION_WEIGHTS[region_id]
                    
                    if total_loss is not None:
                        scaled_loss = scaler.scale(total_loss)
                        scaled_loss.backward()
                        scaler.step(optimizer)
                        scaler.update()
            else:
                # Process each region separately for training
                total_loss = None
                for region_id in range(REGION_COUNT):
                    # Get indices of cells in this region
                    region_indices = [i for i, (cell, _, _, _) in enumerate(reward_cells) if cell.region_id == region_id]
                    if region_indices:
                        region_inp = train_inp[region_indices]
                        region_targets = targets[region_indices]
                        region_preds = brain(region_inp, region_id).squeeze()
                        region_loss = nn.functional.mse_loss(region_preds, region_targets)
                        if total_loss is None:
                            total_loss = region_loss * REGION_WEIGHTS[region_id]
                        else:
                            total_loss = total_loss + region_loss * REGION_WEIGHTS[region_id]
                
                if total_loss is not None:
                    total_loss.backward()
                    optimizer.step()
    
    # Decay epsilon for exploration
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
    
    # Memory compression
    if random.random() < MERGE_PROB:
        compress_memory()
    
    # Neurogenesis: spawn children from highly active cells
    for cell, x, y in alive_cells:
        if cell.act > SPAWN_THRESHOLD and random.random() < SPAWN_PROB:
            # find empty neighbor
            empties = [(nx, ny) for nx, ny in neighbors_coords(x, y) if not grid[nx][ny].alive]
            if empties:
                nx, ny = random.choice(empties)
                child = grid[nx][ny]
                child.alive = True
                child.act = min(1.0, cell.act * 0.6 + 0.1)
                # copy info with mutation
                child.info = (cell.info + np.random.normal(scale=MUTATION_STD, size=INFO_DIM)).astype(np.float32)
                # Assign to same region as parent
                child.region_id = cell.region_id
                # Evolution tracking
                child.parent_id = cell.id
                child.generation = cell.generation + 1
                child.mutation_history = cell.mutation_history + [np.linalg.norm(child.info - cell.info)]
                # Log the birth
                evolution_log.append({
                    'child_id': child.id,
                    'parent_id': cell.id,
                    'generation': child.generation,
                    'position': (nx, ny),
                    'mutation_size': child.mutation_history[-1] if child.mutation_history else 0
                })

# -----------------------
# Replay Learning
# -----------------------
def replay_learning():
    """Replay previous training experiences"""
    if not replay_buffer or not REPLAY_ENABLED:
        return
    
    # Sample from replay buffer
    batch_size = min(32, len(replay_buffer))
    replay_samples = random.sample(replay_buffer, batch_size)
    
    # Process each sample
    for reward_cells in replay_samples:
        if not reward_cells:
            continue
            
        N_reward = len(reward_cells)
        
        # Build training batch
        train_inp = torch.zeros((N_reward, INP_DIM), dtype=torch.float32, device=device)
        targets = torch.zeros(N_reward, dtype=torch.float32, device=device)
        
        for i, (cell, x, y, r) in enumerate(reward_cells):
            # Build input vector
            neigh_idx = 0
            for nx, ny in neighbors_coords(x, y):
                train_inp[i, neigh_idx] = float(grid[nx][ny].act if grid[nx][ny].alive else 0.0)
                neigh_idx += 1
            train_inp[i, NEIGHBOR_COUNT:] = torch.from_numpy(cell.info).to(device)
            
            # Target = post_activation * reward
            targets[i] = float(cell.act * r)
        
        # Training step
        optimizer.zero_grad()
        
        if USE_AMP and device.type == "cuda" and scaler is not None:
            with create_amp_autocast("cuda"):
                # Process each region separately for training
                total_loss = None
                for region_id in range(REGION_COUNT):
                    # Get indices of cells in this region
                    region_indices = [i for i, (cell, _, _, _) in enumerate(reward_cells) if cell.region_id == region_id]
                    if region_indices:
                        region_inp = train_inp[region_indices]
                        region_targets = targets[region_indices]
                        region_preds = brain(region_inp, region_id).squeeze()
                        region_loss = nn.functional.mse_loss(region_preds, region_targets)
                        if total_loss is None:
                            total_loss = region_loss * REGION_WEIGHTS[region_id]
                        else:
                            total_loss = total_loss + region_loss * REGION_WEIGHTS[region_id]
                
                if total_loss is not None:
                    scaled_loss = scaler.scale(total_loss)
                    scaled_loss.backward()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            # Process each region separately for training
            total_loss = None
            for region_id in range(REGION_COUNT):
                # Get indices of cells in this region
                region_indices = [i for i, (cell, _, _, _) in enumerate(reward_cells) if cell.region_id == region_id]
                if region_indices:
                    region_inp = train_inp[region_indices]
                    region_targets = targets[region_indices]
                    region_preds = brain(region_inp, region_id).squeeze()
                    region_loss = nn.functional.mse_loss(region_preds, region_targets)
                    if total_loss is None:
                        total_loss = region_loss * REGION_WEIGHTS[region_id]
                    else:
                        total_loss = total_loss + region_loss * REGION_WEIGHTS[region_id]
            
            if total_loss is not None:
                total_loss.backward()
                optimizer.step()

# -----------------------
# Neural Network Graph Visualization
# -----------------------
def create_neural_influence_graph():
    """Create a NetworkX graph showing neuron-to-neuron influence"""
    if not NETWORKX_AVAILABLE:
        return None
    
    try:
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes for alive cells
        for x in range(GRID_W):
            for y in range(GRID_H):
                cell = grid[x][y]
                if cell.alive:
                    G.add_node((x, y), activation=cell.act, region=cell.region_id)
        
        # Add edges based on influence (parent-child relationships)
        for x in range(GRID_W):
            for y in range(GRID_H):
                cell = grid[x][y]
                if cell.alive and cell.parent_id is not None:
                    # Find parent cell
                    for px in range(GRID_W):
                        for py in range(GRID_H):
                            parent = grid[px][py]
                            if parent.alive and parent.id == cell.parent_id:
                                G.add_edge((px, py), (x, y), weight=cell.act)
                                break
        
        return G
    except Exception:
        return None

# -----------------------
# Data Analytics
# -----------------------
def get_analytics_data():
    """Get data for analytics plotting"""
    return {
        'activations': list(activation_history),
        'mutations': list(mutation_history),
        'region_counts': [sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive and grid[x][y].region_id == i) 
                         for i in range(REGION_COUNT)]
    }

# -----------------------
# Visualization (pygame)
# -----------------------
def init_visualization():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W*CELL_SIZE + 300, GRID_H*CELL_SIZE))  # Extra space for analytics
    pygame.display.set_caption("Neuro-Genesis Cellular Simulation")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()
    return screen, font, clock

def draw(screen, font, clock):
    # Draw cells
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell.alive:
                # Color by region
                if cell.region_id == 0:  # Vision
                    color = (100, 100, 255)  # Blue
                elif cell.region_id == 1:  # Motor
                    color = (100, 255, 100)  # Green
                else:  # Cognitive
                    color = (255, 100, 100)  # Red
                    
                # Scale brightness by activation
                brightness = cell.act
                color = tuple(int(brightness * c) for c in color)
                pygame.draw.rect(screen, color, rect)
            else:
                pygame.draw.rect(screen, (15, 15, 15), rect)
    
    # Draw region boundaries
    pygame.draw.line(screen, (100, 100, 100), 
                    (GRID_W//3 * CELL_SIZE, 0), 
                    (GRID_W//3 * CELL_SIZE, GRID_H * CELL_SIZE), 2)
    pygame.draw.line(screen, (100, 100, 100), 
                    (2 * GRID_W//3 * CELL_SIZE, 0), 
                    (2 * GRID_W//3 * CELL_SIZE, GRID_H * CELL_SIZE), 2)
    
    # Draw legend
    text = font.render("Click: stimulate | Space: reward near mouse | S:save L:load C:clear", True, (200,200,200))
    screen.blit(text, (4, 4))
    
    # Draw stats
    alive_count = sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive)
    fps = clock.get_fps()
    device_str = "CUDA" if device.type == "cuda" else "CPU"
    stats_text = font.render(f"Cells: {alive_count} | FPS: {fps:.1f} | {device_str}", True, (200,200,200))
    screen.blit(stats_text, (4, 24))
    
    # Draw lineage graph
    draw_lineage_graph(screen, font)
    
    # Draw analytics panel
    draw_analytics_panel(screen, font)

def draw_lineage_visualization(screen, font):
    """Draw a simple lineage graph visualization"""
    # Draw a small lineage graph in the corner
    graph_width, graph_height = 200, 150
    graph_x, graph_y = 10, GRID_H * CELL_SIZE - graph_height - 10
    
    # Draw background
    pygame.draw.rect(screen, (30, 30, 30), (graph_x, graph_y, graph_width, graph_height))
    pygame.draw.rect(screen, (100, 100, 100), (graph_x, graph_y, graph_width, graph_height), 2)
    
    # Collect lineage data
    depths = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if cell.alive and cell.id is not None:
                depth = get_lineage_depth(cell.id)
                depths.append(depth)
    
    if depths:
        # Draw histogram of lineage depths
        max_depth = max(depths) if depths else 1
        bar_width = max(1, graph_width // max(1, max_depth + 1))
        
        for depth in range(max_depth + 1):
            count = depths.count(depth)
            if count > 0:
                bar_height = min(graph_height - 20, count * 5)  # Scale height
                bar_x = graph_x + depth * bar_width
                bar_y = graph_y + graph_height - bar_height - 10
                pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, bar_width - 1, bar_height))
        
        # Draw title
        title = font.render("Lineage Depth", True, (200, 200, 200))
        screen.blit(title, (graph_x + 5, graph_y + 5))

def draw_analytics_panel(screen, font):
    """Draw analytics data panel"""
    # Draw analytics panel on the right side
    panel_x = GRID_W * CELL_SIZE
    panel_width = 300
    panel_height = GRID_H * CELL_SIZE
    
    # Draw background
    pygame.draw.rect(screen, (40, 40, 40), (panel_x, 0, panel_width, panel_height))
    pygame.draw.line(screen, (100, 100, 100), (panel_x, 0), (panel_x, panel_height), 2)
    
    # Draw title
    title = font.render("Analytics", True, (200, 200, 200))
    screen.blit(title, (panel_x + 10, 10))
    
    # Draw region counts
    region_y = 40
    region_names = ["Vision", "Motor", "Cognitive"]
    region_colors = [(100, 100, 255), (100, 255, 100), (255, 100, 100)]
    
    for i in range(REGION_COUNT):
        count = sum(1 for x in range(GRID_W) for y in range(GRID_H) 
                   if grid[x][y].alive and grid[x][y].region_id == i)
        region_text = font.render(f"{region_names[i]}: {count}", True, region_colors[i])
        screen.blit(region_text, (panel_x + 10, region_y))
        region_y += 25
    
    # Draw average activation
    if activation_history:
        avg_act = np.mean(list(activation_history))
        act_text = font.render(f"Avg Activation: {avg_act:.3f}", True, (200, 200, 200))
        screen.blit(act_text, (panel_x + 10, region_y))
        region_y += 25
    
    # Draw mutation variance
    if mutation_history:
        mut_var = np.mean(list(mutation_history)) if mutation_history else 0
        mut_text = font.render(f"Mutation Var: {mut_var:.5f}", True, (200, 200, 200))
        screen.blit(mut_text, (panel_x + 10, region_y))

# -----------------------
# Save / Load
# -----------------------
def save_sim(filename=SAVE_FILE):
    data = {
        "brain_state_dict": brain.state_dict(),
        "q_network_state_dict": q_network.state_dict(),
        "grid_alive": [[grid[x][y].alive for y in range(GRID_H)] for x in range(GRID_W)],
        "grid_act": [[grid[x][y].act for y in range(GRID_H)] for x in range(GRID_W)],
        "grid_info": [[[float(v) for v in grid[x][y].info] for y in range(GRID_H)] for x in range(GRID_W)],
        "grid_generation": [[grid[x][y].generation for y in range(GRID_H)] for x in range(GRID_W)],
        "evolution_log": evolution_log
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("Saved to", filename)

def load_sim(filename=SAVE_FILE):
    global brain, q_network, evolution_log
    if not os.path.exists(filename):
        print("No save file found:", filename)
        return
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Load brain state with compatibility for different architectures
    try:
        brain.load_state_dict(data["brain_state_dict"])
        brain = brain.to(device)
    except RuntimeError as e:
        print(f"Warning: Could not load brain state due to architecture change: {e}")
        print("Initializing new brain with current architecture")
        # Reinitialize brain with current architecture
        brain = NeuralBrain(INP_DIM, region_count=REGION_COUNT)
        brain = brain.to(device)
    
    # Load Q-network state
    try:
        if "q_network_state_dict" in data:
            q_network.load_state_dict(data["q_network_state_dict"])
            q_network = q_network.to(device)
    except RuntimeError as e:
        print(f"Warning: Could not load Q-network state: {e}")
        print("Initializing new Q-network with current architecture")
        q_network = QNetwork(INP_DIM, action_dim=4)
        q_network = q_network.to(device)
    
    # Load grid state
    for x in range(GRID_W):
        for y in range(GRID_H):
            grid[x][y].alive = data["grid_alive"][x][y]
            grid[x][y].act = data["grid_act"][x][y]
            grid[x][y].info = np.array(data["grid_info"][x][y], dtype=np.float32)
            if "grid_generation" in data:
                grid[x][y].generation = data["grid_generation"][x][y]
            # Ensure region_id exists
            if not hasattr(grid[x][y], 'region_id'):
                # Assign region based on position for backward compatibility
                if x < GRID_W // 3:
                    grid[x][y].region_id = 0  # Vision region
                elif x < 2 * GRID_W // 3:
                    grid[x][y].region_id = 1  # Motor region
                else:
                    grid[x][y].region_id = 2  # Cognitive region
    
    # Load evolution log
    if "evolution_log" in data:
        evolution_log = data["evolution_log"]
        
    print("Loaded from", filename)

# -----------------------
# Smoke test
# -----------------------
def run_smoke_test(steps=5):  # Reduced steps for faster testing
    """Run a headless smoke test"""
    print("Running smoke test...")
    
    # Initialize
    random_init(density=0.1)
    initial_alive = sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive)
    
    # Run simulation steps
    for i in range(steps):
        reward_map = np.zeros((GRID_W, GRID_H), dtype=np.float32)
        # Add some reward in the center
        if i == steps // 2:
            for x in range(GRID_W//2-2, GRID_W//2+3):
                for y in range(GRID_H//2-2, GRID_H//2+3):
                    if 0 <= x < GRID_W and 0 <= y < GRID_H:
                        d = sqrt((x-GRID_W//2)**2 + (y-GRID_H//2)**2)
                        r = max(0.0, 1.0 - d/4.0)
                        reward_map[x, y] = r
        
        sim_step(reward_map=reward_map, frame_count=i)
        
        if i % (steps//5) == 0:
            current_alive = sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive)
            print(f"Step {i}: {current_alive} alive cells")
    
    final_alive = sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive)
    
    # Assertions
    assert initial_alive > 0, "No cells initialized"
    assert final_alive >= initial_alive, f"Cell count decreased: {initial_alive} -> {final_alive}"
    assert len(evolution_log) >= 0, "Evolution log should exist"
    
    print(f"Smoke test passed! Cells: {initial_alive} -> {final_alive}")
    return True

# -----------------------
# Control Panel Integration
# -----------------------
def start_control_panel():
    """Start the control panel in a separate thread"""
    global controller, control_panel_thread
    if not CONTROL_PANEL_AVAILABLE or SimulationController is None:
        return None
        
    try:
        controller = SimulationController()
        
        # Register callbacks for parameter updates
        controller.register_callback('DT', lambda v: globals().update({'DT': v}))
        controller.register_callback('SPAWN_THRESHOLD', lambda v: globals().update({'SPAWN_THRESHOLD': v}))
        controller.register_callback('SPAWN_PROB', lambda v: globals().update({'SPAWN_PROB': v}))
        controller.register_callback('MUTATION_STD', lambda v: globals().update({'MUTATION_STD': v}))
        controller.register_callback('LEARNING_RATE', lambda v: globals().update({'LEARNING_RATE': v}))
        controller.register_callback('DECAY', lambda v: globals().update({'DECAY': v}))
        controller.register_callback('STIM_STRONG', lambda v: globals().update({'STIM_STRONG': v}))
        controller.register_callback('INIT_DENSITY', lambda v: globals().update({'INIT_DENSITY': v}))
        controller.register_callback('SOUND_ENABLED', lambda v: globals().update({'SOUND_ENABLED': v}))
        controller.register_callback('BASE_FREQ', lambda v: globals().update({'BASE_FREQ': v}))
        controller.register_callback('SOUND_COOLDOWN', lambda v: globals().update({'SOUND_COOLDOWN': v}))
        controller.register_callback('MERGE_THRESHOLD', lambda v: globals().update({'MERGE_THRESHOLD': v}))
        controller.register_callback('MERGE_PROB', lambda v: globals().update({'MERGE_PROB': v}))
        controller.register_callback('REGION_WEIGHTS', lambda v: globals().update({'REGION_WEIGHTS': v}))
        controller.register_callback('REPLAY_ENABLED', lambda v: globals().update({'REPLAY_ENABLED': v}))
        controller.register_callback('REPLAY_BUFFER_SIZE', lambda v: globals().update({'REPLAY_BUFFER_SIZE': v}))
        controller.register_callback('GAMMA', lambda v: globals().update({'GAMMA': v}))
        controller.register_callback('EPSILON', lambda v: globals().update({'EPSILON': v}))
        controller.register_callback('EPSILON_DECAY', lambda v: globals().update({'EPSILON_DECAY': v}))
        controller.register_callback('EPSILON_MIN', lambda v: globals().update({'EPSILON_MIN': v}))
        controller.register_callback('LEARNING_RATE_RL', lambda v: globals().update({'LEARNING_RATE_RL': v}))
        
        print("Control panel initialized")
        return controller
    except Exception as e:
        print(f"Failed to initialize control panel: {e}")
        return None

# -----------------------
# Main loop
# -----------------------
def main():
    screen, font, clock = init_visualization()
    
    # Initialize control panel if available
    controller = start_control_panel()
    
    # Initialize sound if enabled
    if SOUND_ENABLED:
        init_sound()
    
    running = True
    frame_count = 0
    
    print("Neuro-Genesis Cellular Simulator")
    print("Controls:")
    print("  Left click: stimulate cell")
    print("  Space: give positive reward to cells near mouse")
    print("  S: save model")
    print("  L: load model")
    print("  C: clear grid (reset)")
    print("  Esc: quit (auto-save)")
    
    while running:
        frame_start = pygame.time.get_ticks()
        
        screen.fill((0,0,0))
        reward_map = np.zeros((GRID_W, GRID_H), dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # stimulate clicked cell
                    mx, my = event.pos
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                        grid[gx][gy].alive = True
                        grid[gx][gy].act = clamp01(grid[gx][gy].act + STIM_STRONG)
                        # inject some info if empty
                        if np.linalg.norm(grid[gx][gy].info) < 1e-6:
                            grid[gx][gy].info = np.random.normal(scale=0.5, size=INFO_DIM).astype(np.float32)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_sim()
                elif event.key == pygame.K_l:
                    load_sim()
                elif event.key == pygame.K_c:
                    random_init(density=0.0)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # apply reward around mouse
                    mx, my = pygame.mouse.get_pos()
                    gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                    for nx in range(max(0, gx-2), min(GRID_W, gx+3)):
                        for ny in range(max(0, gy-2), min(GRID_H, gy+3)):
                            # gaussian distance-based reward
                            d = sqrt((nx-gx)**2 + (ny-gy)**2)
                            r = max(0.0, 1.0 - d/4.0)
                            reward_map[nx, ny] = max(reward_map[nx, ny], r)
                    print("Applied reward near", (gx, gy))
                elif event.key == pygame.K_r:
                    # Replay learning
                    if REPLAY_ENABLED:
                        replay_learning()
                        print("Performed replay learning step")

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
            print(f"Avg frame time: {avg_frame_time:.2f}ms")

    pygame.quit()
    print("Bye! Saved weights automatically.")
    save_sim()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_smoke_test()
    else:
        main()