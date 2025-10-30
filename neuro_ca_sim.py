# File: neuro_ca_sim.py
# Simple Neuro-Genesis Cellular Simulator
# Run: python neuro_ca_sim.py
# Controls:
#  - Left click: stimulate cell
#  - Space: give positive reward to cells near mouse
#  - S: save model (sim_save.pkl)
#  - L: load model
#  - C: clear grid (reset)
#  - Esc or close window: quit
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
import pygame.sndarray
import pygame.mixer

# -----------------------
# Parameters (tweakable)
# -----------------------
GRID_W, GRID_H = 64, 48           # grid (columns x rows)
CELL_SIZE = 12                    # pixels
INFO_DIM = 6                      # dimension of info vector stored per cell
NEIGHBOR_RADIUS = 1               # neighborhood radius (1 => 3x3)
DT = 0.1                          # simulation time step (affects update scale)
SPAWN_THRESHOLD = 0.85            # activation threshold to spawn new cell
SPAWN_PROB = 0.35                 # probability to spawn when threshold met
MUTATION_STD = 0.05               # noise added to child info
LEARNING_RATE = 0.002             # learning rate
DECAY = 0.01                      # activation decay per step
STIM_STRONG = 0.9                 # activation added by clicking
INIT_DENSITY = 0.06               # initial fractional occupancy
SAVE_FILE = "sim_save.pkl"

# Neural network parameters
HIDDEN_DIM = 32               # hidden layer size for the neural network
USE_CUDA = torch.cuda.is_available()

# Memory compression parameters
MERGE_THRESHOLD = 0.9         # cosine similarity threshold for merging
MERGE_PROB = 0.01             # probability of checking for merges each step

# Sound parameters
SOUND_ENABLED = False  # Disable sound by default to avoid irritation
BASE_FREQ = 220               # base frequency for sound feedback
SOUND_COOLDOWN = 10           # frames between sounds
last_sound_frame = 0          # track when last sound was played

# -----------------------
# Utilities
# -----------------------
def clamp01(x): return max(0.0, min(1.0, x))

def info_to_color(vec):
    # map info vector to RGB via simple projection
    # normalize
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
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(NeuralBrain, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.network(x)

# -----------------------
# Grid / Cell structure
# -----------------------
class Cell:
    def __init__(self, id=None):
        self.alive = False
        self.act = 0.0
        self.info: np.ndarray = np.zeros(INFO_DIM, dtype=float)
        self.id = id
        # Evolution tracking
        self.parent_id = None
        self.generation = 0
        self.mutation_history = []

# initialize grid
grid = [[Cell(r*GRID_W + c) for r in range(GRID_H)] for c in range(GRID_W)]

# Initialize neural network brain instead of weight vector
INP_DIM = ((2*NEIGHBOR_RADIUS+1)**2) + INFO_DIM  # neighbor acts flattened + cell info
brain = NeuralBrain(INP_DIM, HIDDEN_DIM)

# Move to GPU if available
if USE_CUDA:
    brain = brain.cuda()

# Optimizer for learning
optimizer = torch.optim.Adam(brain.parameters(), lr=LEARNING_RATE)

# Evolution log
evolution_log = []

# -----------------------
# Initialize with some random cells
# -----------------------
def random_init(density=INIT_DENSITY):
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if random.random() < density:
                cell.alive = True
                cell.act = random.random()*0.6
                cell.info = np.random.normal(scale=0.5, size=INFO_DIM)
                cell.generation = 0
            else:
                cell.alive = False
                cell.act = 0.0
                cell.info = np.zeros(INFO_DIM)
                cell.generation = 0

random_init()

# -----------------------
# Helper: neighborhood
# -----------------------
def neighbors_coords(x, y):
    for dx in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
        for dy in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS+1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                yield nx, ny

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
    """Merge similar info vectors"""
    # Collect all alive cells
    alive_cells = []
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if cell.alive:
                alive_cells.append((cell, x, y))
    
    # Compare pairs of cells for similarity
    merged_count = 0
    for i in range(len(alive_cells)):
        for j in range(i+1, len(alive_cells)):
            cell1, x1, y1 = alive_cells[i]
            cell2, x2, y2 = alive_cells[j]
            
            # Calculate similarity
            similarity = cosine_similarity(cell1.info, cell2.info)
            
            # If similar enough, merge (average their info)
            if similarity > MERGE_THRESHOLD:
                # Average the info vectors
                merged_info = (cell1.info + cell2.info) / 2
                
                # Apply to both cells
                cell1.info = merged_info
                cell2.info = merged_info
                
                merged_count += 1
    
    if merged_count > 0:
        print(f"Merged {merged_count} similar cell pairs")

# -----------------------
# Sound Feedback
# -----------------------
def init_sound():
    """Initialize sound system"""
    if SOUND_ENABLED:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            return True
        except pygame.error:
            print("Sound initialization failed")
            return False
    return False

def play_activation_sound(activation_level):
    """Play a tone with pitch based on activation level"""
    global last_sound_frame
    if not SOUND_ENABLED:
        return
        
    # Limit sound frequency
    import pygame
    current_frame = pygame.time.get_ticks()
    if current_frame - last_sound_frame < SOUND_COOLDOWN:
        return
    
    try:
        # Map activation (0-1) to frequency (base_freq to base_freq*2)
        freq = BASE_FREQ + activation_level * BASE_FREQ
        
        # Generate a simple sine wave
        duration = 20  # ms - shorter duration
        sample_rate = 22050
        frames = int(duration * sample_rate / 1000)
        
        # Reduce volume
        arr = np.zeros((frames, 2))
        for i in range(frames):
            wave_value = 2048 * np.sin(2 * np.pi * freq * i / sample_rate)  # Lower amplitude
            arr[i][0] = wave_value  # left channel
            arr[i][1] = wave_value  # right channel
            
        sound = pygame.sndarray.make_sound(arr.astype(np.int16))
        sound.play()
        last_sound_frame = current_frame
    except:
        pass  # Ignore sound errors

# -----------------------
# Simulation step
# -----------------------
def sim_step(reward_map=None):
    global brain, optimizer
    # reward_map: 2D array same dims with reward floats (0..1) applied this timestep
    delta_acts = np.zeros((GRID_W, GRID_H), dtype=float)

    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if not cell.alive:
                continue
            # gather neighbor activations
            neigh = []
            for nx, ny in neighbors_coords(x, y):
                neigh_cell = grid[nx][ny]
                neigh.append(neigh_cell.act if neigh_cell.alive else 0.0)
            neigh = np.array(neigh)
            # Ensure the input vector has the correct dimension
            if len(neigh) != (2*NEIGHBOR_RADIUS+1)**2:
                # Pad with zeros if necessary
                padded_neigh = np.zeros((2*NEIGHBOR_RADIUS+1)**2)
                padded_neigh[:len(neigh)] = neigh
                neigh = padded_neigh
            inp = np.concatenate([neigh, cell.info])
            
            # Run through neural network
            inp_tensor = torch.FloatTensor(inp).unsqueeze(0)
            if USE_CUDA:
                inp_tensor = inp_tensor.cuda()
            d_act = brain(inp_tensor).item() * DT
            
            # small decay
            d_act -= DECAY * cell.act * DT
            delta_acts[x, y] = d_act

    # apply deltas
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if not cell.alive:
                continue
            cell.act = clamp01(cell.act + delta_acts[x, y])
            
            # Play sound feedback less frequently
            if SOUND_ENABLED and cell.act > 0.3:  # Increased threshold
                play_activation_sound(cell.act)

    # Learning: Update neural network when reward present
    if reward_map is not None:
        total_loss = 0.0
        loss_count = 0
        
        for x in range(GRID_W):
            for y in range(GRID_H):
                r = reward_map[x, y]
                if r <= 0:
                    continue
                cell = grid[x][y]
                if not cell.alive:
                    continue
                # build pre vector (neighbors + info)
                pre = []
                for nx, ny in neighbors_coords(x, y):
                    pre.append(grid[nx][ny].act if grid[nx][ny].alive else 0.0)
                # Ensure the pre vector has the correct dimension
                if len(pre) != (2*NEIGHBOR_RADIUS+1)**2:
                    # Pad with zeros if necessary
                    padded_pre = np.zeros((2*NEIGHBOR_RADIUS+1)**2)
                    padded_pre[:len(pre)] = pre
                    pre = padded_pre
                pre = np.concatenate([np.array(pre), cell.info])
                
                # Neural network learning
                pre_tensor = torch.FloatTensor(pre).unsqueeze(0)
                if USE_CUDA:
                    pre_tensor = pre_tensor.cuda()
                    
                # Calculate loss (negative because we want to maximize reward)
                post = cell.act
                prediction = brain(pre_tensor).item()
                loss = -r * post * prediction
                
                total_loss += loss
                loss_count += 1
        
        # Apply learning update
        if loss_count > 0:
            optimizer.zero_grad()
            avg_loss = torch.tensor(total_loss / loss_count, requires_grad=True)
            avg_loss.backward()
            optimizer.step()

    # Memory compression
    if random.random() < MERGE_PROB:
        compress_memory()

    # Neurogenesis: spawn children from highly active cells
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            if not cell.alive:
                continue
            if cell.act > SPAWN_THRESHOLD and random.random() < SPAWN_PROB:
                # find empty neighbor
                empties = [(nx, ny) for nx, ny in neighbors_coords(x, y) if not grid[nx][ny].alive]
                if empties:
                    nx, ny = random.choice(empties)
                    child = grid[nx][ny]
                    child.alive = True
                    child.act = min(1.0, cell.act * 0.6 + 0.1)
                    # copy info with mutation
                    child.info = cell.info + np.random.normal(scale=MUTATION_STD, size=INFO_DIM)
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
# Visualization (pygame)
# -----------------------
pygame.init()
screen = pygame.display.set_mode((GRID_W*CELL_SIZE, GRID_H*CELL_SIZE))
pygame.display.set_caption("Neuro-Genesis Cellular Simulation")
font = pygame.font.SysFont("Arial", 16)

# Initialize sound
sound_initialized = init_sound()

clock = pygame.time.Clock()

def draw():
    # Draw cells
    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = grid[x][y]
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if cell.alive:
                # color by info for hue, multiply by activation for brightness
                base = info_to_color(cell.info)
                # scale brightness by activation
                brightness = cell.act
                color = tuple(int(brightness * c) for c in base)
                pygame.draw.rect(screen, color, rect)
                
                # Draw synapses to neighbors with high activation
                if cell.act > 0.5:  # Only draw synapses for highly active neurons
                    for nx, ny in neighbors_coords(x, y):
                        neighbor = grid[nx][ny]
                        if neighbor.alive and neighbor.act > 0.3:
                            # Draw line between cells
                            start_pos = (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2)
                            end_pos = (nx*CELL_SIZE + CELL_SIZE//2, ny*CELL_SIZE + CELL_SIZE//2)
                            # Color based on activation difference
                            synapse_strength = abs(cell.act - neighbor.act)
                            synapse_color = (int(255*synapse_strength), int(100*synapse_strength), int(200*synapse_strength))
                            pygame.draw.line(screen, synapse_color, start_pos, end_pos, 1)
            else:
                pygame.draw.rect(screen, (15, 15, 15), rect)
    
    # Draw legend
    text = font.render("Click: stimulate | Space: reward near mouse | S:save L:load C:clear", True, (200,200,200))
    screen.blit(text, (4, 4))
    
    # Draw stats
    alive_count = sum(1 for x in range(GRID_W) for y in range(GRID_H) if grid[x][y].alive)
    stats_text = font.render(f"Cells: {alive_count} | Gen: {len(evolution_log)}", True, (200,200,200))
    screen.blit(stats_text, (4, 24))

# -----------------------
# Save / Load
# -----------------------
def save_sim(filename=SAVE_FILE):
    data = {
        "brain_state_dict": brain.state_dict(),
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
    global brain, evolution_log
    if not os.path.exists(filename):
        print("No save file found:", filename)
        return
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Load brain state
    brain.load_state_dict(data["brain_state_dict"])
    if USE_CUDA:
        brain = brain.cuda()
    
    # Load grid state
    for x in range(GRID_W):
        for y in range(GRID_H):
            grid[x][y].alive = data["grid_alive"][x][y]
            grid[x][y].act = data["grid_act"][x][y]
            grid[x][y].info = np.array(data["grid_info"][x][y], dtype=float)
            if "grid_generation" in data:
                grid[x][y].generation = data["grid_generation"][x][y]
    
    # Load evolution log
    if "evolution_log" in data:
        evolution_log = data["evolution_log"]
        
    print("Loaded from", filename)

# -----------------------
# Main loop
# -----------------------
running = True
mouse_pos = (0,0)
while running:
    clock.tick(30)
    screen.fill((0,0,0))
    reward_map = np.zeros((GRID_W, GRID_H), dtype=float)

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
                        grid[gx][gy].info = np.random.normal(scale=0.5, size=INFO_DIM)
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

    # Every frame, do a sim step with current reward_map
    sim_step(reward_map=reward_map)

    draw()
    pygame.display.flip()

pygame.quit()
print("Bye! Saved weights automatically.")
save_sim()