# Neuro-Genesis Architecture Documentation

## System Overview

Neuro-Genesis is a sophisticated neural network simulator that combines cellular automaton principles with reinforcement learning to create an emergent intelligent system.

## Core Architecture

### 1. Cellular Grid System

```
┌─────────────────────────────────────┐
│         64 x 48 Grid                │
│  ┌──────┬──────┬──────┬──────┐     │
│  │Vision│Vision│Vision│Vision│     │
│  ├──────┼──────┼──────┼──────┤     │
│  │Motor │Motor │Motor │Motor │     │
│  ├──────┼──────┼──────┼──────┤     │
│  │Cognitive│Cognitive│Cognitive│   │
│  └──────┴──────┴──────┴──────┘     │
└─────────────────────────────────────┘
```

**Grid Properties:**
- Dimensions: 64 (width) × 48 (height)
- Total cells: 3,072 neurons
- Each cell represents a neuron
- Moore neighborhood (8-connected)

### 2. Brain Region Architecture

#### Vision Region (Region 0)
- **Location**: Top portion of grid
- **Function**: Primary sensory input processing
- **Characteristics**:
  - High receptive field density
  - Fast activation dynamics
  - Input pattern recognition

#### Motor Region (Region 1)
- **Location**: Middle portion of grid
- **Function**: Action generation and output
- **Characteristics**:
  - Output command generation
  - Action coordination
  - Temporal sequencing

#### Cognitive Region (Region 2)
- **Location**: Bottom portion of grid
- **Function**: Higher-level processing
- **Characteristics**:
  - Pattern integration
  - Decision making
  - Memory formation

### 3. Neuron Cell Model

Each cell has the following properties:

```python
class NeuronCell:
    activation: float        # Current activation level [0, 1]
    threshold: float         # Firing threshold
    connections: dict        # Weights to neighboring cells
    region_id: int          # Brain region assignment
    state_history: list     # Temporal trace
```

**Activation Dynamics:**
```
activation(t+1) = σ(Σ(weight_i × neighbor_i) + bias)
where σ is the sigmoid function
```

### 4. Reinforcement Learning System

#### Q-Network Architecture
```
Input Layer (100 dims) → Hidden Layer 1 (128 units) → Hidden Layer 2 (64 units) → Output Layer (4 actions)
                ReLU activation              ReLU activation           Linear
```

**State Representation (100 dimensions):**
- Global statistics: mean activation, variance, entropy (10 dims)
- Region-specific statistics: per-region metrics (30 dims)
- Spatial features: local patterns, gradients (30 dims)
- Temporal features: activation changes, momentum (30 dims)

**Action Space (4 discrete actions):**
1. Increase global excitation
2. Decrease global excitation
3. Enhance inter-region connections
4. Modulate learning rate

**Reward Function:**
```python
reward = α * (activation_change) + 
         β * (optimal_range_bonus) + 
         γ * (region_balance) -
         δ * (instability_penalty)

where:
α = 10.0  # Activation change weight
β = 5.0   # Optimal range bonus
γ = 3.0   # Region balance weight
δ = 2.0   # Instability penalty
```

#### Training Process
1. **Experience Collection**
   - State observation
   - Action selection (ε-greedy)
   - Environment step
   - Reward computation
   - Next state observation

2. **Experience Replay**
   - Buffer size: 10,000 transitions
   - Batch size: 32
   - Random sampling for stability

3. **Q-Learning Update**
   - Target: r + γ × max(Q(s', a'))
   - Loss: MSE(Q(s,a), target)
   - Optimizer: Adam (lr=0.001)

### 5. Visualization System

#### Pygame Display
```
┌────────────────────────────────────┐
│  Neural Activity Visualization     │
│  ┌──────────────────────────────┐ │
│  │ ▓▓░░▒▒▓▓░░▒▒  [Color coded] │ │
│  │ ░░▓▓▒▒░░▓▓▒▒   by region    │ │
│  │ ▒▒░░▓▓▒▒░░▓▓                 │ │
│  └──────────────────────────────┘ │
│  Stats: FPS, Activation, etc.     │
└────────────────────────────────────┘
```

**Color Scheme:**
- Vision region: Blue tones
- Motor region: Red tones
- Cognitive region: Green tones
- Brightness: Activation level

#### Control Panel (Tkinter)
```
┌─────────────────────────────┐
│  Simulation Control Panel   │
├─────────────────────────────┤
│  [Pause] [Resume] [Reset]   │
│  Learning Rate: [====|---]  │
│  Epsilon: [======|-----]    │
│  Threshold: [====|------]   │
│                             │
│  Current Stats:             │
│  Avg Activation: 0.45       │
│  Total Reward: 1250         │
│  Episode: 42                │
└─────────────────────────────┘
```

## Data Flow

```
┌─────────────┐
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐
│   Vision    │───▶│ Q-Network    │
│   Region    │    │ (State       │
└──────┬──────┘    │  Encoder)    │
       │           └──────┬───────┘
       ▼                  │
┌─────────────┐           │
│   Motor     │◀──────────┘
│   Region    │    (Action)
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌──────────────┐
│  Cognitive  │───▶│   Reward     │
│   Region    │    │  Calculator  │
└──────┬──────┘    └──────┬───────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌──────────────┐
│   Output    │    │  Q-Network   │
│  (Actions)  │    │   Update     │
└─────────────┘    └──────────────┘
```

## File Organization

```
neuro-genesis-simulator/
│
├── Core Simulation
│   ├── main.py                    # Entry point
│   ├── neuro_genesis_sim.py       # Main simulation with RL
│   ├── neuro_ca_sim.py           # Cellular automaton version
│   └── creature_sim.py            # Creature variant
│
├── Control & Interface
│   └── control_panel.py           # GUI control panel
│
├── Testing
│   ├── tests/test_regions.py      # Region functionality
│   ├── tests/test_rl.py           # RL components
│   └── tests/test_rl_integration.py # RL integration
│
├── Documentation
│   ├── README.md                  # Main documentation
│   ├── CONTRIBUTING.md            # Contribution guide
│   ├── CHANGELOG.md               # Version history
│   └── docs/
│       ├── QUICKSTART.md          # Quick start guide
│       └── ARCHITECTURE.md        # This file
│
└── Configuration
    ├── requirements.txt           # Dependencies
    ├── setup.py                   # Package setup
    ├── .gitignore                # Git ignore rules
    └── LICENSE                    # MIT License
```

## Key Algorithms

### 1. Cellular Update Rule

```python
def update_cell(cell, neighbors):
    # Sum weighted inputs from neighbors
    total_input = sum(
        neighbor.activation * cell.connections[neighbor]
        for neighbor in neighbors
    )
    
    # Apply activation function
    new_activation = sigmoid(total_input - cell.threshold)
    
    # Hebbian learning: strengthen connections
    for neighbor in neighbors:
        if neighbor.activation > 0.5 and cell.activation > 0.5:
            cell.connections[neighbor] += learning_rate * 
                (neighbor.activation * cell.activation)
    
    return new_activation
```

### 2. State Encoding

```python
def get_state(grid):
    # Global features
    activations = [cell.activation for row in grid for cell in row]
    global_mean = np.mean(activations)
    global_std = np.std(activations)
    global_entropy = entropy(activations)
    
    # Region features
    region_stats = []
    for region_id in range(3):
        region_cells = [cell for row in grid for cell in row 
                       if cell.region_id == region_id]
        region_activations = [c.activation for c in region_cells]
        region_stats.extend([
            np.mean(region_activations),
            np.std(region_activations),
            np.max(region_activations)
        ])
    
    # Spatial features
    gradients = compute_spatial_gradients(grid)
    patterns = detect_local_patterns(grid)
    
    # Temporal features
    activation_changes = compute_temporal_changes(grid)
    
    return np.concatenate([
        [global_mean, global_std, global_entropy],
        region_stats,
        gradients,
        patterns,
        activation_changes
    ])
```

### 3. Action Application

```python
def apply_action(grid, action):
    if action == 0:  # Increase excitation
        for row in grid:
            for cell in row:
                cell.threshold *= 0.95
                
    elif action == 1:  # Decrease excitation
        for row in grid:
            for cell in row:
                cell.threshold *= 1.05
                
    elif action == 2:  # Enhance connections
        for row in grid:
            for cell in row:
                for neighbor in cell.connections:
                    cell.connections[neighbor] *= 1.1
                    
    elif action == 3:  # Modulate learning
        global learning_rate
        learning_rate = min(learning_rate * 1.1, 0.1)
```

## Performance Considerations

### Optimization Strategies

1. **NumPy Vectorization**
   - Batch process cell updates
   - Use array operations instead of loops
   - Pre-allocate arrays

2. **Efficient Neighbor Lookup**
   - Pre-compute neighbor indices
   - Cache frequently accessed data
   - Use spatial hashing for large grids

3. **GPU Acceleration (Future)**
   - PyTorch tensors on CUDA
   - Parallel cell updates
   - Batch RL training

### Scalability

**Current Performance:**
- Grid size: 64×48 (3,072 cells)
- Update rate: ~30-60 FPS
- Memory usage: ~100-200 MB

**Tested Configurations:**
- Small: 32×24 (768 cells) - 120+ FPS
- Medium: 64×48 (3,072 cells) - 30-60 FPS
- Large: 128×96 (12,288 cells) - 10-15 FPS

## Extension Points

### Adding New Brain Regions

```python
# In initialize_grid_with_regions()
REGION_COUNT = 4  # Add fourth region

# Define boundaries
region_boundaries = {
    0: (0, width//4),      # Vision
    1: (width//4, width//2),    # Motor
    2: (width//2, 3*width//4),  # Cognitive
    3: (3*width//4, width)      # New region
}
```

### Custom Reward Functions

```python
def custom_reward(grid, old_activation, new_activation):
    # Your custom reward logic
    reward = 0.0
    
    # Example: Reward sparse activation
    active_cells = sum(1 for row in grid for cell in row 
                      if cell.activation > 0.5)
    sparsity = 1.0 - (active_cells / total_cells)
    reward += sparsity * 10.0
    
    return reward
```

### New RL Algorithms

```python
class PPONetwork(nn.Module):
    """Proximal Policy Optimization network"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        # Define actor-critic architecture
        pass
    
    def forward(self, state):
        # Implement forward pass
        return action_probs, value
```

## Future Enhancements

1. **Advanced Learning**
   - Meta-learning
   - Transfer learning
   - Multi-agent systems

2. **Visualization**
   - 3D neural activity
   - Connection topology graphs
   - Real-time analytics dashboard

3. **Performance**
   - CUDA acceleration
   - Multi-threading
   - Distributed simulation

4. **Features**
   - Dynamic region creation
   - Neuroplasticity simulation
   - Sensory input integration

## References

- Cellular Automaton Theory
- Reinforcement Learning (Sutton & Barto)
- Neural Networks and Deep Learning
- Complex Systems and Emergence

---

**Last Updated:** October 30, 2025
**Version:** 1.0.0
