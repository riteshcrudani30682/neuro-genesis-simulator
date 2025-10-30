# Neuro-Genesis Cellular Simulator 🧠

A sophisticated neural network cellular automaton simulator with reinforcement learning capabilities, multi-region brain architecture, and real-time visualization.

## 🌟 Features

- **Multi-Region Brain Architecture**: Simulates different brain regions (Vision, Motor, Cognitive)
- **Cellular Automaton**: Each neuron is a cell with local interactions
- **Reinforcement Learning**: Q-learning based optimization for neural patterns
- **Real-time Visualization**: Pygame-based interactive display
- **Control Panel**: GUI for parameter adjustment and simulation control
- **State Persistence**: Save and load simulation states

## 📋 Requirements

- Python 3.8+
- NumPy
- Pygame
- PyTorch (for RL components)
- tkinter (usually comes with Python)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/neuro-genesis-simulator.git
cd neuro-genesis-simulator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage
Run the main simulation with integrated control panel:
```bash
python main.py
```

### Testing Components
Test multi-region functionality:
```bash
python test_regions.py
```

Test reinforcement learning:
```bash
python test_rl.py
```

Test RL integration:
```bash
python test_rl_integration.py
```

### Running Individual Modules

**Neuro-Genesis Simulation (Main):**
```bash
python neuro_genesis_sim.py
```

**Cellular Automaton Version:**
```bash
python neuro_ca_sim.py
```

**Creature Simulation:**
```bash
python creature_sim.py
```

**Control Panel (Standalone):**
```bash
python control_panel.py
```

## 🎮 Controls

### Keyboard Controls
- **SPACE**: Pause/Resume simulation
- **R**: Reset simulation
- **S**: Save current state
- **L**: Load saved state
- **+/-**: Adjust simulation speed
- **ESC**: Exit simulation

### Control Panel
The GUI control panel allows you to adjust:
- Neural activation thresholds
- Learning rates
- Exploration vs exploitation (epsilon)
- Region-specific parameters
- Visualization settings

## 🧬 Architecture

### Core Components

1. **Neuron Cells**: Individual cellular automaton units with:
   - Activation state
   - Connection weights
   - Region assignment
   - Learning capabilities

2. **Brain Regions**:
   - **Vision Region**: Input processing
   - **Motor Region**: Output generation
   - **Cognitive Region**: Higher-level processing

3. **Reinforcement Learning**:
   - Q-Network for policy learning
   - Experience replay
   - Reward computation based on activation patterns

4. **Visualization**:
   - Real-time neural activity display
   - Region-based color coding
   - Connection strength visualization

## 📊 Technical Details

### State Representation
- Grid size: 64x48 cells
- Input dimensions: Configurable (default: 100)
- Action space: 4 discrete actions
- Reward: Based on optimal activation ranges

### Learning Parameters
- Learning rate: Configurable via control panel
- Discount factor (gamma): 0.99
- Epsilon (exploration): Configurable
- Replay buffer size: 10000

## 🔧 Configuration

Key parameters can be adjusted in the code or via the control panel:

```python
# In neuro_genesis_sim.py
INP_DIM = 100          # Input dimension
REGION_COUNT = 3       # Number of brain regions
GRID_WIDTH = 64        # Grid width
GRID_HEIGHT = 48       # Grid height
```

## 📁 Project Structure

```
neuro-genesis-simulator/
├── main.py                      # Main entry point
├── neuro_genesis_sim.py         # Core simulation with RL
├── neuro_ca_sim.py             # Cellular automaton version
├── creature_sim.py             # Creature-based simulation
├── creature_rl_ns.py           # Creature with RL
├── control_panel.py            # GUI control panel
├── test_regions.py             # Region testing
├── test_rl.py                  # RL component tests
├── test_rl_integration.py      # RL integration tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                 # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Known Issues

- Large grids may cause performance issues on slower systems
- Control panel may need adjustment for high DPI displays

## 🔮 Future Enhancements

- [ ] Multi-threaded simulation for better performance
- [ ] Advanced RL algorithms (PPO, A3C)
- [ ] Network topology visualization
- [ ] Batch simulation and analysis tools
- [ ] REST API for remote control
- [ ] Web-based visualization

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Inspired by biological neural networks
- Built with PyTorch and Pygame
- Cellular automaton concepts from Conway's Game of Life

---

**Made with ❤️ for neural network enthusiasts**
