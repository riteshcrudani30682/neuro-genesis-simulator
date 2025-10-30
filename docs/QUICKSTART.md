# Quick Start Guide 🚀

Get up and running with Neuro-Genesis Simulator in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neuro-genesis-simulator.git
cd neuro-genesis-simulator
```

### 2. Create Virtual Environment (Recommended)
```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Simulator
```bash
python main.py
```

## First Time Use

### Understanding the Interface

When you run `main.py`, you'll see two windows:

1. **Simulation Window** (Pygame): Shows the neural network in action
   - Different colors represent different brain regions
   - Brightness indicates neural activation levels

2. **Control Panel** (Tkinter): Allows you to adjust parameters in real-time
   - Pause/Resume simulation
   - Adjust learning rate
   - Modify exploration rate (epsilon)
   - Change activation thresholds

### Basic Controls

**Keyboard Shortcuts:**
- `SPACE` - Pause/Resume
- `R` - Reset simulation
- `S` - Save current state
- `L` - Load saved state
- `+/-` - Adjust speed
- `ESC` - Exit

### Try These First

1. **Watch the Patterns**: Let it run for a minute to see neural patterns emerge
2. **Adjust Learning**: Use the control panel to change the learning rate
3. **Pause and Observe**: Press SPACE to freeze and examine the state
4. **Save Your State**: Press 'S' when you find interesting patterns

## Common Issues

### Import Errors
```bash
# If you see "ModuleNotFoundError"
pip install --upgrade -r requirements.txt
```

### Display Issues
```bash
# If Pygame window doesn't appear
# Make sure your display is configured properly
export DISPLAY=:0  # On Linux with X11
```

### Performance Issues
- Try reducing the grid size in the code
- Close other applications
- Run without the control panel: `python neuro_genesis_sim.py`

## Running Tests

Verify everything is working:
```bash
python test_regions.py
python test_rl.py
python test_rl_integration.py
```

All tests should pass with "test passed!" message.

## Next Steps

1. **Explore the Code**: Start with `main.py` and follow the imports
2. **Read the Full README**: Check `README.md` for detailed documentation
3. **Experiment**: Modify parameters and see what happens
4. **Contribute**: Found a bug or want to add features? See `CONTRIBUTING.md`

## Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check existing issues or start a discussion
- **Documentation**: See the `docs/` folder for detailed guides

## Resources

- [Full Documentation](./README.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [API Reference](./docs/)
- [Examples](./docs/examples/)

Happy Simulating! 🧠✨
