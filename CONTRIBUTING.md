# Contributing to Neuro-Genesis Simulator

First off, thank you for considering contributing to Neuro-Genesis Simulator! 🎉

## How Can I Contribute?

### Reporting Bugs 🐛

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** to demonstrate the steps
- **Describe the behavior you observed** and what you expected to see
- **Include screenshots** if relevant
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements 💡

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternative solutions** you've considered

### Pull Requests 🔧

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. Ensure the test suite passes
4. Make sure your code follows the existing style
5. Write a clear commit message
6. Open a Pull Request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/neuro-genesis-simulator.git
cd neuro-genesis-simulator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_regions.py
python test_rl.py
python test_rl_integration.py
```

## Coding Standards

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Write comments for complex logic

## Testing

- Add tests for new features
- Ensure all existing tests pass
- Test on multiple Python versions if possible

## Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update comments for modified code

## Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Example:
```
Add reinforcement learning visualization

- Implement Q-value heatmap display
- Add action probability distribution chart
- Update control panel with RL metrics

Closes #123
```

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Questions?

Feel free to open an issue for any questions!

Thank you for contributing! 🙏
