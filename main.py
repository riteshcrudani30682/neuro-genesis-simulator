#!/usr/bin/env python3
"""Launch the simulator or run a reproducible headless smoke test."""
import argparse


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-panel", action="store_true", help="Run Pygame only")
    parser.add_argument("--headless", action="store_true", help="Run smoke test without windows")
    parser.add_argument("--steps", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.headless:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    import neuro_genesis_sim as sim
    if args.headless:
        sim.run_smoke_test(steps=args.steps)
    else:
        sim.main(with_control_panel=not args.no_panel)


def run_with_control_panel():
    """Console-script entry point."""
    main()


if __name__ == "__main__":
    main()
