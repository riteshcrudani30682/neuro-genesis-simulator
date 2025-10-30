#!/usr/bin/env python3
"""
Neuro-Genesis Cellular Simulator with GUI Control Panel
Run: python main.py
"""

import threading
import time
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_simulation(controller=None):
    """Run the Neuro-Genesis simulation"""
    try:
        # Import and run the simulation
        from neuro_genesis_sim import main as sim_main, start_control_panel
        
        # If we have a controller, use it
        if controller is not None:
            # Start the simulation with the controller
            print("Starting simulation with control panel...")
            sim_main()
        else:
            # Run standalone simulation
            print("Starting standalone simulation...")
            sim_main()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()

def run_control_panel():
    """Run the control panel and return the controller"""
    try:
        # Import and run the control panel
        from control_panel import SimulationController, ControlPanel
        
        controller = SimulationController()
        panel = ControlPanel(controller)
        
        # Run the control panel in a separate thread
        panel_thread = threading.Thread(target=panel.run)
        panel_thread.daemon = True
        panel_thread.start()
        
        print("Control panel started in background")
        return controller, panel_thread
    except Exception as e:
        print(f"Error running control panel: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_with_control_panel():
    """Run both the control panel and simulation integrated"""
    # Start the control panel
    controller, panel_thread = run_control_panel()
    
    if controller is None:
        print("Failed to start control panel, running standalone simulation")
        run_simulation()
        return
    
    # Start the simulation
    print("Starting simulation with control panel integration...")
    run_simulation(controller)

if __name__ == "__main__":
    # Run with control panel integration
    run_with_control_panel()