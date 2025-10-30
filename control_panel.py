import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import sys
import os
import json
import numpy as np

class SimulationController:
    def __init__(self):
        self.params = {
            'DT': 0.1,
            'SPAWN_THRESHOLD': 0.85,
            'SPAWN_PROB': 0.35,
            'MUTATION_STD': 0.05,
            'LEARNING_RATE': 0.002,
            'DECAY': 0.01,
            'STIM_STRONG': 0.9,
            'INIT_DENSITY': 0.06,
            'MERGE_THRESHOLD': 0.95,
            'MERGE_PROB': 0.005,
            'SOUND_ENABLED': False,
            'BASE_FREQ': 220,
            'SOUND_COOLDOWN': 100,
            'REGION_WEIGHTS': [1.0, 1.0, 1.0],  # Weights for Vision, Motor, Cognitive regions
            'REPLAY_ENABLED': False,
            'REPLAY_BUFFER_SIZE': 1000,
            'GAMMA': 0.95,           # RL discount factor
            'EPSILON': 0.1,          # RL exploration rate
            'EPSILON_DECAY': 0.995,  # RL exploration decay
            'EPSILON_MIN': 0.01,     # RL minimum exploration
            'LEARNING_RATE_RL': 0.001  # RL learning rate
        }
        self.callbacks = {}
        self.simulation_running = False
        self.profiles_dir = "profiles"
        os.makedirs(self.profiles_dir, exist_ok=True)
        
    def register_callback(self, param_name, callback):
        """Register a callback function to be called when a parameter changes"""
        self.callbacks[param_name] = callback
        
    def update_param(self, param_name, value):
        """Update a parameter and call its callback if registered"""
        self.params[param_name] = value
        if param_name in self.callbacks:
            self.callbacks[param_name](value)
            
    def start_simulation(self):
        """Start the simulation"""
        self.simulation_running = True
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.simulation_running = False
        
    def save_profile(self, profile_name):
        """Save current parameters to a profile file"""
        try:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            with open(profile_path, 'w') as f:
                json.dump(self.params, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
            
    def load_profile(self, profile_name):
        """Load parameters from a profile file"""
        try:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            with open(profile_path, 'r') as f:
                params = json.load(f)
                
            # Update current parameters
            for key, value in params.items():
                if key in self.params:
                    self.params[key] = value
                    if key in self.callbacks:
                        self.callbacks[key](value)
            return True
        except Exception as e:
            print(f"Error loading profile: {e}")
            return False
            
    def list_profiles(self):
        """List available profiles"""
        try:
            profiles = []
            for file in os.listdir(self.profiles_dir):
                if file.endswith('.json'):
                    profiles.append(file[:-5])  # Remove .json extension
            return profiles
        except Exception:
            return []

class ControlPanel:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Neuro-Genesis Control Panel")
        self.root.geometry("500x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.simulation_tab = ttk.Frame(self.notebook)
        self.sound_tab = ttk.Frame(self.notebook)
        self.memory_tab = ttk.Frame(self.notebook)
        self.regions_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.profiles_tab = ttk.Frame(self.notebook)
        self.rl_tab = ttk.Frame(self.notebook)  # Reinforcement Learning tab
        
        self.notebook.add(self.simulation_tab, text="Simulation")
        self.notebook.add(self.sound_tab, text="Sound")
        self.notebook.add(self.memory_tab, text="Memory")
        self.notebook.add(self.regions_tab, text="Regions")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.notebook.add(self.profiles_tab, text="Profiles")
        self.notebook.add(self.rl_tab, text="RL")  # Add RL tab
        
        self.create_simulation_controls()
        self.create_sound_controls()
        self.create_memory_controls()
        self.create_regions_controls()
        self.create_analytics_controls()
        self.create_profiles_controls()
        self.create_rl_controls()  # Create RL controls
        
        # Add simulation control buttons
        sim_control_frame = ttk.Frame(self.root)
        sim_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_button = ttk.Button(sim_control_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(sim_control_frame, text="Stop Simulation", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Add quit button
        quit_button = ttk.Button(sim_control_frame, text="Quit", command=self.quit_application)
        quit_button.pack(side=tk.RIGHT, padx=5)
        
    def create_simulation_controls(self):
        """Create controls for simulation parameters"""
        frame = self.simulation_tab
        
        # DT (time step)
        dt_frame = ttk.Frame(frame)
        dt_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(dt_frame, text="Time Step (DT):").pack(side=tk.LEFT)
        self.dt_var = tk.DoubleVar(value=self.controller.params['DT'])
        dt_scale = ttk.Scale(dt_frame, from_=0.01, to=0.5, variable=self.dt_var, orient=tk.HORIZONTAL)
        dt_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.dt_label = ttk.Label(dt_frame, text=f"{self.controller.params['DT']:.3f}")
        self.dt_label.pack(side=tk.LEFT)
        dt_scale.configure(command=self.on_dt_change)
        
        # SPAWN_THRESHOLD
        spawn_thresh_frame = ttk.Frame(frame)
        spawn_thresh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(spawn_thresh_frame, text="Spawn Threshold:").pack(side=tk.LEFT)
        self.spawn_thresh_var = tk.DoubleVar(value=self.controller.params['SPAWN_THRESHOLD'])
        spawn_thresh_scale = ttk.Scale(spawn_thresh_frame, from_=0.1, to=1.0, variable=self.spawn_thresh_var, orient=tk.HORIZONTAL)
        spawn_thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.spawn_thresh_label = ttk.Label(spawn_thresh_frame, text=f"{self.controller.params['SPAWN_THRESHOLD']:.2f}")
        self.spawn_thresh_label.pack(side=tk.LEFT)
        spawn_thresh_scale.configure(command=self.on_spawn_thresh_change)
        
        # SPAWN_PROB
        spawn_prob_frame = ttk.Frame(frame)
        spawn_prob_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(spawn_prob_frame, text="Spawn Probability:").pack(side=tk.LEFT)
        self.spawn_prob_var = tk.DoubleVar(value=self.controller.params['SPAWN_PROB'])
        spawn_prob_scale = ttk.Scale(spawn_prob_frame, from_=0.01, to=1.0, variable=self.spawn_prob_var, orient=tk.HORIZONTAL)
        spawn_prob_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.spawn_prob_label = ttk.Label(spawn_prob_frame, text=f"{self.controller.params['SPAWN_PROB']:.2f}")
        self.spawn_prob_label.pack(side=tk.LEFT)
        spawn_prob_scale.configure(command=self.on_spawn_prob_change)
        
        # MUTATION_STD
        mutation_frame = ttk.Frame(frame)
        mutation_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mutation_frame, text="Mutation Std Dev:").pack(side=tk.LEFT)
        self.mutation_var = tk.DoubleVar(value=self.controller.params['MUTATION_STD'])
        mutation_scale = ttk.Scale(mutation_frame, from_=0.001, to=0.2, variable=self.mutation_var, orient=tk.HORIZONTAL)
        mutation_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.mutation_label = ttk.Label(mutation_frame, text=f"{self.controller.params['MUTATION_STD']:.3f}")
        self.mutation_label.pack(side=tk.LEFT)
        mutation_scale.configure(command=self.on_mutation_change)
        
        # LEARNING_RATE
        lr_frame = ttk.Frame(frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=self.controller.params['LEARNING_RATE'])
        lr_scale = ttk.Scale(lr_frame, from_=0.0001, to=0.01, variable=self.lr_var, orient=tk.HORIZONTAL)
        lr_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.lr_label = ttk.Label(lr_frame, text=f"{self.controller.params['LEARNING_RATE']:.4f}")
        self.lr_label.pack(side=tk.LEFT)
        lr_scale.configure(command=self.on_lr_change)
        
        # DECAY
        decay_frame = ttk.Frame(frame)
        decay_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(decay_frame, text="Activation Decay:").pack(side=tk.LEFT)
        self.decay_var = tk.DoubleVar(value=self.controller.params['DECAY'])
        decay_scale = ttk.Scale(decay_frame, from_=0.001, to=0.1, variable=self.decay_var, orient=tk.HORIZONTAL)
        decay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.decay_label = ttk.Label(decay_frame, text=f"{self.controller.params['DECAY']:.3f}")
        self.decay_label.pack(side=tk.LEFT)
        decay_scale.configure(command=self.on_decay_change)
        
        # STIM_STRONG
        stim_frame = ttk.Frame(frame)
        stim_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(stim_frame, text="Stimulus Strength:").pack(side=tk.LEFT)
        self.stim_var = tk.DoubleVar(value=self.controller.params['STIM_STRONG'])
        stim_scale = ttk.Scale(stim_frame, from_=0.1, to=1.0, variable=self.stim_var, orient=tk.HORIZONTAL)
        stim_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.stim_label = ttk.Label(stim_frame, text=f"{self.controller.params['STIM_STRONG']:.2f}")
        self.stim_label.pack(side=tk.LEFT)
        stim_scale.configure(command=self.on_stim_change)
        
        # INIT_DENSITY
        density_frame = ttk.Frame(frame)
        density_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(density_frame, text="Initial Density:").pack(side=tk.LEFT)
        self.density_var = tk.DoubleVar(value=self.controller.params['INIT_DENSITY'])
        density_scale = ttk.Scale(density_frame, from_=0.01, to=0.5, variable=self.density_var, orient=tk.HORIZONTAL)
        density_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.density_label = ttk.Label(density_frame, text=f"{self.controller.params['INIT_DENSITY']:.2f}")
        self.density_label.pack(side=tk.LEFT)
        density_scale.configure(command=self.on_density_change)
        
        # REPLAY_ENABLED
        replay_frame = ttk.Frame(frame)
        replay_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(replay_frame, text="Enable Replay Learning:").pack(side=tk.LEFT)
        self.replay_var = tk.BooleanVar(value=self.controller.params['REPLAY_ENABLED'])
        replay_check = ttk.Checkbutton(replay_frame, variable=self.replay_var)
        replay_check.pack(side=tk.LEFT, padx=5)
        replay_check.configure(command=self.on_replay_change)
        
        # REPLAY_BUFFER_SIZE
        replay_buffer_frame = ttk.Frame(frame)
        replay_buffer_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(replay_buffer_frame, text="Replay Buffer Size:").pack(side=tk.LEFT)
        self.replay_buffer_var = tk.IntVar(value=self.controller.params['REPLAY_BUFFER_SIZE'])
        replay_buffer_scale = ttk.Scale(replay_buffer_frame, from_=100, to=5000, variable=self.replay_buffer_var, orient=tk.HORIZONTAL)
        replay_buffer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.replay_buffer_label = ttk.Label(replay_buffer_frame, text=f"{self.controller.params['REPLAY_BUFFER_SIZE']}")
        self.replay_buffer_label.pack(side=tk.LEFT)
        replay_buffer_scale.configure(command=self.on_replay_buffer_change)
        
    def create_sound_controls(self):
        """Create controls for sound parameters"""
        frame = self.sound_tab
        
        # SOUND_ENABLED
        sound_enabled_frame = ttk.Frame(frame)
        sound_enabled_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sound_enabled_frame, text="Enable Sound:").pack(side=tk.LEFT)
        self.sound_enabled_var = tk.BooleanVar(value=self.controller.params['SOUND_ENABLED'])
        sound_enabled_check = ttk.Checkbutton(sound_enabled_frame, variable=self.sound_enabled_var)
        sound_enabled_check.pack(side=tk.LEFT, padx=5)
        sound_enabled_check.configure(command=self.on_sound_enabled_change)
        
        # BASE_FREQ
        base_freq_frame = ttk.Frame(frame)
        base_freq_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(base_freq_frame, text="Base Frequency:").pack(side=tk.LEFT)
        self.base_freq_var = tk.IntVar(value=self.controller.params['BASE_FREQ'])
        base_freq_scale = ttk.Scale(base_freq_frame, from_=100, to=1000, variable=self.base_freq_var, orient=tk.HORIZONTAL)
        base_freq_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.base_freq_label = ttk.Label(base_freq_frame, text=f"{self.controller.params['BASE_FREQ']} Hz")
        self.base_freq_label.pack(side=tk.LEFT)
        base_freq_scale.configure(command=self.on_base_freq_change)
        
        # SOUND_COOLDOWN
        cooldown_frame = ttk.Frame(frame)
        cooldown_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(cooldown_frame, text="Sound Cooldown:").pack(side=tk.LEFT)
        self.cooldown_var = tk.IntVar(value=self.controller.params['SOUND_COOLDOWN'])
        cooldown_scale = ttk.Scale(cooldown_frame, from_=10, to=1000, variable=self.cooldown_var, orient=tk.HORIZONTAL)
        cooldown_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cooldown_label = ttk.Label(cooldown_frame, text=f"{self.controller.params['SOUND_COOLDOWN']} ms")
        self.cooldown_label.pack(side=tk.LEFT)
        cooldown_scale.configure(command=self.on_cooldown_change)
        
    def create_memory_controls(self):
        """Create controls for memory parameters"""
        frame = self.memory_tab
        
        # MERGE_THRESHOLD
        merge_thresh_frame = ttk.Frame(frame)
        merge_thresh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(merge_thresh_frame, text="Merge Threshold:").pack(side=tk.LEFT)
        self.merge_thresh_var = tk.DoubleVar(value=self.controller.params['MERGE_THRESHOLD'])
        merge_thresh_scale = ttk.Scale(merge_thresh_frame, from_=0.8, to=0.99, variable=self.merge_thresh_var, orient=tk.HORIZONTAL)
        merge_thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.merge_thresh_label = ttk.Label(merge_thresh_frame, text=f"{self.controller.params['MERGE_THRESHOLD']:.2f}")
        self.merge_thresh_label.pack(side=tk.LEFT)
        merge_thresh_scale.configure(command=self.on_merge_thresh_change)
        
        # MERGE_PROB
        merge_prob_frame = ttk.Frame(frame)
        merge_prob_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(merge_prob_frame, text="Merge Probability:").pack(side=tk.LEFT)
        self.merge_prob_var = tk.DoubleVar(value=self.controller.params['MERGE_PROB'])
        merge_prob_scale = ttk.Scale(merge_prob_frame, from_=0.001, to=0.1, variable=self.merge_prob_var, orient=tk.HORIZONTAL)
        merge_prob_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.merge_prob_label = ttk.Label(merge_prob_frame, text=f"{self.controller.params['MERGE_PROB']:.3f}")
        self.merge_prob_label.pack(side=tk.LEFT)
        merge_prob_scale.configure(command=self.on_merge_prob_change)
        
    def create_regions_controls(self):
        """Create controls for multi-region brain parameters"""
        frame = self.regions_tab
        
        ttk.Label(frame, text="Region Weights (Vision, Motor, Cognitive):").pack(pady=5)
        
        # Vision Region Weight
        vision_frame = ttk.Frame(frame)
        vision_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(vision_frame, text="Vision Weight:").pack(side=tk.LEFT)
        self.vision_weight_var = tk.DoubleVar(value=self.controller.params['REGION_WEIGHTS'][0])
        vision_scale = ttk.Scale(vision_frame, from_=0.1, to=3.0, variable=self.vision_weight_var, orient=tk.HORIZONTAL)
        vision_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.vision_weight_label = ttk.Label(vision_frame, text=f"{self.controller.params['REGION_WEIGHTS'][0]:.2f}")
        self.vision_weight_label.pack(side=tk.LEFT)
        vision_scale.configure(command=self.on_vision_weight_change)
        
        # Motor Region Weight
        motor_frame = ttk.Frame(frame)
        motor_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(motor_frame, text="Motor Weight:").pack(side=tk.LEFT)
        self.motor_weight_var = tk.DoubleVar(value=self.controller.params['REGION_WEIGHTS'][1])
        motor_scale = ttk.Scale(motor_frame, from_=0.1, to=3.0, variable=self.motor_weight_var, orient=tk.HORIZONTAL)
        motor_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.motor_weight_label = ttk.Label(motor_frame, text=f"{self.controller.params['REGION_WEIGHTS'][1]:.2f}")
        self.motor_weight_label.pack(side=tk.LEFT)
        motor_scale.configure(command=self.on_motor_weight_change)
        
        # Cognitive Region Weight
        cognitive_frame = ttk.Frame(frame)
        cognitive_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(cognitive_frame, text="Cognitive Weight:").pack(side=tk.LEFT)
        self.cognitive_weight_var = tk.DoubleVar(value=self.controller.params['REGION_WEIGHTS'][2])
        cognitive_scale = ttk.Scale(cognitive_frame, from_=0.1, to=3.0, variable=self.cognitive_weight_var, orient=tk.HORIZONTAL)
        cognitive_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cognitive_weight_label = ttk.Label(cognitive_frame, text=f"{self.controller.params['REGION_WEIGHTS'][2]:.2f}")
        self.cognitive_weight_label.pack(side=tk.LEFT)
        cognitive_scale.configure(command=self.on_cognitive_weight_change)
        
    def create_analytics_controls(self):
        """Create controls for data analytics"""
        frame = self.analytics_tab
        
        ttk.Label(frame, text="Data Analytics Mode").pack(pady=10)
        ttk.Label(frame, text="Analytics are displayed in the simulation window").pack(pady=5)
        
        # Add buttons for different analytics views
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.activation_btn = ttk.Button(buttons_frame, text="Show Activation Plot", 
                                        command=lambda: self.show_analytics("activation"))
        self.activation_btn.pack(side=tk.LEFT, padx=5)
        
        self.mutation_btn = ttk.Button(buttons_frame, text="Show Mutation Plot", 
                                      command=lambda: self.show_analytics("mutation"))
        self.mutation_btn.pack(side=tk.LEFT, padx=5)
        
        self.region_btn = ttk.Button(buttons_frame, text="Show Region Plot", 
                                    command=lambda: self.show_analytics("region"))
        self.region_btn.pack(side=tk.LEFT, padx=5)
        
    def create_profiles_controls(self):
        """Create controls for saveable profiles"""
        frame = self.profiles_tab
        
        # Profile name entry
        name_frame = ttk.Frame(frame)
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(name_frame, text="Profile Name:").pack(side=tk.LEFT)
        self.profile_name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.profile_name_var)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Save/Load buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        save_btn = ttk.Button(buttons_frame, text="Save Profile", command=self.save_profile)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(buttons_frame, text="Load Profile", command=self.load_profile)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Profile list
        ttk.Label(frame, text="Saved Profiles:").pack(pady=5)
        self.profile_listbox = tk.Listbox(frame, height=10)
        self.profile_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(frame, text="Refresh List", command=self.refresh_profiles)
        refresh_btn.pack(pady=5)
        
        # Load profiles list
        self.refresh_profiles()
        
    def create_rl_controls(self):
        """Create controls for reinforcement learning parameters"""
        frame = self.rl_tab
        
        ttk.Label(frame, text="Reinforcement Learning Parameters").pack(pady=10)
        
        # GAMMA (discount factor)
        gamma_frame = ttk.Frame(frame)
        gamma_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(gamma_frame, text="Discount Factor (γ):").pack(side=tk.LEFT)
        self.gamma_var = tk.DoubleVar(value=self.controller.params['GAMMA'])
        gamma_scale = ttk.Scale(gamma_frame, from_=0.1, to=0.99, variable=self.gamma_var, orient=tk.HORIZONTAL)
        gamma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.gamma_label = ttk.Label(gamma_frame, text=f"{self.controller.params['GAMMA']:.2f}")
        self.gamma_label.pack(side=tk.LEFT)
        gamma_scale.configure(command=self.on_gamma_change)
        
        # EPSILON (exploration rate)
        epsilon_frame = ttk.Frame(frame)
        epsilon_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(epsilon_frame, text="Exploration Rate (ε):").pack(side=tk.LEFT)
        self.epsilon_var = tk.DoubleVar(value=self.controller.params['EPSILON'])
        epsilon_scale = ttk.Scale(epsilon_frame, from_=0.01, to=1.0, variable=self.epsilon_var, orient=tk.HORIZONTAL)
        epsilon_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.epsilon_label = ttk.Label(epsilon_frame, text=f"{self.controller.params['EPSILON']:.2f}")
        self.epsilon_label.pack(side=tk.LEFT)
        epsilon_scale.configure(command=self.on_epsilon_change)
        
        # EPSILON_DECAY
        epsilon_decay_frame = ttk.Frame(frame)
        epsilon_decay_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(epsilon_decay_frame, text="Exploration Decay:").pack(side=tk.LEFT)
        self.epsilon_decay_var = tk.DoubleVar(value=self.controller.params['EPSILON_DECAY'])
        epsilon_decay_scale = ttk.Scale(epsilon_decay_frame, from_=0.9, to=0.999, variable=self.epsilon_decay_var, orient=tk.HORIZONTAL)
        epsilon_decay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.epsilon_decay_label = ttk.Label(epsilon_decay_frame, text=f"{self.controller.params['EPSILON_DECAY']:.3f}")
        self.epsilon_decay_label.pack(side=tk.LEFT)
        epsilon_decay_scale.configure(command=self.on_epsilon_decay_change)
        
        # EPSILON_MIN
        epsilon_min_frame = ttk.Frame(frame)
        epsilon_min_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(epsilon_min_frame, text="Min Exploration (ε):").pack(side=tk.LEFT)
        self.epsilon_min_var = tk.DoubleVar(value=self.controller.params['EPSILON_MIN'])
        epsilon_min_scale = ttk.Scale(epsilon_min_frame, from_=0.001, to=0.1, variable=self.epsilon_min_var, orient=tk.HORIZONTAL)
        epsilon_min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.epsilon_min_label = ttk.Label(epsilon_min_frame, text=f"{self.controller.params['EPSILON_MIN']:.3f}")
        self.epsilon_min_label.pack(side=tk.LEFT)
        epsilon_min_scale.configure(command=self.on_epsilon_min_change)
        
        # LEARNING_RATE_RL
        lr_rl_frame = ttk.Frame(frame)
        lr_rl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_rl_frame, text="RL Learning Rate:").pack(side=tk.LEFT)
        self.lr_rl_var = tk.DoubleVar(value=self.controller.params['LEARNING_RATE_RL'])
        lr_rl_scale = ttk.Scale(lr_rl_frame, from_=0.0001, to=0.01, variable=self.lr_rl_var, orient=tk.HORIZONTAL)
        lr_rl_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.lr_rl_label = ttk.Label(lr_rl_frame, text=f"{self.controller.params['LEARNING_RATE_RL']:.4f}")
        self.lr_rl_label.pack(side=tk.LEFT)
        lr_rl_scale.configure(command=self.on_lr_rl_change)
        
    def refresh_profiles(self):
        """Refresh the profiles list"""
        self.profile_listbox.delete(0, tk.END)
        profiles = self.controller.list_profiles()
        for profile in profiles:
            self.profile_listbox.insert(tk.END, profile)
            
    def save_profile(self):
        """Save current parameters to a profile"""
        profile_name = self.profile_name_var.get().strip()
        if not profile_name:
            messagebox.showwarning("Warning", "Please enter a profile name")
            return
            
        if self.controller.save_profile(profile_name):
            messagebox.showinfo("Success", f"Profile '{profile_name}' saved successfully")
            self.refresh_profiles()
        else:
            messagebox.showerror("Error", "Failed to save profile")
            
    def load_profile(self):
        """Load selected profile"""
        selection = self.profile_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a profile to load")
            return
            
        profile_name = self.profile_listbox.get(selection[0])
        if self.controller.load_profile(profile_name):
            messagebox.showinfo("Success", f"Profile '{profile_name}' loaded successfully")
            # Update UI to reflect loaded values
            self.update_ui_from_params()
        else:
            messagebox.showerror("Error", "Failed to load profile")
            
    def update_ui_from_params(self):
        """Update UI controls to match current parameter values"""
        # Update all slider labels and values
        self.dt_var.set(self.controller.params['DT'])
        self.dt_label.config(text=f"{self.controller.params['DT']:.3f}")
        
        self.spawn_thresh_var.set(self.controller.params['SPAWN_THRESHOLD'])
        self.spawn_thresh_label.config(text=f"{self.controller.params['SPAWN_THRESHOLD']:.2f}")
        
        self.spawn_prob_var.set(self.controller.params['SPAWN_PROB'])
        self.spawn_prob_label.config(text=f"{self.controller.params['SPAWN_PROB']:.2f}")
        
        self.mutation_var.set(self.controller.params['MUTATION_STD'])
        self.mutation_label.config(text=f"{self.controller.params['MUTATION_STD']:.3f}")
        
        self.lr_var.set(self.controller.params['LEARNING_RATE'])
        self.lr_label.config(text=f"{self.controller.params['LEARNING_RATE']:.4f}")
        
        self.decay_var.set(self.controller.params['DECAY'])
        self.decay_label.config(text=f"{self.controller.params['DECAY']:.3f}")
        
        self.stim_var.set(self.controller.params['STIM_STRONG'])
        self.stim_label.config(text=f"{self.controller.params['STIM_STRONG']:.2f}")
        
        self.density_var.set(self.controller.params['INIT_DENSITY'])
        self.density_label.config(text=f"{self.controller.params['INIT_DENSITY']:.2f}")
        
        self.sound_enabled_var.set(self.controller.params['SOUND_ENABLED'])
        
        self.base_freq_var.set(self.controller.params['BASE_FREQ'])
        self.base_freq_label.config(text=f"{self.controller.params['BASE_FREQ']} Hz")
        
        self.cooldown_var.set(self.controller.params['SOUND_COOLDOWN'])
        self.cooldown_label.config(text=f"{self.controller.params['SOUND_COOLDOWN']} ms")
        
        self.merge_thresh_var.set(self.controller.params['MERGE_THRESHOLD'])
        self.merge_thresh_label.config(text=f"{self.controller.params['MERGE_THRESHOLD']:.2f}")
        
        self.merge_prob_var.set(self.controller.params['MERGE_PROB'])
        self.merge_prob_label.config(text=f"{self.controller.params['MERGE_PROB']:.3f}")
        
        self.replay_var.set(self.controller.params['REPLAY_ENABLED'])
        
        self.replay_buffer_var.set(self.controller.params['REPLAY_BUFFER_SIZE'])
        self.replay_buffer_label.config(text=f"{self.controller.params['REPLAY_BUFFER_SIZE']}")
        
        # Update region weights
        self.vision_weight_var.set(self.controller.params['REGION_WEIGHTS'][0])
        self.vision_weight_label.config(text=f"{self.controller.params['REGION_WEIGHTS'][0]:.2f}")
        
        self.motor_weight_var.set(self.controller.params['REGION_WEIGHTS'][1])
        self.motor_weight_label.config(text=f"{self.controller.params['REGION_WEIGHTS'][1]:.2f}")
        
        self.cognitive_weight_var.set(self.controller.params['REGION_WEIGHTS'][2])
        self.cognitive_weight_label.config(text=f"{self.controller.params['REGION_WEIGHTS'][2]:.2f}")
        
    def show_analytics(self, chart_type):
        """Show analytics chart (would be implemented in the simulation)"""
        print(f"Showing {chart_type} analytics chart")
        
    # Callback methods for parameter changes
    def on_dt_change(self, value):
        val = float(value)
        self.dt_label.config(text=f"{val:.3f}")
        self.controller.update_param('DT', val)
        
    def on_spawn_thresh_change(self, value):
        val = float(value)
        self.spawn_thresh_label.config(text=f"{val:.2f}")
        self.controller.update_param('SPAWN_THRESHOLD', val)
        
    def on_spawn_prob_change(self, value):
        val = float(value)
        self.spawn_prob_label.config(text=f"{val:.2f}")
        self.controller.update_param('SPAWN_PROB', val)
        
    def on_mutation_change(self, value):
        val = float(value)
        self.mutation_label.config(text=f"{val:.3f}")
        self.controller.update_param('MUTATION_STD', val)
        
    def on_lr_change(self, value):
        val = float(value)
        self.lr_label.config(text=f"{val:.4f}")
        self.controller.update_param('LEARNING_RATE', val)
        
    def on_decay_change(self, value):
        val = float(value)
        self.decay_label.config(text=f"{val:.3f}")
        self.controller.update_param('DECAY', val)
        
    def on_stim_change(self, value):
        val = float(value)
        self.stim_label.config(text=f"{val:.2f}")
        self.controller.update_param('STIM_STRONG', val)
        
    def on_density_change(self, value):
        val = float(value)
        self.density_label.config(text=f"{val:.2f}")
        self.controller.update_param('INIT_DENSITY', val)
        
    def on_sound_enabled_change(self):
        val = self.sound_enabled_var.get()
        self.controller.update_param('SOUND_ENABLED', val)
        
    def on_base_freq_change(self, value):
        val = int(float(value))
        self.base_freq_label.config(text=f"{val} Hz")
        self.controller.update_param('BASE_FREQ', val)
        
    def on_cooldown_change(self, value):
        val = int(float(value))
        self.cooldown_label.config(text=f"{val} ms")
        self.controller.update_param('SOUND_COOLDOWN', val)
        
    def on_merge_thresh_change(self, value):
        val = float(value)
        self.merge_thresh_label.config(text=f"{val:.2f}")
        self.controller.update_param('MERGE_THRESHOLD', val)
        
    def on_merge_prob_change(self, value):
        val = float(value)
        self.merge_prob_label.config(text=f"{val:.3f}")
        self.controller.update_param('MERGE_PROB', val)
        
    def on_replay_change(self):
        val = self.replay_var.get()
        self.controller.update_param('REPLAY_ENABLED', val)
        
    def on_replay_buffer_change(self, value):
        val = int(float(value))
        self.replay_buffer_label.config(text=f"{val}")
        self.controller.update_param('REPLAY_BUFFER_SIZE', val)
        
    def on_vision_weight_change(self, value):
        val = float(value)
        self.vision_weight_label.config(text=f"{val:.2f}")
        weights = self.controller.params['REGION_WEIGHTS'].copy()
        weights[0] = val
        self.controller.update_param('REGION_WEIGHTS', weights)
        
    def on_motor_weight_change(self, value):
        val = float(value)
        self.motor_weight_label.config(text=f"{val:.2f}")
        weights = self.controller.params['REGION_WEIGHTS'].copy()
        weights[1] = val
        self.controller.update_param('REGION_WEIGHTS', weights)
        
    def on_cognitive_weight_change(self, value):
        val = float(value)
        self.cognitive_weight_label.config(text=f"{val:.2f}")
        weights = self.controller.params['REGION_WEIGHTS'].copy()
        weights[2] = val
        self.controller.update_param('REGION_WEIGHTS', weights)
        
    # Callback methods for RL parameter changes
    def on_gamma_change(self, value):
        val = float(value)
        self.gamma_label.config(text=f"{val:.2f}")
        self.controller.update_param('GAMMA', val)
        
    def on_epsilon_change(self, value):
        val = float(value)
        self.epsilon_label.config(text=f"{val:.2f}")
        self.controller.update_param('EPSILON', val)
        
    def on_epsilon_decay_change(self, value):
        val = float(value)
        self.epsilon_decay_label.config(text=f"{val:.3f}")
        self.controller.update_param('EPSILON_DECAY', val)
        
    def on_epsilon_min_change(self, value):
        val = float(value)
        self.epsilon_min_label.config(text=f"{val:.3f}")
        self.controller.update_param('EPSILON_MIN', val)
        
    def on_lr_rl_change(self, value):
        val = float(value)
        self.lr_rl_label.config(text=f"{val:.4f}")
        self.controller.update_param('LEARNING_RATE_RL', val)
        
    def start_simulation(self):
        """Start the simulation"""
        self.controller.start_simulation()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.controller.stop_simulation()
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
    def quit_application(self):
        """Quit the application"""
        self.controller.stop_simulation()
        self.root.quit()
        self.root.destroy()
        os._exit(0)  # Force exit to terminate all threads
        
    def run(self):
        """Run the control panel"""
        self.root.mainloop()