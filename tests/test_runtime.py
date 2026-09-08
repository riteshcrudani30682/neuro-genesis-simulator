"""Regression coverage for controls, live parameters and terminal learning."""
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import neuro_genesis_sim as sim
from control_panel import SimulationController, ControlPanel
from main import positive_int


def test_terminal_transition_updates_weights(monkeypatch):
    network = sim.QNetwork(sim.INP_DIM, action_dim=4).to(sim.device)
    monkeypatch.setattr(sim, 'q_network', network)
    monkeypatch.setattr(sim, 'q_optimizer', torch.optim.Adam(network.parameters(), lr=0.01))
    before = [p.detach().clone() for p in network.parameters()]
    state = np.ones(sim.INP_DIM, dtype=np.float32)
    loss = sim.update_q_network(state, 0, 1.0, state, done=True)
    assert np.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, network.parameters()))


def test_epsilon_uses_live_value(monkeypatch):
    monkeypatch.setattr(sim, 'EPSILON', 1.0)
    monkeypatch.setattr(sim.random, 'randint', lambda a, b: 3)
    assert sim.choose_action(np.zeros(sim.INP_DIM, dtype=np.float32)) == 3


def test_controller_updates_actual_optimizers_and_buffer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim, 'optimizer', SimpleNamespace(param_groups=[{'lr': 0.1}]))
    monkeypatch.setattr(sim, 'q_optimizer', SimpleNamespace(param_groups=[{'lr': 0.1}]))
    monkeypatch.setattr(sim, 'replay_buffer', deque(range(5), maxlen=5))
    control = SimulationController()
    assert sim.start_control_panel(control) is control
    control.update_param('LEARNING_RATE', 0.003)
    control.update_param('LEARNING_RATE_RL', 0.004)
    control.update_param('REPLAY_BUFFER_SIZE', 2)
    assert sim.optimizer.param_groups[0]['lr'] == 0.003
    assert sim.q_optimizer.param_groups[0]['lr'] == 0.004
    assert sim.replay_buffer.maxlen == 2
    assert list(sim.replay_buffer) == [3, 4]


def test_quit_requests_graceful_shutdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = ControlPanel.__new__(ControlPanel)
    panel.controller = SimulationController()
    panel.root = Mock()
    panel.closed = False
    panel.quit_application()
    panel.close()
    assert panel.controller.quit_requested
    assert not panel.controller.simulation_running
    panel.root.destroy.assert_called_once()


@pytest.mark.parametrize('running, expected_steps', [(True, 1), (False, 0)])
def test_main_loop_honors_panel_pause(monkeypatch, tmp_path, running, expected_steps):
    import control_panel
    monkeypatch.chdir(tmp_path)
    control = SimulationController()
    class FakePanel:
        def __init__(self, controller):
            assert controller is control
        def start_simulation(self):
            control.simulation_running = running
        def process_events(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(control_panel, 'ControlPanel', FakePanel)
    monkeypatch.setattr(sim, 'start_control_panel', lambda: control)
    monkeypatch.setattr(sim, 'init_visualization', lambda: (Mock(), Mock(), Mock()))
    events = iter([[], [SimpleNamespace(type=sim.pygame.QUIT)]])
    monkeypatch.setattr(sim.pygame.event, 'get', lambda: next(events))
    monkeypatch.setattr(sim.pygame.display, 'flip', lambda: None)
    monkeypatch.setattr(sim, 'draw', Mock())
    step, save = Mock(), Mock()
    monkeypatch.setattr(sim, 'sim_step', step)
    monkeypatch.setattr(sim, 'save_sim', save)
    sim.main()
    assert step.call_count == expected_steps
    save.assert_called_once()


@pytest.mark.parametrize('value', ['0', '-1'])
def test_reject_nonpositive_steps(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)
