"""Hermetic HTTP fixtures test the wire protocol; they are NOT live-model evidence."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from brains.strategy import OllamaPlanner
from experiments.ollama_validation import main, run_validation, select_model


@pytest.fixture
def fake_ollama():
    received = []
    mode = {'value': 'valid'}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, status=200):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            if self.path == '/api/version':
                self.send_json({'version': 'test-fixture'})
            elif self.path == '/api/tags':
                self.send_json({'models': [{'name': 'fixture:tiny', 'digest': 'fixture-digest', 'size': 123}]})
            else:
                self.send_json({'models': [{'name': 'fixture:tiny', 'size_vram': 123, 'context_length': 2048}]})

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            if mode['value'] == 'http_error':
                self.send_json({'error': 'fixture error'}, 500)
                return
            content = json.dumps({'goal': 'seek_food', 'confidence': .1 if mode['value'] == 'uncertain' else .9,
                                  'reason': 'Fixture response, not an actual model'})
            if mode['value'] == 'invalid':
                content = '{"command":"not allowed"}'
            self.send_json({'message': {'content': content}, 'prompt_eval_count': 25, 'eval_count': 12})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', received, mode
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_http_adapter_gate_and_memory_roundtrip(fake_ollama, tmp_path):
    endpoint, received, _ = fake_ollama
    report = run_validation(model='fixture:tiny', endpoint=endpoint, output=tmp_path)
    assert report['status'] == 'passed' and all(report['checks'].values())
    assert len(received) == 3 and report['world_steps'] == 60
    assert [e['tick'] for e in report['events']] == [0, 20, 40]
    assert report['token_totals'] == {'prompt_eval_count': 75, 'eval_count': 36}
    assert report['loaded_model'][0]['size_vram'] == 123
    assert all(e['latency_s'] >= 0 for e in report['events'])
    assert json.loads((tmp_path/'report.json').read_text()) == report
    assert all(len(payload['messages']) == 2 and payload['stream'] is False for payload in received)


@pytest.mark.parametrize('mode', ['invalid', 'uncertain', 'http_error'])
def test_fallback_never_counts_as_live_llm_success(fake_ollama, tmp_path, mode):
    endpoint, received, behavior = fake_ollama
    behavior['value'] = mode
    report = run_validation(model='fixture:tiny', endpoint=endpoint, output=tmp_path, requests=2)
    assert report['status'] == 'failed'
    assert len(received) == 2 and report['checks']['world_transitions']
    assert not report['checks']['all_requests_accepted_as_llm']
    assert all(e['source'].endswith('fallback') for e in report['events'])
    assert report['token_totals']['eval_count'] == (0 if mode == 'http_error' else 24)


def test_discovery_failure_reports_blocked_and_cli_nonzero(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError('fixture unavailable')
    monkeypatch.setattr('experiments.ollama_validation.get_json', unavailable)
    assert main(['--output', str(tmp_path)]) == 2
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['status'] == 'blocked' and report['events'] == []


def test_uninstalled_model_does_not_make_inference_calls(fake_ollama, tmp_path):
    endpoint, received, _ = fake_ollama
    report = run_validation(model='missing', endpoint=endpoint, output=tmp_path)
    assert report['status'] == 'blocked' and not received


def test_model_selection_requires_explicit_choice(monkeypatch):
    models = [{'name': 'a'}, {'name': 'b'}]
    with pytest.raises(ValueError):
        select_model(models)
    monkeypatch.setattr('builtins.input', lambda _: '2')
    assert select_model(models, interactive=True) == 'b'
    monkeypatch.setattr('builtins.input', lambda _: '0')
    with pytest.raises(ValueError):
        select_model(models, interactive=True)


def test_failed_request_clears_previous_token_usage():
    count = 0
    def transport(payload):
        nonlocal count
        count += 1
        if count > 1:
            raise TimeoutError('fixture timeout')
        return {'message': {'content': '{"goal":"explore","confidence":1,"reason":"fixture"}'}, 'eval_count': 9}
    planner = OllamaPlanner('fixture', transport=transport)
    planner.plan({})
    assert planner.last_usage == {'eval_count': 9}
    with pytest.raises(TimeoutError):
        planner.plan({})
    assert planner.last_usage == {} and planner.last_latency_s >= 0
