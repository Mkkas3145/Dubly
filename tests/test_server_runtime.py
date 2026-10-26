from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import server_runtime


@pytest.mark.parametrize('value', ['0', '65536', '', 'abc', None])
def test_invalid_port_is_rejected(value):
    with pytest.raises(ValueError, match='between 1 and 65535'):
        server_runtime.port({'DUBLY_PORT': value})


def test_default_and_custom_ports():
    assert server_runtime.port({}) == 20806
    assert server_runtime.port({'DUBLY_PORT': '34567'}) == 34567


@pytest.mark.parametrize('host,token', [('127.0.0.1', ''), ('127.0.0.1', 'wrong'), ('192.0.2.3', 'owned')])
def test_foreign_clients_cannot_stop_server(monkeypatch, host, token):
    monkeypatch.setenv('DUBLY_LAUNCHER_TOKEN', 'owned')
    stopped = []
    monkeypatch.setattr(server_runtime, '_shutdown', lambda: stopped.append(True))
    app = FastAPI(); app.include_router(server_runtime.router())
    with TestClient(app, client=(host, 12345)) as client:
        headers = {'Authorization': 'Bearer ' + token}
        status = client.get('/api/launcher/health', headers=headers).json()
        assert status['managed'] is False and status['pid'] is None
        assert client.post('/api/launcher/shutdown', headers=headers).status_code == 403
    assert not stopped


def test_owned_loopback_client_can_stop_server(monkeypatch):
    import time
    monkeypatch.setenv('DUBLY_LAUNCHER_TOKEN', 'owned')
    stopped = []
    monkeypatch.setattr(server_runtime, '_shutdown', lambda: stopped.append(True))
    app = FastAPI(); app.include_router(server_runtime.router())
    with TestClient(app, client=('127.0.0.1', 12345)) as client:
        headers = {'Authorization': 'Bearer owned'}
        assert client.get('/api/launcher/health', headers=headers).json()['managed'] is True
        assert client.post('/api/launcher/shutdown', headers=headers).status_code == 200
        time.sleep(.15)
    assert stopped == [True]
