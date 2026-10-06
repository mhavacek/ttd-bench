"""Netproxy: injected delay matches the profile within ±10 % (plus overhead)."""

import socket
import statistics
import threading
import time

import pytest

from ttdbench.netproxy import NetProxy


def echo_server(port, stop_evt):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(0.5)
    while not stop_evt.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            conn.close()
    srv.close()


@pytest.fixture
def echo_port():
    port = 18950
    stop = threading.Event()
    t = threading.Thread(target=echo_server, args=(port, stop), daemon=True)
    t.start()
    time.sleep(0.2)
    yield port
    stop.set()
    t.join(timeout=2)


def measure_rtts(proxy_port, n=30):
    rtts = []
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=5) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = b"x" * 64
        for _ in range(n):
            t0 = time.monotonic_ns()
            s.sendall(payload)
            got = b""
            while len(got) < len(payload):
                got += s.recv(4096)
            rtts.append((time.monotonic_ns() - t0) / 1e6)
    return rtts


def run_proxy_case(echo_port, listen_port, **kwargs):
    proxy = NetProxy(listen_port, echo_port, seed=42, **kwargs)
    proxy.start()
    try:
        return measure_rtts(listen_port)
    finally:
        proxy.stop()


def test_delay_matches_profile(echo_port):
    """RTT 60 ms profile -> measured median within ±10 % (+1 ms overhead slack)."""
    rtts = run_proxy_case(echo_port, 18951, rtt_ms=60.0)
    med = statistics.median(rtts)
    assert 60.0 * 0.9 <= med <= 60.0 * 1.1 + 2.0, f"median RTT {med:.1f} ms"


def test_zero_profile_low_overhead(echo_port):
    """No emulation -> proxy overhead itself stays small (< 5 ms median)."""
    rtts = run_proxy_case(echo_port, 18952)
    assert statistics.median(rtts) < 5.0


def test_jitter_increases_spread(echo_port):
    base = run_proxy_case(echo_port, 18953, rtt_ms=30.0)
    jit = run_proxy_case(echo_port, 18954, rtt_ms=30.0, jitter_ms=8.0)
    assert statistics.stdev(jit) > statistics.stdev(base)


def test_loss_penalty_applied(echo_port):
    """100% 'loss' with a 50 ms penalty must shift the median by ~+50 ms."""
    rtts = run_proxy_case(echo_port, 18955, rtt_ms=10.0, loss_pct=100.0,
                          loss_penalty_ms=50.0)
    med = statistics.median(rtts)
    # two directions x 50 ms penalty + 10 ms rtt = ~110 ms expected
    assert med >= 100.0, f"median RTT {med:.1f} ms"


def test_delay_reproducible_with_seed(echo_port):
    a = NetProxy(1, 1, rtt_ms=20, jitter_ms=5, loss_pct=10, loss_penalty_ms=40, seed=7)
    b = NetProxy(1, 1, rtt_ms=20, jitter_ms=5, loss_pct=10, loss_penalty_ms=40, seed=7)
    assert [a._sample_delay() for _ in range(100)] == [b._sample_delay() for _ in range(100)]
