"""Userspace TCP network-emulation proxy (no root, no tc/netem).

Sits between the pipeline client and the remote inference server on the same
node:  client -> 127.0.0.1:listen_port -> [delay/jitter/loss] -> 127.0.0.1:target_port

Per forwarded chunk and per direction, an artificial delay is applied:

    delay = rtt_ms / 2 + N(0, jitter_ms)          (clamped at >= 0)
    with probability loss_pct/100: delay += loss_penalty_ms

Loss emulation rationale: dropping bytes of an established TCP stream in
userspace would corrupt the stream, not emulate IP packet loss — real TCP
recovers lost packets via retransmission, whose externally observable effect
is added latency. We therefore map loss to a stochastic latency penalty
(default of the order of a fast-retransmit/RTO recovery). This methodological
choice is documented in README and the paper.

Implementation: asyncio; each direction is forwarded through an ordered queue
whose consumer releases each chunk not earlier than its scheduled time, which
preserves TCP byte ordering while shaping latency.
"""

from __future__ import annotations

import asyncio
import random
import threading
from typing import Any, Optional


class NetProxy:
    def __init__(
        self,
        listen_port: int,
        target_port: int,
        rtt_ms: float = 0.0,
        jitter_ms: float = 0.0,
        loss_pct: float = 0.0,
        loss_penalty_ms: float = 0.0,
        seed: int = 42,
        host: str = "127.0.0.1",
    ):
        self.listen_port = listen_port
        self.target_port = target_port
        self.host = host
        self.one_way_s = rtt_ms / 2.0 / 1000.0
        self.jitter_s = jitter_ms / 1000.0
        self.loss_p = loss_pct / 100.0
        self.loss_penalty_s = loss_penalty_ms / 1000.0
        self.rng = random.Random(seed)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._started = threading.Event()
        self.n_chunks = 0
        self.n_loss_events = 0

    # ------------------------------------------------------------------ delay
    def _sample_delay(self) -> float:
        d = self.one_way_s
        if self.jitter_s > 0:
            d += self.rng.gauss(0.0, self.jitter_s)
        if self.loss_p > 0 and self.rng.random() < self.loss_p:
            d += self.loss_penalty_s
            self.n_loss_events += 1
        return max(0.0, d)

    # ------------------------------------------------------------- forwarding
    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        async def consumer() -> None:
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    release_at, chunk = item
                    wait = release_at - loop.time()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    writer.write(chunk)
                    await writer.drain()
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

        consumer_task = asyncio.create_task(consumer())
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                self.n_chunks += 1
                release_at = loop.time() + self._sample_delay()
                await queue.put((release_at, chunk))
        finally:
            await queue.put(None)
            await consumer_task

    async def _handle_client(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        # retry upstream for a grace period — the inference server process may
        # still be starting; on HPC a cold start (torch import from NFS +
        # model load + CUDA init) can take minutes, so the window must
        # comfortably exceed it (the client's handshake retry depends on us
        # keeping its connection open until the server is up)
        upstream_reader = upstream_writer = None
        deadline = asyncio.get_running_loop().time() + 300.0
        try:
            while True:
                try:
                    upstream_reader, upstream_writer = await asyncio.open_connection(
                        self.host, self.target_port
                    )
                    break
                except OSError:
                    if asyncio.get_running_loop().time() > deadline:
                        client_writer.close()
                        return
                    await asyncio.sleep(0.2)
            await asyncio.gather(
                self._pipe(client_reader, upstream_writer),
                self._pipe(upstream_reader, client_writer),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            # proxy is shutting down mid-connection — close quietly
            for w in (upstream_writer, client_writer):
                if w is not None:
                    w.close()
            raise

    async def _serve(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.listen_port
        )
        self._started.set()
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass

    async def _drain(self) -> None:
        """Cancel and await all outstanding connection tasks (clean shutdown)."""
        tasks = [t for t in asyncio.all_tasks(self._loop)
                 if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except asyncio.CancelledError:
                pass
            finally:
                # await the cancellation of in-flight connection tasks before
                # closing the loop, so none is garbage-collected while pending
                try:
                    self._loop.run_until_complete(self._drain())
                except Exception:
                    pass
                self._loop.close()

        self._thread = threading.Thread(target=run, name="netproxy", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise TimeoutError("netproxy failed to start within 10 s")

    def stop(self) -> None:
        if self._loop and self._server:
            def _shutdown() -> None:
                assert self._server is not None
                self._server.close()
                # stop serve_forever; _drain() then cleans up connection tasks
                for task in asyncio.all_tasks(self._loop):
                    if task.get_coro().__qualname__.endswith("_serve"):
                        task.cancel()

            self._loop.call_soon_threadsafe(_shutdown)
        if self._thread:
            self._thread.join(timeout=5)

    @classmethod
    def from_profile(
        cls, listen_port: int, target_port: int, profile: dict[str, Any], seed: int = 42
    ) -> "NetProxy":
        return cls(
            listen_port=listen_port,
            target_port=target_port,
            rtt_ms=float(profile.get("rtt_ms", 0.0)),
            jitter_ms=float(profile.get("jitter_ms", 0.0)),
            loss_pct=float(profile.get("loss_pct", 0.0)),
            loss_penalty_ms=float(profile.get("loss_penalty_ms", 0.0)),
            seed=seed,
        )


# ---------------------------------------------------------------------------
# CLI: standalone proxy (useful for manual testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="TTD-Bench userspace network proxy")
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--target-port", type=int, required=True)
    p.add_argument("--rtt-ms", type=float, default=0.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--loss-pct", type=float, default=0.0)
    p.add_argument("--loss-penalty-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    proxy = NetProxy(
        args.listen_port, args.target_port, args.rtt_ms, args.jitter_ms,
        args.loss_pct, args.loss_penalty_ms, args.seed,
    )
    proxy.start()
    print(f"[netproxy] {args.listen_port} -> {args.target_port} "
          f"(rtt {args.rtt_ms} ms, jitter {args.jitter_ms} ms, loss {args.loss_pct}%)",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        proxy.stop()
