"""Bounded local HTTP/SSE adapter. One worker owns every engine operation."""
import argparse
import ipaddress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time
import uuid

from nanovllm import LLM, SamplingParams


@dataclass
class Client:
    body: dict
    events: queue.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seq_id: int | None = None
    sent: int = 0
    text: str = ""
    created: float = field(default_factory=time.monotonic)


class Service:
    def __init__(self, engine, max_clients=16, buffer_tokens=16, deadline=120):
        self.engine = engine
        self.commands = queue.Queue(max_clients)
        self.cancellations = queue.Queue(max_clients)
        self.buffer_tokens = buffer_tokens
        self.deadline = deadline
        self.clients = {}
        self.stop = threading.Event()
        self.snapshot = {}
        self.counts = dict(accepted=0, completed=0, cancelled=0, failed=0)
        self.thread = threading.Thread(target=self.run, name="model-owner")
        self.thread.start()

    def terminal(self, client, event, status):
        if client.seq_id is not None:
            self.engine.cancel_request(client.seq_id)
            self.clients.pop(client.seq_id, None)
        self.counts[status] += 1
        while True:
            try:
                client.events.put_nowait(event)
                break
            except queue.Full:
                try:
                    client.events.get_nowait()
                except queue.Empty:
                    pass

    def admit(self, client):
        body = client.body
        try:
            params = SamplingParams(temperature=body.get("temperature", 0), max_tokens=body.get("max_tokens", 16), ignore_eos=body.get("ignore_eos", False))
            prompt = body.get("prompt")
            if "messages" in body:
                messages = body["messages"]
                if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
                    raise ValueError("messages must have valid roles and string content")
                prompt = self.engine.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if not isinstance(prompt, (str, list)):
                raise ValueError("prompt must be a string or token ID list")
            client.seq_id = self.engine.add_request(prompt, params)
            self.clients[client.seq_id] = client
            self.counts["accepted"] += 1
        except (ValueError, TypeError) as exc:
            client.events.put_nowait(dict(error=str(exc), status=400))

    def update_stats(self):
        self.snapshot = dict(self.counts, **self.engine.scheduler.block_manager.stats(),
            live_requests=len(self.clients), pending_commands=self.commands.qsize(),
            waiting=len(self.engine.scheduler.waiting), running=len(self.engine.scheduler.running),
            graph_replays=getattr(self.engine.model_runner, "graph_replays", 0))

    def tick(self):
        for _ in range(self.cancellations.maxsize):
            try:
                request_id = self.cancellations.get_nowait()
            except queue.Empty:
                break
            for client in self.clients.values():
                if client.request_id == request_id:
                    client.cancelled.set()
        for _ in range(self.commands.maxsize):
            try:
                client = self.commands.get_nowait()
            except queue.Empty:
                break
            if not client.cancelled.is_set():
                self.admit(client)
        for client in list(self.clients.values()):
            if client.cancelled.is_set() or time.monotonic() - client.created > self.deadline:
                self.terminal(client, dict(error="cancelled or deadline exceeded", status=408), "cancelled")
        if not self.engine.is_finished():
            before = {s.seq_id: s for s in list(self.engine.scheduler.waiting) + list(self.engine.scheduler.running) + list(self.engine.scheduler.completed)}
            finished, _ = self.engine.step()
            done = dict(finished)
            for seq_id, seq in before.items():
                client = self.clients.get(seq_id)
                if client is None:
                    continue
                tokens = seq.completion_token_ids
                if seq_id in done:
                    self.terminal(client, dict(done=True, token_ids=tokens, text=self.engine.tokenizer.decode(tokens, skip_special_tokens=True), request_id=client.request_id), "completed")
                elif len(tokens) > client.sent:
                    text = self.engine.tokenizer.decode(tokens, skip_special_tokens=True).rstrip("\ufffd")
                    event = dict(token_ids=tokens[client.sent:], text=text[len(client.text):], request_id=client.request_id)
                    try:
                        client.events.put_nowait(event)
                        client.sent, client.text = len(tokens), text
                    except queue.Full:
                        self.terminal(client, dict(error="client buffer overflow", status=429), "cancelled")
        self.update_stats()

    def run(self):
        try:
            while not self.stop.is_set():
                try:
                    self.tick()
                except Exception as exc:
                    for client in list(self.clients.values()):
                        self.terminal(client, dict(error=str(exc), status=500), "failed")
                    self.update_stats()
                if not self.clients and self.commands.empty():
                    self.stop.wait(0.005)
        finally:
            for client in list(self.clients.values()):
                self.terminal(client, dict(error="server stopped", status=503), "cancelled")
            self.engine.exit()
            self.update_stats_after_exit()

    def update_stats_after_exit(self):
        self.snapshot = dict(self.snapshot, live_requests=0, stopped=True)

    def close(self):
        self.stop.set()
        self.thread.join()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service, max_clients=16, max_body=65536):
        self.service, self.max_body = service, max_body
        self.slots = threading.BoundedSemaphore(max_clients)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        request.settimeout(5)
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/metrics"):
            self.reply(200, self.server.service.snapshot)
        else:
            self.reply(404, dict(error="unknown route"))

    def do_POST(self):
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self.reply(404, dict(error="unknown route"))
            return
        client = None
        streaming = False
        try:
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, dict(error="chunked request bodies are unsupported"))
                return
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= self.server.max_body:
                self.reply(413, dict(error="bounded Content-Length required"))
                return
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("incomplete request body")
            body = json.loads(data)
            if not isinstance(body, dict) or not isinstance(body.get("stream", False), bool):
                raise ValueError("body must be an object and stream must be boolean")
            service = self.server.service
            client = Client(body, queue.Queue(service.buffer_tokens))
            try:
                service.commands.put_nowait(client)
            except queue.Full:
                self.reply(429, dict(error="admission queue full"))
                return
            while True:
                event = client.events.get(timeout=service.deadline + 5)
                if "error" in event:
                    if streaming:
                        self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
                        self.wfile.flush()
                    else:
                        self.reply(event["status"], event)
                    return
                if body.get("stream"):
                    if not streaming:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        streaming = True
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                if event.get("done"):
                    if streaming:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        self.reply(200, dict(id=event["request_id"], choices=[dict(text=event["text"], token_ids=event["token_ids"], finish_reason="stop")]))
                    return
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if not streaming:
                self.reply(400, dict(error=str(exc)))
        except (OSError, queue.Empty):
            pass
        finally:
            if client is not None:
                client.cancelled.set()

    def do_DELETE(self):
        prefix = "/requests/"
        if not self.path.startswith(prefix) or len(self.path) != len(prefix) + 32:
            self.reply(404, dict(error="unknown request route"))
            return
        try:
            self.server.service.cancellations.put_nowait(self.path[len(prefix):])
            self.reply(202, dict(cancellation_requested=True))
        except queue.Full:
            self.reply(429, dict(error="cancellation queue full"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("torch", "native"), default="torch")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-clients", type=int, default=16)
    parser.add_argument("--client-buffer", type=int, default=16)
    parser.add_argument("--deadline", type=float, default=120)
    parser.add_argument("--max-body", type=int, default=65536)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    try:
        address = ipaddress.ip_address(args.host)
        if address.version != 4 or not address.is_loopback:
            raise ValueError
    except ValueError:
        parser.error("host must be a numeric IPv4 loopback address")
    if min(args.max_clients, args.client_buffer, args.deadline, args.max_body) <= 0:
        parser.error("server bounds must be positive")
    engine = LLM(args.model, device=args.device, backend=args.backend,
        kvcache_block_size=args.page_size, num_kvcache_blocks=args.pages,
        max_num_seqs=args.max_running, max_num_batched_tokens=args.token_budget,
        max_model_len=args.max_model_len, max_pending_requests=args.max_clients,
        enforce_eager=args.eager, compile=args.compile)
    service = Service(engine, args.max_clients, args.client_buffer, args.deadline)
    server = Server((args.host, args.port), service, args.max_clients, args.max_body)
    print(json.dumps(dict(ready=True, port=server.server_port)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
