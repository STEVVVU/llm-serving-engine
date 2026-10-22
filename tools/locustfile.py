"""Locust owns load/statistics; this adapter parses the product's custom SSE."""
import json
import os
import time
from collections import Counter
from pathlib import Path

from locust import HttpUser, constant, events, task

COUNTS = Counter()
FIXTURE = json.loads(Path(os.environ["SERVING_ORACLE"]).read_text())


class ServingUser(HttpUser):
    wait_time = constant(0)
    index = 0

    @task
    def completion(self):
        index = self.index % len(FIXTURE["prompts"])
        self.index += 1
        COUNTS["offered"] += 1
        classified = False
        start = time.perf_counter()
        ticks, final = [], None
        body = {"prompt": FIXTURE["prompts"][index], "max_tokens": FIXTURE["output_tokens"], "ignore_eos": True, "temperature": 0, "stream": True}
        try:
            with self.client.post("/v1/completions", json=body, stream=True, catch_response=True, name="completion", timeout=120) as response:
                try:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1):
                        if not line.startswith(b"data: ") or line == b"data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        if "error" in event:
                            raise RuntimeError(event["error"])
                        if event.get("done"):
                            final = event["token_ids"]
                            ticks.extend([time.perf_counter()] * max(0, len(final) - len(ticks)))
                        else:
                            ticks.extend([time.perf_counter()] * len(event["token_ids"]))
                    assert final == FIXTURE["expected"][index]
                    COUNTS["completed"] += 1
                    classified = True
                    for name, value in [("TTFT", ticks[0] - start)] + [("ITL", b - a) for a, b in zip(ticks, ticks[1:])]:
                        self.environment.events.request.fire(request_type="SSE", name=name, response_time=value * 1000, response_length=0, exception=None)
                except Exception as error:
                    key = "rejected" if response.status_code in (429, 503) else "failed"
                    COUNTS[key] += 1
                    classified = True
                    response.failure(str(error))
                finally:
                    response.request_meta["response_time"] = (time.perf_counter() - start) * 1000
        finally:
            if not classified:
                COUNTS["client_aborted"] += 1


@events.test_stop.add_listener
def retained_accounting(environment, **kwargs):
    output = Path(os.environ["SERVING_COUNTS"])
    output.write_text(json.dumps(dict(COUNTS, policy="Locust closed-loop users; no open-loop/coordinated-omission claim"), indent=2))
    assert COUNTS["offered"] == COUNTS["completed"] + COUNTS["rejected"] + COUNTS["failed"] + COUNTS["client_aborted"]
