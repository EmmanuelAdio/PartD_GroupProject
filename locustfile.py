from __future__ import annotations

import random

from locust import HttpUser, between, task

from scripts.benchmark_questions import BENCHMARK_QUESTIONS


class QueryUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query_endpoint(self) -> None:
        item = random.choice(BENCHMARK_QUESTIONS)
        payload = {
            "query": item["question"],
            "debug": False,
        }
        with self.client.post("/query", json=payload, catch_response=True, name="POST /query") as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
                return
            try:
                data = response.json()
            except Exception as exc:
                response.failure(f"Response was not valid JSON: {exc}")
                return

            if isinstance(data, dict) and data.get("detail"):
                response.failure(f"API returned error payload: {data['detail']}")
                return
            if not isinstance(data, dict) or not data.get("answer"):
                response.failure("Missing answer in response payload.")
                return
            response.success()
