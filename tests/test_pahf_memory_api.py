"""Tests for PAHF memory API endpoints."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.pahf_memory.service import PAHFMemoryService


class FakeEmbedding:
    @staticmethod
    def _vector(text: str) -> list[float]:
        text = (text or "").lower()
        return [
            float("concise" in text),
            float("direct" in text),
            float("xiaohao" in text),
            float("shoe" in text),
        ]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeLLM:
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        if "Decision: ASK or PROCEED" in prompt:
            return "Decision: PROCEED\nQuestion:"
        if "Store: YES or NO" in prompt:
            return "Store: NO\nSummary:"
        if "Are these two memories about the same user-preference topic/domain?" in prompt:
            return "Yes"
        if "Please create a concise, integrated summary" in prompt:
            return "I prefer concise and direct answers."
        return "No"


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="pahf_api_"))


def build_service(tmp_dir: Path) -> PAHFMemoryService:
    return PAHFMemoryService(
        backend="sqlite",
        sqlite_db_path=str(tmp_dir / "pahf.db"),
        faiss_path=str(tmp_dir / "pahf_index"),
        top_k=5,
        similarity_threshold=0.2,
        llm_client=FakeLLM(),
        query_encoder="unused-query",
        context_encoder="unused-context",
        device=None,
        enable_pre_clarification=True,
        enable_post_correction=True,
        embedding_model=FakeEmbedding(),
    )


@pytest.fixture
def client_with_pahf():
    tmp = _tmp_dir()
    service = build_service(tmp)
    try:
        with patch("backend.main.build_pahf_memory_service", return_value=service):
            with TestClient(app) as client:
                yield client
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pahf_memory_api_crud_and_search(client_with_pahf: TestClient):
    create_res = client_with_pahf.post(
        "/api/v1/memory",
        json={"user_id": "api_user", "text": "I prefer concise answers."},
    )
    assert create_res.status_code == 200
    memory = create_res.json()
    memory_id = memory["id"]
    assert memory["person_id"] == "api_user"

    list_res = client_with_pahf.get("/api/v1/memory", params={"user_id": "api_user"})
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    get_res = client_with_pahf.get(f"/api/v1/memory/{memory_id}", params={"user_id": "api_user"})
    assert get_res.status_code == 200
    assert get_res.json()["id"] == memory_id

    update_res = client_with_pahf.put(
        f"/api/v1/memory/{memory_id}",
        json={"user_id": "api_user", "text": "I prefer concise and direct answers."},
    )
    assert update_res.status_code == 200
    assert "direct" in update_res.json()["text"]

    search_res = client_with_pahf.post(
        "/api/v1/memory/search",
        json={"user_id": "api_user", "query": "concise response", "top_k": 3},
    )
    assert search_res.status_code == 200
    assert search_res.json()["hits"]

    similar_res = client_with_pahf.post(
        "/api/v1/memory/find-similar",
        json={"user_id": "api_user", "text": "direct concise style", "threshold": 0.1},
    )
    assert similar_res.status_code == 200
    assert similar_res.json() is not None
