import urllib.request

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Unit tests must never reach a real Ollama (chat or embeddings). Tests that need a response fake urlopen themselves."""
    def boom(*a, **k):
        raise AssertionError("test tried to open a network connection; pass a fake chat/embed instead")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
