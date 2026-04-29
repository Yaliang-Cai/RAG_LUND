# tests/test_observability.py
from raganything.observability import setup_phoenix


def test_setup_phoenix_does_not_raise_without_phoenix(monkeypatch):
    """setup_phoenix must not raise even when arize-phoenix is not installed."""
    import sys
    monkeypatch.setitem(sys.modules, "phoenix", None)
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    setup_phoenix()
    setup_phoenix()  # second call: idempotent


def test_setup_phoenix_sets_initialized_flag():
    import raganything.observability as obs
    obs._phoenix_initialized = False
    setup_phoenix()
    assert obs._phoenix_initialized is True
