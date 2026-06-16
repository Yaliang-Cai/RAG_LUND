import pytest
from raganything.agent import vision
from raganything.agent.vision import build_multimodal_messages, vlm_generate


def test_invalid_images_yield_text_only_message(monkeypatch):
    monkeypatch.setattr(vision, "validate_image_file", lambda p: False)
    messages, n = build_multimodal_messages("prompt text", ["a.jpg", "b.png"])
    assert n == 0
    assert messages == [{"role": "user", "content": "prompt text"}]


def test_valid_images_interleaved_and_capped(monkeypatch):
    monkeypatch.setattr(vision, "validate_image_file", lambda p: True)
    monkeypatch.setattr(vision, "encode_image_to_base64", lambda p: "B64" + p)
    messages, n = build_multimodal_messages("ctx", ["1.jpg", "2.jpg", "3.jpg"], max_images=2)
    assert n == 2  # hard cap honored
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "ctx"}
    image_parts = [c for c in content if c["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,B64")


def test_unreadable_image_skipped(monkeypatch):
    monkeypatch.setattr(vision, "validate_image_file", lambda p: True)
    monkeypatch.setattr(vision, "encode_image_to_base64", lambda p: "")  # encode fails
    messages, n = build_multimodal_messages("ctx", ["x.jpg"])
    assert n == 0 and messages[0]["content"] == "ctx"


@pytest.mark.asyncio
async def test_vlm_generate_calls_vision_fn(monkeypatch):
    monkeypatch.setattr(vision, "validate_image_file", lambda p: True)
    monkeypatch.setattr(vision, "encode_image_to_base64", lambda p: "B64")
    captured = {}

    async def vision_fn(prompt, messages=None):
        captured["prompt"] = prompt
        captured["messages"] = messages
        return "VLM ANSWER"

    out = await vlm_generate(vision_fn, "ctx", ["x.jpg"])
    assert out == "VLM ANSWER"
    assert captured["prompt"] == ""                       # query.py convention
    assert captured["messages"][0]["role"] == "user"
