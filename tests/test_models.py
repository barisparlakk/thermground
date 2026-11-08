"""Model adapter and registry tests; mlx-vlm is replaced by a fake, so nothing is downloaded."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from thermground.models.base import Query
from thermground.models.registry import load_model


@pytest.fixture
def fake_mlx_vlm(monkeypatch):
    calls = {}

    def load(model_id, revision=None):
        calls["load"] = (model_id, revision)
        return SimpleNamespace(config={"model_type": "fake"}), "processor"

    def generate(model, processor, prompt, image, max_tokens, temperature, verbose):
        calls["generate"] = {
            "prompt": prompt,
            "image": image,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return SimpleNamespace(text="  Yes.\n")

    def apply_chat_template(processor, config, prompt, num_images):
        return f"<chat images={num_images}>{prompt}"

    module = types.ModuleType("mlx_vlm")
    module.load, module.generate = load, generate
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = apply_chat_template
    monkeypatch.setitem(sys.modules, "mlx_vlm", module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    return calls


def write_config(tmp_path, **overrides):
    cfg = {
        "name": "fake_vlm",
        "backend": "mlx_vlm",
        "model_id": "org/fake-vlm",
        "revision": "abc123",
        "coord_format": "norm1000",
        "max_tokens": {"short": 24, "point": 256},
        **overrides,
    }
    path = tmp_path / "fake.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def query(family):
    image = np.arange(512 * 640, dtype=np.uint32).reshape(512, 640).astype(np.uint8)
    return Query(image, "Is there a person?", family, ["yes", "no"], "Person", None)


def test_mlx_adapter_call(fake_mlx_vlm, tmp_path):
    model, cfg = load_model(str(write_config(tmp_path)), {}, lambda n: None, [])
    assert fake_mlx_vlm["load"] == ("org/fake-vlm", "abc123")
    assert (model.name, model.coord_format) == ("fake_vlm", "norm1000")
    assert cfg["model_id"] == "org/fake-vlm"

    assert model.answer(query("presence")) == "Yes."
    call = fake_mlx_vlm["generate"]
    assert call["prompt"] == "<chat images=1>Is there a person?"
    assert call["temperature"] == 0.0 and call["max_tokens"] == 24
    (image,) = call["image"]
    assert image.mode == "RGB" and image.size == (640, 512)
    assert np.array_equal(np.asarray(image)[..., 0], query("presence").image)

    model.answer(query("point"))
    assert fake_mlx_vlm["generate"]["max_tokens"] == 256


def test_registry_reference_and_errors(tmp_path):
    model, cfg = load_model("always_no", {}, lambda n: None, ["0", "1"])
    assert model.name == "always_no" and cfg == {"name": "always_no"}
    with pytest.raises(ValueError, match="neither"):
        load_model("not_a_model", {}, lambda n: None, [])
    with pytest.raises(ValueError, match="unknown backend"):
        load_model(str(write_config(tmp_path, backend="nope")), {}, lambda n: None, [])
