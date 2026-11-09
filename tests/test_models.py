"""Model adapter and registry tests; mlx-vlm is replaced by a fake, so nothing is downloaded."""

import contextlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from thermground.device import get_device
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


def fake_torch(cuda=False, mps=False):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    torch.inference_mode = contextlib.nullcontext
    return torch


@pytest.mark.parametrize(
    "cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_get_device(monkeypatch, cuda, mps, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda, mps))
    assert get_device() == expected
    assert get_device("cpu") == "cpu"


class FakeProcessor:
    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        assert chat[0]["content"][0] == {"type": "image"} and add_generation_prompt
        return "<chat>" + chat[0]["content"][1]["text"]

    def __call__(self, text, images, return_tensors):
        self.seen = (text, images)
        return SimpleNamespace(to=lambda device: {"input_ids": np.zeros((1, 3), dtype=int)})

    def decode(self, ids, skip_special_tokens):
        return " yes \n" if list(ids) == [7, 8] else "?"


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, input_ids, max_new_tokens, do_sample):
        self.max_new_tokens = max_new_tokens
        return np.array([[0, 0, 0, 7, 8]])


def test_hf_adapter_call(monkeypatch, tmp_path):
    from thermground.models import hf_vlm

    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    processor, model = FakeProcessor(), FakeModel()
    monkeypatch.setattr(hf_vlm, "load_base", lambda model_id, revision, dtype: (processor, model))
    spec = write_config(tmp_path, backend="hf", dtype="bfloat16")
    vlm, cfg = load_model(str(spec), {}, lambda n: None, [])
    assert isinstance(vlm, hf_vlm.HFVLM) and cfg["backend"] == "hf" and model.device == "cpu"

    assert vlm.answer(query("presence")) == "yes"
    (text,), (image,) = processor.seen
    assert text == "<chat>Is there a person?" and image.mode == "RGB"
    assert model.max_new_tokens == 24
