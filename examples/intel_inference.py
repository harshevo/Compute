"""Place an OpenVINO model.xml and model.bin beside this script before submitting."""
from pathlib import Path
import numpy as np
from compute_sdk import OpenVINOModel


def build_model(device):
    target = device.split(":", 1)[1] if device.startswith("openvino:") else "CPU"
    return OpenVINOModel(Path(__file__).with_name("model.xml"), target)


def generate(model, messages, parameters):
    # Adapt preprocessing and outputs to your model; this illustrates numeric inference.
    shape = model.model.input(0).shape
    outputs = model(np.ones(shape, dtype=np.float32))
    yield str(next(iter(outputs.values())).tolist())
