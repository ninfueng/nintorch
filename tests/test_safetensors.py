import os
import tempfile

from torchvision.models import resnet18

from nintorch.io import load_safetensors, save_safetensors


def test_save_load_safetensors() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = resnet18()
        model_dir = os.path.join(tmpdir, 'resnet18.safetensors')

        save_safetensors(model.state_dict(), model_dir)
        model_state_dict = load_safetensors(model_dir)
        model.load_state_dict(model_state_dict)
