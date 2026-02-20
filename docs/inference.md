# Inference
Inference can be run by loading a checkpoint into the same model class and calling the model's `predict_step`.

## 1) Build model from config + load checkpoint
```python
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer

from data.datamodule import DepthTileDataModule
from data.dataset_temp_v1 import SurfaceTempPatchLightDataset
from models.difFF import PixelDiffusionConditional


def load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


model_config_path = "configs/model_config.yaml"
data_config_path = "configs/data_config.yaml"
training_config_path = "configs/training_config.yaml"
ckpt_path = "logs/<run>/best.ckpt"

dataset = SurfaceTempPatchLightDataset.from_config(data_config_path, split="all")
datamodule = DepthTileDataModule(dataset=dataset)
datamodule.setup("fit")

model = PixelDiffusionConditional.from_config(
    model_config_path=model_config_path,
    data_config_path=data_config_path,
    training_config_path=training_config_path,
    datamodule=datamodule,
)

checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

If you run the EO multiband mode, swap the dataset class to `SurfaceTempPatch4BandsLightDataset` and use the EO config files.

## 2) Run a single `predict_step` directly
For `PixelDiffusionConditional`, `predict_step` expects at least `x`; it will also use `valid_mask` and optional `coords` if present.

```python
batch = next(iter(datamodule.val_dataloader()))
batch = {k: (v if not torch.is_tensor(v) else v.to(model.device)) for k, v in batch.items()}

with torch.no_grad():
    pred = model.predict_step(batch, batch_idx=0)

y_hat = pred["y_hat"]               # standardized output
y_hat_denorm = pred["y_hat_denorm"] # temperature-denormalized output
```

## 3) Run prediction through Lightning
```python
trainer = Trainer(accelerator="auto", devices="auto", logger=False)
predictions = trainer.predict(model=model, dataloaders=datamodule.val_dataloader())
```

Each prediction item includes:
- `y_hat`
- `y_hat_denorm`
- `denoise_samples` (if intermediates requested)
- `sampler`
