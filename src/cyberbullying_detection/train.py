import hydra
import lightning as L  # noqa: N812
import torch
from omegaconf import DictConfig

from cyberbullying_detection.data import CyberbullyingDataModule
from cyberbullying_detection.model import SimpleAttnMLP
from cyberbullying_detection.module import CyberbullyingModule


@hydra.main(version_base=None, config_path="../../configs", config_name="conf")
def train(cfg: DictConfig):
    datamodule = CyberbullyingDataModule(cfg.data)
    model = SimpleAttnMLP(
        vocab_size=datamodule.tokenizer.vocab_size,
        cfg=cfg.model,
    )

    module = CyberbullyingModule(model=model, cfg=cfg.module)

    trainer = L.Trainer(max_epochs=cfg.num_epochs)
    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
