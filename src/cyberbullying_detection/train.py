import subprocess
import tempfile
from pathlib import Path

import hydra
import lightning as L  # noqa: N812
import mlflow
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from cyberbullying_detection.data import CyberbullyingDataModule
from cyberbullying_detection.model import SimpleAttnMLP
from cyberbullying_detection.module import CyberbullyingModule


@hydra.main(version_base=None, config_path="../../configs", config_name="conf")
def train(cfg: DictConfig):
    if not all(
        (Path("data") / file).exists()
        for file in ("data_train.csv", "data_val.csv", "data_test.csv")
    ):
        subprocess.run(["dvc", "pull", "-r", "cyberbullying-remote", "data"], check=True)

    tracking_uri = "http://127.0.0.1:8080"
    experiment_name = getattr(cfg, "experiment_name", "cyberbullying_detection")
    registered_model_name = getattr(cfg, "registered_model_name", "CyberbullyingModel")

    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    datamodule = CyberbullyingDataModule(cfg.data)
    model = SimpleAttnMLP(vocab_size=datamodule.tokenizer.vocab_size, cfg=cfg.model)
    module = CyberbullyingModule(model=model, cfg=cfg.module)

    trainer = L.Trainer(max_epochs=cfg.num_epochs, logger=mlf_logger, callbacks=[checkpoint])
    trainer.fit(module, datamodule=datamodule)

    run_id = mlf_logger.run_id
    mlflow.set_tracking_uri(tracking_uri)

    if checkpoint.best_model_path:
        loaded_checkpoint = torch.load(
            checkpoint.best_model_path, map_location="cpu", weights_only=False
        )
        module.load_state_dict(loaded_checkpoint["state_dict"])
    module.eval()

    with mlflow.start_run(run_id=run_id):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "config.yaml"
            OmegaConf.save(cfg, cfg_path)
            mlflow.log_artifact(str(cfg_path), artifact_path="config")

            tok_dir = Path(d) / "tokenizer"
            datamodule.tokenizer.save_pretrained(tok_dir)
            mlflow.log_artifacts(str(tok_dir), artifact_path="tokenizer")

        mlflow.pytorch.log_model(
            module.model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        client = MlflowClient(tracking_uri=tracking_uri)
        versions = client.search_model_versions(
            f"name='{registered_model_name}' and run_id='{run_id}'"
        )
        new_version = max(int(v.version) for v in versions)

        client.transition_model_version_stage(
            name=registered_model_name,
            version=str(new_version),
            stage="Staging",
            archive_existing_versions=False,
        )

    print(f"Registered {registered_model_name} v{new_version} -> Staging")


if __name__ == "__main__":
    train()
