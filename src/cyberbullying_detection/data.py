from typing import Any

import lightning as L  # noqa: N812
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def init_dataloader(dataset: Any, batch_size: int, shuffle: bool, num_workers: int = 6):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class CyberbullyingDataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.train_batch_size = cfg.train_batch_size
        self.predict_batch_size = cfg.predict_batch_size
        self.train_data_path = cfg.train_data_path
        self.val_data_path = cfg.val_data_path
        self.test_data_path = cfg.test_data_path
        self.text_column_name = cfg.text_column_name
        self.label_column_name = cfg.label_column_name
        self.label2id = None

    def create_label_map(self, dataset):
        if self.label2id is not None:
            return
        uniq = dataset.unique(self.label_column_name)

        try:
            uniq_sorted = sorted(uniq)
        except TypeError:
            uniq_sorted = sorted([str(x) for x in uniq])

        self.label2id = {lbl: i for i, lbl in enumerate(uniq_sorted)}

    def encode_labels(self, examples):
        labels = examples[self.label_column_name]
        return {self.label_column_name: [self.label2id[lbl] for lbl in labels]}

    def setup(self, stage: str | None = None):
        if stage is None or stage == "fit":
            train_dd = load_dataset("csv", data_files={"train": self.train_data_path})
            val_dd = load_dataset("csv", data_files={"train": self.val_data_path})

            self.create_label_map(train_dd["train"])

            self.train_dataset = (
                train_dd["train"]
                .map(self.encode_labels, batched=True)
                .map(self.tokenize_function, batched=True)
            )
            self.val_dataset = (
                val_dd["train"]
                .map(self.encode_labels, batched=True)
                .map(self.tokenize_function, batched=True)
            )

            self.train_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", self.label_column_name]
            )
            self.val_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", self.label_column_name]
            )

        if stage is None or stage == "test":
            test_dd = load_dataset("csv", data_files={"train": self.test_data_path})
            self.test_dataset = test_dd["train"].map(self.tokenize_function, batched=True)
            self.test_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", self.label_column_name]
            )

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples[self.text_column_name],
            padding="max_length",
            truncation=True,
        )

    def train_dataloader(self):
        return init_dataloader(self.train_dataset, self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return init_dataloader(self.val_dataset, self.predict_batch_size, shuffle=False)

    def test_dataloader(self):
        return init_dataloader(self.test_dataset, self.predict_batch_size, shuffle=False)
