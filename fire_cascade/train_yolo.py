from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cascade import YOLO_PRETRAIN, torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import autocast


class TrainOnlyDetector(DetectionTrainer):
    def get_dataset(self):
        images = Path(self.args.data).resolve()
        if not images.is_dir():
            raise FileNotFoundError(images)
        return {"train": str(images), "names": {0: "fire"}, "nc": 1, "channels": 3}

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            return None
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if len(loader.dataset) != 1100:
            raise ValueError(f"Expected 1100 usable training pairs, found {len(loader.dataset)}")
        return loader

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def _setup_train(self):
        # The real detector/loss is checked on one batch below. Skip the library's
        # separate AMP demo, which otherwise downloads and runs another YOLO model.
        with patch("ultralytics.engine.trainer.check_amp", return_value=True):
            super()._setup_train()
        torch.set_autocast_dtype("cuda", torch.bfloat16)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=1024, enabled=self.amp)
        self.metrics = {}
        self.best_fitness = float("-inf")

    def validate(self):
        # BaseTrainer calls this unconditionally on the last epoch.
        return {}, None

    def final_eval(self):
        pass

    def _handle_nan_recovery(self, epoch):
        if not torch.isfinite(self.tloss).all().item():
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
        return False

    def save_model(self):
        final = self.last
        self.last = final.with_suffix(".pt.tmp")
        try:
            super().save_model()
            self.last.replace(final)
        finally:
            self.last = final


def trainer_for(run, config, name):
    return TrainOnlyDetector(overrides={
        "model": str(YOLO_PRETRAIN), "task": "detect",
        "data": str(run / "training_data" / "images"),
        "project": str(run), "name": name, "exist_ok": False,
        "epochs": config["epochs"], "imgsz": config["image_size"],
        "batch": config["yolo_batch"], "workers": config["workers"],
        "device": "0", "seed": config["seed"], "amp": True,
        "val": False, "patience": 0, "save": True, "plots": False, "cache": "disk",
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "mosaic": 1.0, "flipud": 0.0, "fliplr": 0.5,
        "close_mosaic": 10,
    })


def check_yolo(run, config):
    trainer = trainer_for(run, config, "yolo_check")
    trainer._setup_train()
    trainer.model.train()
    torch.cuda.reset_peak_memory_stats()
    with autocast(trainer.amp):
        batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
        loss, components = trainer.model(batch)
    trainer.scaler.scale(loss.sum()).backward()
    trainer.scaler.unscale_(trainer.optimizer)
    grad = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=float("inf"))
    torch.cuda.synchronize()
    print(f"YOLO one batch: loss={components.tolist()}, grad_norm={grad.item():.3f}, "
          f"peak allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    if not (torch.isfinite(components).all().item() and grad.isfinite().item() and grad.item() > 0):
        raise RuntimeError("YOLO real-batch AMP loss or gradients failed")
