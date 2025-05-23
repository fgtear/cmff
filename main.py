import os
import warnings

# warnings.filterwarnings("error")
# os.environ["http_proxy"] = "http://u-KgKRFF:5XvYKmDW@10.255.128.102:3128"
# os.environ["https_proxy"] = "http://u-KgKRFF:5XvYKmDW@10.255.128.102:3128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import importlib
import torch
from config import Config
from utils.mlflow_utils import log_files

from utils.utils import seed_all
import gc
import time
import mlflow

# mlflow.enable_system_metrics_logging()
# mlflow.set_system_metrics_sampling_interval(1)

# np.seterr(all='raise')


def main(config):
    if config.platform != "Darwin":
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("high")  # optional: 'highest', 'high', 'medium'
    seed_all(config.seed)
    LightningData = importlib.import_module(f"{config.method}.lightning_data").LightningData
    LightningModel = importlib.import_module(f"{config.method}.lightning_model").LightningModel

    logger = MLFlowLogger(
        tracking_uri="file:./mlruns",
        experiment_name="Default",  # config.Dataset, Default
        run_name=config.run_name,
        # log_model=True,
    )
    log_files(logger=logger, code_dir=os.getcwd(), save_dir="code", target_extensions=[".py", ".toml", ".lock"])

    checkpoint_path = os.path.join("mlruns", logger.experiment_id, logger.run_id, "artifacts/checkpoints")
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        dirpath=checkpoint_path,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=1,
        every_n_epochs=1,  # default is None
        # save_last=True,  # default is None
        # every_n_train_steps=None,  # default is None
        # filename="best",
    )

    earlystop_callback = EarlyStopping(
        monitor=config.monitor,
        mode=config.monitor_mode,
        patience=8,
        verbose=False,
        check_finite=True,
    )

    trainer = L.Trainer(
        check_val_every_n_epoch=1,  # default is 1
        logger=logger,
        callbacks=[
            checkpoint_callback,
            # earlystop_callback,
        ],
        # deterministic="True",  # default is None, optional is True/"warn"
        log_every_n_steps=1,  # default is 50/None, mean log every 50 steps and the end of epoch, 0 means log at end of epoch
        fast_dev_run=config.fast_dev_run,
        gradient_clip_val=config.gradient_clip_val,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # profiler=AdvancedProfiler(),  # default is None, optional is SimpleProfiler, AdvancedProfiler
        # default_root_dir="logs",
        num_sanity_val_steps=0 if config.platform == "Darwin" else 2,
        limit_train_batches=1 if config.platform == "Darwin" else 1.0,
        limit_val_batches=1 if config.platform == "Darwin" else 1.0,
        limit_test_batches=1 if config.platform == "Darwin" else 1.0,
    )

    # log_files(
    #     run_id=logger.run_id,
    #     code_dir=os.getcwd(),
    #     save_dir="code",
    #     target_extensions=[".py"],
    # )
    model = LightningModel(config)
    dm = LightningData(config)
    # trainer.validate(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", dataloaders=dm)  # best or last

    # 将模型改名为best_model
    os.rename(checkpoint_callback.best_model_path, os.path.join(checkpoint_path, "best_model.ckpt"))
    # 把最优的epoch保存到txt文件中
    with open(os.path.join(checkpoint_path, "best_model.txt"), "w") as f:
        f.write(str(checkpoint_callback.best_model_path))
    if not config.save_model:
        os.remove(os.path.dirname(checkpoint_callback.best_model_path))


if __name__ == "__main__":
    config = Config()
    main(config)
