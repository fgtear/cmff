import datetime
import platform
import copy
import torch
import os


class Config:
    def __init__(self):
        self.platform = platform.system()
        self.devices = [0] if self.platform != "Darwin" else "auto"
        self.strategy = "auto"
        self.accelerator = "auto" if self.platform != "Darwin" else "cpu"
        self.num_workers = 2 if self.platform != "Darwin" else 0
        self.num_GPU = len(self.devices) if self.platform != "Darwin" else 1
        self.precision = "16-mixed" if self.platform != "Darwin" else "32"  # 16-mixed, bf16-mixed

        self.dataset = "MOSI"  # MOSI, MOSEI
        self.save_model = True
        self.method = "m1"
        # bert-base-uncased, roberta-large
        self.text_extractor = "roberta-large"
        # facebook/data2vec-audio-large-960h
        # facebook/data2vec-audio-base-960h
        # facebook/wav2vec2-large-robust-ft-libri-960h
        self.audio_extractor = "facebook/wav2vec2-large-robust-ft-libri-960h"
        self.monitor = "metrics/MAE_val"  # metrics/MAE_val, metrics/MAE_test
        self.monitor_mode = "min"

        self.fast_dev_run = 0  # default is 0
        self.max_epochs = 26 if self.platform != "Darwin" else 1  # default is -1  for infinite
        self.batch_size_train = 8 if self.platform != "Darwin" else 32
        self.batch_size_eval = 32 if self.platform != "Darwin" else 32
        self.audio_max_length = 163840  # 163840, 327680, 655360
        self.text_max_length = 512
        # self.layer_index = None

        self.seed = 0
        self.learning_rate = 1e-5
        self.dropout = 0.3
        self.weight_decay = 0.0
        self.accumulate_grad_batches = 1  # default is 1
        self.gradient_clip_val = 0  # default is None
        self.data_time = datetime.datetime.now().strftime("%m%d%H%M")
        self.logs_dir = "logs/s" + self.data_time
        self.checkpoints_dir = "checkpoints/s" + self.data_time
        self.run_name = f"s{self.data_time}"

    def get_hparams(self):
        dic = copy.deepcopy(self.__dict__)
        dic["description"] = "  "
        ##########################################################
        dic["learning_rate"] = str(dic["learning_rate"])  # tensorboard记录的lr低于1e-4都当成1e-4
        ##########################################################
        dic.pop("strategy")
        dic.pop("accelerator")
        dic.pop("platform")
        dic.pop("batch_size_eval")
        dic.pop("num_GPU")
        dic.pop("num_workers")
        dic.pop("data_time")
        dic.pop("logs_dir")
        dic.pop("checkpoints_dir")
        dic.pop("monitor")
        dic.pop("monitor_mode")
        dic.pop("fast_dev_run")
        return dic


if __name__ == "__main__":
    config = Config()
    print(config.data_time)
