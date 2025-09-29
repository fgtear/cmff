"""
return:
        text,
        text_input_ids,
        text_attention_mask,
        audio_input_values,
        audio_attention_mask,
        audio_output_attention_mask,
        label,
"""

import lightning as L
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, Wav2Vec2Processor
import typing
import pickle as pkl
import os


class DatasetMosi(torch.utils.data.Dataset):
    def __init__(self, mode: typing.Literal["train", "test", "valid"], config):
        if config.dataset == "MOSI":
            self.data_root = "datasets/MOSI"
        elif config.dataset == "MOSEI":
            self.data_root = "datasets/MOSEI"
        else:
            raise ValueError("Invalid dataset configuration.")

        # 缓存
        if os.path.exists(f"datasets/{config.dataset}/{mode}/file_name.pkl"):
            print(f"load cache {mode} data")
            self.file_name = pkl.load(open(f"datasets/{config.dataset}/{mode}/file_name.pkl", "rb"))
            self.text = pkl.load(open(f"datasets/{config.dataset}/{mode}/text.pkl", "rb"))
            self.label = pkl.load(open(f"datasets/{config.dataset}/{mode}/label.pkl", "rb"))
            self.text_input_ids = pkl.load(open(f"datasets/{config.dataset}/{mode}/text_input_ids.pkl", "rb"))
            self.text_attention_mask = pkl.load(open(f"datasets/{config.dataset}/{mode}/text_attention_mask.pkl", "rb"))
            self.audio_input_values = pkl.load(open(f"datasets/{config.dataset}/{mode}/audio_input_values.pkl", "rb"))
            self.audio_attention_mask = pkl.load(open(f"datasets/{config.dataset}/{mode}/audio_attention_mask.pkl", "rb"))
            return

        print(f"Loading {mode} data...")
        self.mode = mode
        tokenizer = AutoTokenizer.from_pretrained(
            config.text_extractor,
            clean_up_tokenization_spaces=True,
            cache_dir=config.cache_dir,
        )
        processor = Wav2Vec2Processor.from_pretrained(config.audio_extractor, cache_dir=config.cache_dir)

        df = pd.read_csv(self.data_root + "/label.csv")
        df = df[df["mode"] == self.mode].sort_values(by=["video_id", "clip_id"]).reset_index()
        # print(df)
        self.audio_wav_dic = pkl.load(open(self.data_root + f"/wav_{self.mode}.pkl", "rb"))

        self.file_name = []
        self.text = []
        self.label = []
        self.text_input_ids = []
        self.text_attention_mask = []
        self.audio_input_values = []
        self.audio_attention_mask = []
        for index, row in df.iterrows():
            file_name = f"{row['video_id']}/{row['clip_id']}.wav"
            self.file_name.append(file_name)
            self.text.append(text := row["text"].lower())
            self.label.append(row["label"])
            with torch.no_grad():
                text_token = tokenizer(
                    text,
                    return_attention_mask=True,
                    truncation=True,  # 截断
                    max_length=config.text_max_length,
                    add_special_tokens=True,  # [CLS], [SEP], <s>, etc.
                    return_tensors="pt",
                )
            text_input_ids = text_token["input_ids"].squeeze(0)
            text_attention_mask = text_token["attention_mask"].squeeze(0)
            self.text_input_ids.append(text_input_ids)
            self.text_attention_mask.append(text_attention_mask)

            wav, sr = self.audio_wav_dic[file_name]
            wav = wav[:, : config.audio_max_length]  # 截断
            wav = torch.mean(wav, dim=0, keepdim=False)  # 合并声道， [T1]
            with torch.no_grad():
                audio_token = processor(
                    wav,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                )
            audio_input_values = audio_token.input_values.squeeze(0).to(torch.float16)
            audio_attention_mask = audio_token.attention_mask.squeeze(0).to(torch.bool)
            self.audio_input_values.append(audio_input_values)
            self.audio_attention_mask.append(audio_attention_mask)

        if config.dataset == "MOSI":
            if mode == "train":
                assert len(self.label) == 1284
            elif mode == "valid":
                assert len(self.label) == 229
            elif mode == "test":
                assert len(self.label) == 686
        elif config.dataset == "MOSEI":
            if mode == "train":
                assert len(self.label) == 16326
            elif mode == "valid":
                assert len(self.label) == 1871
            elif mode == "test":
                assert len(self.label) == 4659

        # 保存数据
        if not os.path.exists(f"datasets/{config.dataset}/{mode}"):
            os.makedirs(f"datasets/{config.dataset}/{mode}")
        pkl.dump(self.file_name, open(f"datasets/{config.dataset}/{mode}/file_name.pkl", "wb"))
        pkl.dump(self.text, open(f"datasets/{config.dataset}/{mode}/text.pkl", "wb"))
        pkl.dump(self.label, open(f"datasets/{config.dataset}/{mode}/label.pkl", "wb"))
        pkl.dump(self.text_input_ids, open(f"datasets/{config.dataset}/{mode}/text_input_ids.pkl", "wb"))
        pkl.dump(self.text_attention_mask, open(f"datasets/{config.dataset}/{mode}/text_attention_mask.pkl", "wb"))
        pkl.dump(self.audio_input_values, open(f"datasets/{config.dataset}/{mode}/audio_input_values.pkl", "wb"))
        pkl.dump(self.audio_attention_mask, open(f"datasets/{config.dataset}/{mode}/audio_attention_mask.pkl", "wb"))

    def __getitem__(self, index):
        file_name = self.file_name[index]
        text = self.text[index]
        text_input_ids = self.text_input_ids[index]
        text_attention_mask = self.text_attention_mask[index]
        audio_input_values = self.audio_input_values[index]
        audio_attention_mask = self.audio_attention_mask[index]
        label = self.label[index]

        return file_name, text, text_input_ids, text_attention_mask, audio_input_values, audio_attention_mask, label

    def __len__(self):
        return len(self.label)


class LightningData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_extractor, clean_up_tokenization_spaces=True)

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        if self.config.dataset == "MOSI":
            Dataset = DatasetMosi
        elif self.config.dataset == "MOSEI":
            Dataset = DatasetMosi
        else:
            raise ValueError("Dataset must be MOSI or MOSEI")

        if stage == "fit":
            self.train_data = Dataset("train", self.config)
            self.val_data = Dataset("valid", self.config)
            self.test_data = Dataset("test", self.config)
        elif stage == "validate":
            self.val_data = Dataset("valid", self.config)
            self.test_data = Dataset("test", self.config)
        elif stage == "test" or stage == "predict":
            self.test_data = Dataset("test", self.config)
        else:
            raise ValueError("Stage must be fit, test or predict")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size_train,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_data,
                batch_size=self.config.batch_size_eval,
                shuffle=False,
                num_workers=self.config.num_workers,
                persistent_workers=True if self.config.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                self.test_data,
                batch_size=self.config.batch_size_eval,
                shuffle=False,
                num_workers=self.config.num_workers,
                persistent_workers=True if self.config.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config.batch_size_eval,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        file_name, text, text_input_ids, text_attention_mask, audio_input_values, audio_attention_mask, label = zip(*batch)

        label = torch.tensor(label)
        text_input_ids = torch.nn.utils.rnn.pad_sequence(text_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # 生成掩码。有效位置为 1，填充位置为 0
        text_attention_mask = torch.nn.utils.rnn.pad_sequence(text_attention_mask, batch_first=True, padding_value=0).bool()
        text_length = text_attention_mask.sum(1)

        audio_input_values = torch.nn.utils.rnn.pad_sequence(audio_input_values, batch_first=True, padding_value=0)
        audio_attention_mask = torch.nn.utils.rnn.pad_sequence(audio_attention_mask, batch_first=True, padding_value=0).bool()
        audio_output_total_length = torch.tensor(audio_attention_mask.size(1) / 320).ceil()
        audio_output_valid_length = audio_attention_mask.sum(1) / 320
        audio_output_total_length = audio_output_total_length.ceil().int() - 1
        audio_output_valid_length = audio_output_valid_length.ceil().int() - 1

        audio_output_attention_mask = [[1] * i + [0] * (audio_output_total_length.item() - i) for i in audio_output_valid_length]
        audio_output_attention_mask = torch.tensor(audio_output_attention_mask).bool()
        audio_output_attention_mask = ~audio_output_attention_mask.bool()

        return (
            file_name,
            text,
            text_input_ids,
            text_attention_mask,
            text_length,
            audio_input_values,
            audio_attention_mask,
            audio_output_attention_mask,
            label,  # label要放在最后，后面计算batch_size时需要
        )


if __name__ == "__main__":
    # import shutil

    # shutil.rmtree("data/MOSI/train")
    # shutil.rmtree("data/MOSI/valid")
    # shutil.rmtree("data/MOSI/test")
    # shutil.rmtree("data/MOSEI/train")
    # shutil.rmtree("data/MOSEI/valid")
    # shutil.rmtree("data/MOSEI/test")

    from config import Config

    config = Config()

    dm = LightningData(config)  # audio, text, multimodal
    dm.prepare_data()
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        print(batch)
        break

    # ds = DatasetMosi("valid")
    # print(ds[0])
