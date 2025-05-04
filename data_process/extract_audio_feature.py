import pandas as pd
import torchaudio
import pickle
import torch
from transformers import Wav2Vec2Processor, Data2VecAudioModel
import tqdm

MODEL = "facebook/data2vec-audio-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL)
model = Data2VecAudioModel.from_pretrained(MODEL)


def dump_wav(dataset):
    """
    读取所有的音频文件，保存为 pkl 文件
    """

    audio = {"train": {}, "valid": {}, "test": {}}

    df = pd.read_csv(dataset + "/label.csv")
    df = df.sort_values(by=["video_id", "clip_id"]).reset_index()
    # print(df)

    for index, row in df.iterrows():
        file_name = str(row["video_id"]) + "/" + str(row["clip_id"]) + ".wav"
        file_path = dataset + "/wav/" + file_name
        audio_wave, sample_rate = torchaudio.load(file_path)
        audio[row["mode"]][file_name] = [audio_wave, sample_rate]

    with open(dataset + "/wav.pkl", "wb") as f:
        pickle.dump(audio, f)


def extract_wav_feature(audio_pkl_path):
    audio = pickle.load(open(audio_pkl_path, "rb"))
    for mode in ["train", "valid", "test"]:
        for file_name in tqdm.tqdm(audio[mode]):
            audio_wave, sample_rate = audio[mode][file_name]
            audio_wave = torch.mean(audio_wave, dim=0, keepdim=False)  # 合并声道， [T1]
            with torch.no_grad():  # 使用 no_grad 避免计算梯度占内存
                output = processor(
                    audio_wave,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                )
                input_values = output.input_values  # [1, T1]
                attention_mask = output.attention_mask  # [1, T1]on_mask  # [1, T1]，这里batch_size=1，所有都是1，[1, 1, ..., 1]
                last_hidden_state = model(input_values, attention_mask=attention_mask).last_hidden_state  # [1, T2, Dim]
                last_hidden_state = last_hidden_state.squeeze(0)  # [T2, Dim]
            audio[mode][file_name] = last_hidden_state
        # torch.cuda.empty_cache()  # 每个 mode 处理完释放 GPU 内存
    audio_pkl_path = audio_pkl_path.replace(".pkl", "_feature.pkl")
    with open(audio_pkl_path, "wb") as f:
        pickle.dump(audio, f)


if __name__ == "__main__":
    dump_wav("MOSI")
    # extract_wav_feature("MOSI/wav.pkl")
    pass
    dump_wav("MOSEI")
    # extract_wav_feature("MOSEI/wav.pkl")
