import pandas as pd
import torchaudio
import pickle
import pickle as pkl


def dump_wav(dataset):
    """
    读取所有的音频文件，保存为 pkl 文件
    """

    audio = {"train": {}, "valid": {}, "test": {}}

    df = pd.read_csv(f"datasets/{dataset}/label.csv")
    df = df.sort_values(by=["video_id", "clip_id"]).reset_index()
    # print(df)

    for index, row in df.iterrows():
        file_name = str(row["video_id"]) + "/" + str(row["clip_id"]) + ".wav"
        file_path = f"datasets/{dataset}/wav/{file_name}"
        audio_wave, sample_rate = torchaudio.load(file_path)
        audio[row["mode"]][file_name] = [audio_wave, sample_rate]

    with open(f"datasets/{dataset}/wav.pkl", "wb") as f:
        pickle.dump(audio, f)


if __name__ == "__main__":
    dump_wav("MOSI")
    data = pkl.load(open("datasets/MOSI/wav.pkl", "rb"))
    data_train = data["train"]
    data_valid = data["valid"]
    data_test = data["test"]
    with open("datasets/MOSI/wav_train.pkl", "wb") as f:
        pkl.dump(data_train, f)
    with open("datasets/MOSI/wav_valid.pkl", "wb") as f:
        pkl.dump(data_valid, f)
    with open("datasets/MOSI/wav_test.pkl", "wb") as f:
        pkl.dump(data_test, f)

    # dump_wav("MOSEI")
    # data = pkl.load(open("data/MOSEI/wav.pkl", "rb"))
    # data_train = data["train"]
    # data_valid = data["valid"]
    # data_test = data["test"]
    # with open("data/MOSEI/wav_train.pkl", "wb") as f:
    #     pkl.dump(data_train, f)
    # with open("data/MOSEI/wav_valid.pkl", "wb") as f:
    #     pkl.dump(data_valid, f)
    # with open("data/MOSEI/wav_test.pkl", "wb") as f:
    #     pkl.dump(data_test, f)
