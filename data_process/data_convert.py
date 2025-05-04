"""
将数据集中的wav.pkl文件分为train.pkl、valid.pkl和test.pkl三个文件
"""

import pickle as pkl


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


data = pkl.load(open("data/MOSI/wav.pkl", "rb"))
data_train = data["train"]
data_valid = data["valid"]
data_test = data["test"]


with open("data/MOSI/wav_train.pkl", "wb") as f:
    pkl.dump(data_train, f)
with open("data/MOSI/wav_valid.pkl", "wb") as f:
    pkl.dump(data_valid, f)
with open("data/MOSI/wav_test.pkl", "wb") as f:
    pkl.dump(data_test, f)
