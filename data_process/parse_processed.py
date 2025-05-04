import pickle as pkl
import numpy as np


def parse_processed(file_path):
    print("".center(50, "-"))
    # Load the processed data
    data = pkl.load(open(file_path, "rb"))
    print(data.keys())

    data_train = data["train"]
    print(data_train.keys())

    vision_train = data_train["vision"]
    vision_train = np.array(vision_train)  # (num_samples, 500, 20)
    print("vision_train shape", vision_train.shape)

    audio_train = data_train["audio"]
    audio_train = np.array(audio_train)  # (num_samples, 375, 5)
    print("audio_train shape", audio_train.shape)

    raw_text_train = data_train["raw_text"]
    print("raw_text_train", raw_text_train)

    text = data_train["text"]
    text = np.array(text)
    print("text shape", text.shape)

    text_bert = data_train["text_bert"]
    text_bert = np.array(text_bert)  # (num_samples, 3, 50), 3 include: word, position, and segment, 50 is the max_length
    print("text_bert shape", text_bert.shape)

    regression_labels = data_train["regression_labels"]
    print("regression_labels", regression_labels)


# unaligned = parse_processed("MOSI/Processed/unaligned_50.pkl")
unaligned = parse_processed("MOSI/Processed/aligned_50.pkl")
# unaligned = parse_processed("MOSEI/Processed/unaligned_50.pkl")
unaligned = parse_processed("MOSEI/Processed/aligned_50.pkl")