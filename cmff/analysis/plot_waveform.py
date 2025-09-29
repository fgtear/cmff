import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def plot_waveform(wav_file):
    # 读取 WAV 文件
    sample_rate, data = wavfile.read(wav_file)

    # 如果是立体声，取第一个声道
    if len(data.shape) > 1:
        data = data[:, 0]

    # 创建时间轴
    times = np.arange(len(data)) / float(sample_rate)

    # 绘制波形图
    plt.figure(figsize=(15, 5))
    plt.plot(times, data)
    plt.title("波形图 - {}".format(wav_file))
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.xlim([0, times[-1]])
    plt.show()


plot_waveform("data/MOSI/wav/03bSnISJMiM/11.wav")
