import torch
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm  # 可选：用于显示进度条
import matplotlib  # 确保字体可以显示中文

# 增加上层目录到路径
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 或者其他支持中文的字体，如 'Microsoft YaHei'
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

from m1.lightning_model import LightningModel


def analyze_cmff_attention(ckpt_path, config, test_dataloader, use_gpu=True, precision="32"):
    """
    加载一个训练好的 LightningModel, 在测试集上执行推理,
    并分析、可视化 CMFF 的注意力权重。

    具体来说，它会计算 CLS 标记（token）在各个 CMFF 层中对文本 vs 音频特征
    所付出的注意力比例，并绘制这些比例在整个测试集上的分布图。

    Args:
        ckpt_path (str): 保存的模型检查点文件路径 (.ckpt 文件)。
        config (object): 用于训练模型的配置对象
                         (必须包含 LightningModel 初始化时使用的属性，
                          如 text_extractor, audio_extractor, cache_dir, dropout, num_GPU 等)。
                         它是正确实例化模型结构所必需的。
        test_dataloader (torch.utils.data.DataLoader): 测试集的数据加载器。
        use_gpu (bool): 是否使用GPU进行计算
        precision (str): 使用的精度类型，可以是 "16"、"32" 或 "64"
    """
    print(f"从以下路径加载模型: {ckpt_path}")
    # 确保模型类已正确定义或导入
    try:
        # 从检查点加载模型。如果需要，传递 config 用于初始化。
        model = LightningModel.load_from_checkpoint(ckpt_path, config=config)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("请确保 LightningModel 类定义可用且与检查点匹配。")
        return

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置模型精度
    if precision == "16":
        model = model.half()  # 转换为半精度
        print("已将模型转换为半精度 (FP16)")
    elif precision == "32":
        model = model.float()  # 转换为单精度
        print("已将模型转换为单精度 (FP32)")
    elif precision == "64":
        model = model.double()  # 转换为双精度
        print("已将模型转换为双精度 (FP64)")
    else:
        model = model.float()  # 默认使用单精度
        print(f"不支持的精度类型 '{precision}'，默认使用单精度 (FP32)")

    model.to(device)
    model.eval()  # 将模型设置为评估模式

    all_attn_weights = []
    all_labels = []
    print(f"在测试集上运行推理...")

    # --- 执行推理 ---
    with torch.no_grad():  # 在此模式下不计算梯度，节省内存和计算
        for batch in tqdm(test_dataloader, desc="推理进度"):
            # 将批次数据移动到指定设备
            batch_on_device = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    tensor_item = item.to(device)
                    batch_on_device.append(tensor_item)
                else:
                    # 保留非 Tensor 类型的数据 (例如文件名)
                    batch_on_device.append(item)

            # print("batch_on_device",batch_on_device)
            _, (_, attn_output_weights) = model(batch_on_device)

            if isinstance(attn_output_weights, torch.Tensor):
                # 形状: [batch_size, num_cmff_layers, seq_len, seq_len]
                # 或者可能是 [batch_size, num_cmff_layers, num_heads, seq_len, seq_len]
                # 如果存在 heads 维度，对其进行平均:
                if attn_output_weights.dim() == 5:
                    # 在 heads 维度上取平均 (dim=2)
                    attn_output_weights = attn_output_weights.mean(dim=2)

                all_attn_weights.append(attn_output_weights.cpu())
                # 假设标签是批次中的最后一个元素
                all_labels.append(batch[-1].cpu())
            else:
                # 处理注意力权重未被返回的情况
                print("警告：某个批次未返回注意力权重。跳过对此批次的分析。")

    if not all_attn_weights:
        print("未能收集到任何注意力权重。无法进行分析。")
        return

    # 拼接所有批次的结果
    try:
        # 过滤掉可能存在的 None 值 (如果出现错误)
        valid_attn_weights = [w for w in all_attn_weights if w is not None and w.numel() > 0]
        if not valid_attn_weights:
            print("过滤后未能收集到有效的注意力权重。无法进行分析。")
            return

        # [N_samples, num_cmff_layers, seq_len, seq_len]
        attn_weights = torch.cat(valid_attn_weights, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()
        print(f"收集到的注意力权重形状: {attn_weights.shape}")
    except Exception as e:
        print(f"拼接注意力权重时出错: {e}")
        return

    # --- 分析注意力权重 ---
    num_text_hidden_layers = model.text_extractor.config.num_hidden_layers
    num_audio_hidden_layers = model.audio_extractor.config.num_hidden_layers
    num_text_features = num_text_hidden_layers + 1
    num_audio_features = num_audio_hidden_layers + 1

    text_indices = list(range(1, 1 + num_text_features))
    audio_indices = list(range(1 + num_text_features, 1 + num_text_features + num_audio_features))

    cls_attn_to_all = attn_weights[:, :, 0, :]

    attn_to_text = cls_attn_to_all[:, :, text_indices].sum(dim=-1)
    attn_to_audio = cls_attn_to_all[:, :, audio_indices].sum(dim=-1)

    total_attn_from_cls = attn_to_text + attn_to_audio
    epsilon = 1e-9
    total_attn_from_cls = total_attn_from_cls + epsilon

    prop_attn_to_text = attn_to_text / total_attn_from_cls
    prop_attn_to_audio = attn_to_audio / total_attn_from_cls

    avg_prop_text = prop_attn_to_text.mean(dim=1).numpy()
    avg_prop_audio = prop_attn_to_audio.mean(dim=1).numpy()

    print(f"对文本的平均注意力比例 (跨层平均): 平均值={avg_prop_text.mean():.4f}, 标准差={avg_prop_text.std():.4f}")
    print(f"对音频的平均注意力比例 (跨层平均): 平均值={avg_prop_audio.mean():.4f}, 标准差={avg_prop_audio.std():.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(avg_prop_text, kde=True, bins=30)
    plt.title("平均注意力比例：流向文本特征的分布")
    plt.xlabel("CLS 对文本特征的注意力比例")
    plt.ylabel("样本数量")
    plt.xlim(0, 1)

    plt.subplot(1, 2, 2)
    sns.histplot(avg_prop_audio, kde=True, bins=30, color="orange")
    plt.title("平均注意力比例：流向音频特征的分布")
    plt.xlabel("CLS 对音频特征的注意力比例")
    plt.ylabel("样本数量")
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.suptitle("CMFF 中 CLS 标记的注意力分析 (跨层平均)", fontsize=14, y=1.03)

    base_ckpt_name = os.path.basename(ckpt_path)
    if base_ckpt_name.endswith(".ckpt"):
        base_ckpt_name = base_ckpt_name[:-5]
    plot_filename = f"cmff_attention_analysis_{base_ckpt_name}.png"
    plt.savefig(plot_filename)
    print(f"注意力分析图已保存至: {plot_filename}")
    plt.show()


from config import Config
from m1.lightning_data import LightningData

config = Config()

dm = LightningData(config)  # audio, text, multimodal
dm.prepare_data()
dm.setup(stage="test")


analyze_cmff_attention(
    "best_model.ckpt",
    config,
    dm.test_dataloader(),
    use_gpu=False,
    precision="32",
)
