import torch
import torch.nn as nn
import random
import numpy
import os
import math




def seed_all(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    if torch.cuda.is_available():  # GPU
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # python 全局
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class ThreePhaseLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(0.05 * total_steps)
        self.maintain_steps = int(0.55 * total_steps)
        self.cosine_steps = total_steps - self.warmup_steps - self.maintain_steps
        super(ThreePhaseLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm up phase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif self.last_epoch < self.warmup_steps + self.maintain_steps:
            # Maintain phase
            return self.base_lrs

        else:
            # Cosine decay phase
            cosine_epoch = self.last_epoch - self.warmup_steps - self.maintain_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / self.cosine_steps))
            return [base_lr * cosine_factor for base_lr in self.base_lrs]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.normalized_shape = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 计算均方根
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        # 进行归一化
        x_normalized = x / (rms + self.eps)
        # 应用缩放参数
        return self.scale * x_normalized


# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         # `torch.rsqrt` 是开平方并取倒数
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         # 没有 `bias`
#         return output * self.weight


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model

    def precompute_freqs_cis(self, dim: int, end: int, constant: float = 10000.0):
        """
        计算cos和sin的值，cos值在实部，sin值在虚部，类似于 cosx+j*sinx
        :param dim: q,k,v的最后一维，一般为emb_dim/head_num
        :param end: 句长length
        :param constant： 这里指10000
        :return:
        复数计算 torch.polar(a, t)输出， a*(cos(t)+j*sin(t))
        """
        # freqs: 计算 1/(10000^(2i/d) )，将结果作为参数theta
        # 形式化为 [theta_0, theta_1, ..., theta_(d/2-1)]
        freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [d/2]

        # 计算m
        t = torch.arange(end, device=freqs.device)  # [length]
        # 计算m*theta
        freqs = torch.outer(t, freqs).float()  # [length, d/2]
        # freqs形式化为 [m*theta_0, m*theta_1, ..., m*theta_(d/2-1)],其中 m=0,1,...,length-1

        # 计算cos(m*theta)+j*sin(m*theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        # freqs_cis: [cos(m*theta_0)+j*sin(m*theta_0),  cos(m*theta_1)+j*sin(m*theta_1),), ..., cos(m*theta_(d/2-1))+j*sin(m*theta_(d/2-1))]
        # 其中j为虚数单位， m=0,1,...,length-1
        return freqs_cis  # [length, d/2]

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # (1, length, 1, d/2)
        return freqs_cis.view(*shape)  # [1, length, 1, d/2]

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # 先将xq维度变为[bs, length, head,  d/2, 2], 利用torch.view_as_complex转变为复数
        # xq:[q0, q1, .., q(d-1)] 转变为 xq_: [q0+j*q1, q2+j*q3, ..., q(d-2)+j*q(d-1)]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [bs, length, head, d/2]
        # 同样的，xk_:[k0+j*k1, k2+j*k3, ..., k(d-2)+j*k(d-1)]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)  # [1, length, 1, d/2]
        # 下式xq_ * freqs_cis形式化输出，以第一个为例, 如下
        # (q0+j*q1)(cos(m*theta_0)+j*sin(m*theta_0)) = q0*cos(m*theta_0)-q1*sin(m*theta_0) + j*(q1*cos(m*theta_0)+q0*sin(m*theta_0))
        # 上式的实部为q0*cos(m*theta_0)-q1*sin(m*theta_0)，虚部为q1*cos(m*theta_0)+q0*sin(m*theta_0)
        # 然后通过torch.view_as_real函数，取出实部和虚部，维度由[bs, length, head, d/2]变为[bs, length, head, d/2, 2]，最后一维放实部与虚部
        # 最后经flatten函数将维度拉平，即[bs, length, head, d]
        # 此时xq_out形式化为 [实部0，虚部0，实部1，虚部1，..., 实部(d/2-1), 虚部(d/2-1)]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [bs, length, head, d]
        # 即为新生成的q

        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, xq, xk):
        # xq, xk: [bs, length, head, d]
        # xq, xk = xq.float(), xk.float()
        freqs_cis = self.precompute_freqs_cis(dim=self.d_model, end=xq.shape[1])
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)
        return xq, xk


class SinusoidalPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        # >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [d_model/2, ]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x, batch_first=True):
        if batch_first:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, : x.size(0), :].transpose(0, 1)
        return self.dropout(x)


class LearnabledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnabledPositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding)
        pos = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0)
        pos = pos.repeat(x.size(0), 1).to(x.device)
        embedding = self.embedding(pos)
        return x + embedding


def sequence_mean(output: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    计算序列的平均值，只考虑每个序列中的有效元素。

    参数:
        output: 形状为 [batch_size, sequence_length, embedding_size] 的张量
        sequence_lengths: 形状为 [batch_size] 的张量，表示每个序列的有效长度，如 [5,8,3,7,6]

    返回:
        形状为 [batch_size, embedding_size] 的张量，表示每个序列的有效平均值

    示例:
        >>> output = torch.randn(3, 10, 5)  # 3个样本，每个序列长度10，特征维度5
        >>> sequence_lengths = torch.tensor([5, 8, 3])  # 3个样本的有效长度
        >>> result = sequence_mean(output, sequence_lengths)  # 形状为 [3, 5]
    """
    # 检查输入维度是否正确
    if output.dim() != 3:
        raise ValueError(f"期望输出张量为3维，但得到了{output.dim()}维")
    if sequence_lengths.dim() != 1:
        raise ValueError(f"期望 sequence_lengths 为1维，但得到了{sequence_lengths.dim()}维")
    if output.size(0) != sequence_lengths.size(0):
        raise ValueError(f"batch_size 不匹配: {output.size(0)} vs {sequence_lengths.size(0)}")

    # 获取输出的形状
    bs, length, embedding = output.size()

    # 创建一个长度掩码矩阵 (更简洁的方式)
    mask = (torch.arange(length, device=output.device)[None, :] < sequence_lengths[:, None]).unsqueeze(-1)

    # 应用掩码并计算平均值
    masked_output = output * mask

    # 安全地计算平均值，防止除零错误
    sum_output = masked_output.sum(dim=1)
    # 确保长度至少为1，避免除零错误
    safe_lengths = torch.clamp(sequence_lengths.float(), min=1.0).unsqueeze(-1)

    return sum_output / safe_lengths


def save_features_for_tsne(features: torch.Tensor, labels: torch.Tensor, file_path: str):
    """
    将特征向量保存到文本文件中，用于后续t-SNE可视化。

    参数:
        features: 形状为 [batch_size, embedding_dim] 的特征张量
        labels: 可选，形状为 [batch_size] 的标签张量
        file_path: 保存特征的文件路径

    示例:
        >>> features = torch.randn(100, 768)  # 100个样本，每个特征维度为768
        >>> save_features_for_tsne(features, "features_for_tsne.txt")

        # 带标签保存
        >>> labels = torch.randint(0, 10, (100,))  # 100个样本的类别标签
        >>> save_features_for_tsne(features, "labeled_features.txt", labels)
    """
    # 检查输入维度是否正确
    if features.dim() != 2:
        raise ValueError(f"期望特征张量为2维 [batch_size, embedding_dim]，但得到了{features.dim()}维")

    # 转换为numpy数组
    features_np = features.detach().cpu().numpy()

    with open(file_path, "a") as f:
        # 写入头部信息
        f.write(f"# 样本数量: {features_np.shape[0]}, 特征维度: {features_np.shape[1]}\n")

        # 如果有标签，一起保存
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            for i, (feature, label) in enumerate(zip(features_np, labels_np)):
                # 写入样本ID、标签和特征向量
                feature_str = " ".join([f"{value:.6f}" for value in feature])
                f.write(f"{i}\t{int(label)}\t{feature_str}\n")
        else:
            # 没有标签，只保存ID和特征向量
            for i, feature in enumerate(features_np):
                feature_str = " ".join([f"{value:.6f}" for value in feature])
                f.write(f"{i}\t{feature_str}\n")

    print(f"特征已保存到 {file_path}，可用于t-SNE可视化")


def visualize_tsne(
    file_path: str, perplexity: int = 30, n_components: int = 2, random_state: int = 42, title: str = "tSNE Visualization"
):
    """
    读取保存的特征文件，使用t-SNE进行降维可视化。

    参数:
        file_path: 特征文件路径
        perplexity: t-SNE算法的perplexity参数，默认为30
        n_components: 降维后的维度，默认为2
        random_state: 随机种子，默认为42
        title: 图表标题，默认为'tSNE Visualization'

    返回:
        None，但会显示或保存t-SNE可视化图
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # 读取特征文件
    features = []
    labels = []
    has_labels = False

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):  # 跳过注释行
                continue

            parts = line.strip().split("\t")
            if len(parts) >= 3:  # 有标签的格式: id label features
                has_labels = True
                label = int(parts[1])
                feature_values = parts[2].split()
                labels.append(label)
            else:  # 无标签的格式: id features
                feature_values = parts[1].split()

            # 转换为浮点数
            feature = np.array([float(val) for val in feature_values])
            features.append(feature)

    # 将列表转换为numpy数组
    features = np.array(features)
    if has_labels:
        labels = np.array(labels)

    print(f"读取了 {features.shape[0]} 个样本，每个样本的特征维度为 {features.shape[1]}")

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, n_iter=1000)
    embeddings = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))

    if has_labels:
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # 为每个类别绘制散点图
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], color=colors[i], label=f"Class {label}", alpha=0.7, s=50)
        plt.legend()
    else:
        # 无标签情况下的可视化
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7, s=50)

    plt.title(title)
    plt.xlabel("tSNE维度1")
    plt.ylabel("tSNE维度2")
    plt.tight_layout()

    # 保存图像
    save_path = file_path.replace(".txt", "_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"t-SNE可视化已保存至 {save_path}")

    # 显示图像
    plt.show()
