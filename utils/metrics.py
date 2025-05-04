import torch
import numpy as np
from torchmetrics import Metric, Accuracy, F1Score, MeanAbsoluteError, PearsonCorrCoef


class MOSIMetrics(Metric):
    def __init__(self):
        super().__init__()
        # 回归指标
        self.mae = MeanAbsoluteError()
        self.corr = PearsonCorrCoef()

        # 多分类准确率
        self.accuracy_7 = Accuracy(task="multiclass", num_classes=7, average="micro")
        self.accuracy_5 = Accuracy(task="multiclass", num_classes=5, average="micro")
        self.accuracy_3 = Accuracy(task="multiclass", num_classes=3, average="micro")
        self.accuracy_2_non0 = Accuracy(task="multiclass", num_classes=2, average="micro")
        self.accuracy_2_has0 = Accuracy(task="multiclass", num_classes=2, average="micro")

        self.f1_score_2_non0 = F1Score(task="multiclass", num_classes=2, average="weighted")
        self.f1_score_2_has0 = F1Score(task="multiclass", num_classes=2, average="weighted")

    def update(self, y_pred, y_true):
        """
        更新指标的状态
        Args:
            y_pred: 预测值，形状为 [batch_size]
            y_true: 真实值，形状为 [batch_size]
        """
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 创建深度副本以避免推理模式的张量问题
        y_pred_copy = y_pred.clone().detach()
        y_true_copy = y_true.clone().detach()

        # 计算MAE和相关系数 (使用副本)
        mae = self.mae(y_pred_copy, y_true_copy)

        # 特殊处理相关系数计算，使用numpy避免推理模式的限制
        if y_pred.numel() > 1:  # 至少需要两个元素来计算相关系数
            # 转换为numpy计算相关系数
            pred_np = y_pred_copy.cpu().float().numpy()
            true_np = y_true_copy.cpu().float().numpy()
            corr_value = np.corrcoef(pred_np, true_np)[0, 1] if not np.isnan(pred_np).any() and not np.isnan(true_np).any() else 0.0
            corr = torch.tensor(corr_value)
        else:
            corr = torch.tensor(0.0)

        # 多分类评估 (7分类) - 创建副本避免原地修改
        y_pred_a7 = torch.clamp(y_pred.clone(), min=-3.0, max=3.0)
        y_true_a7 = torch.clamp(y_true.clone(), min=-3.0, max=3.0)
        y_pred_a7 = torch.round(y_pred_a7) + 3  # 转换为0-6的类别
        y_true_a7 = torch.round(y_true_a7) + 3
        acc_7 = self.accuracy_7(y_pred_a7, y_true_a7)

        # 多分类评估 (5分类) - 创建副本避免原地修改
        y_pred_a5 = torch.clamp(y_pred.clone(), min=-2.0, max=2.0)
        y_true_a5 = torch.clamp(y_true.clone(), min=-2.0, max=2.0)
        y_pred_a5 = torch.round(y_pred_a5) + 2  # 转换为0-4的类别
        y_true_a5 = torch.round(y_true_a5) + 2
        acc_5 = self.accuracy_5(y_pred_a5, y_true_a5)

        # 二分类评估 (有0) - 使用比较操作生成新张量
        binary_preds = (y_pred >= 0).long()
        binary_truth = (y_true >= 0).long()
        acc_2_has0 = self.accuracy_2_has0(binary_preds, binary_truth)
        f1_score_has0 = self.f1_score_2_has0(binary_preds, binary_truth)

        # 二分类评估 (无0)
        non_zeros = torch.where(y_true != 0)[0]
        if len(non_zeros) > 0:
            non_zeros_binary_preds = (y_pred[non_zeros] > 0).long()
            non_zeros_binary_truth = (y_true[non_zeros] > 0).long()
            acc_2_non0 = self.accuracy_2_non0(non_zeros_binary_preds, non_zeros_binary_truth)
            f1_score_non0 = self.f1_score_2_non0(non_zeros_binary_preds, non_zeros_binary_truth)
        else:
            acc_2_non0 = torch.tensor(0.0)
            f1_score_non0 = torch.tensor(0.0)

        self.mae.update(y_pred, y_true)
        self.corr.update(y_pred, y_true)
        self.accuracy_7.update(y_pred_a7, y_true_a7)
        self.accuracy_5.update(y_pred_a5, y_true_a5)
        self.accuracy_2_non0.update(non_zeros_binary_preds, non_zeros_binary_truth)
        self.accuracy_2_has0.update(binary_preds, binary_truth)
        self.f1_score_2_non0.update(non_zeros_binary_preds, non_zeros_binary_truth)
        self.f1_score_2_has0.update(binary_preds, binary_truth)

    def forward(self, y_pred, y_true):
        """
        评估MOSI回归任务的性能

        Args:
            y_pred: 预测值，形状为 [batch_size]
            y_true: 真实值，形状为 [batch_size]

        Returns:
            dict: 包含各种评估指标的字典
        """

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 创建深度副本以避免推理模式的张量问题
        y_pred_copy = y_pred.clone().detach()
        y_true_copy = y_true.clone().detach()

        # 计算MAE和相关系数 (使用副本)
        mae = self.mae(y_pred_copy, y_true_copy)

        # 特殊处理相关系数计算，使用numpy避免推理模式的限制
        if y_pred.numel() > 1:  # 至少需要两个元素来计算相关系数
            # 转换为numpy计算相关系数
            pred_np = y_pred_copy.cpu().float().numpy()
            true_np = y_true_copy.cpu().float().numpy()
            corr_value = np.corrcoef(pred_np, true_np)[0, 1] if not np.isnan(pred_np).any() and not np.isnan(true_np).any() else 0.0
            corr = torch.tensor(corr_value)
        else:
            corr = torch.tensor(0.0)

        # 多分类评估 (7分类) - 创建副本避免原地修改
        y_pred_a7 = torch.clamp(y_pred.clone(), min=-3.0, max=3.0)
        y_true_a7 = torch.clamp(y_true.clone(), min=-3.0, max=3.0)
        y_pred_a7 = torch.round(y_pred_a7) + 3  # 转换为0-6的类别
        y_true_a7 = torch.round(y_true_a7) + 3
        acc_7 = self.accuracy_7(y_pred_a7, y_true_a7)

        # 多分类评估 (5分类) - 创建副本避免原地修改
        y_pred_a5 = torch.clamp(y_pred.clone(), min=-2.0, max=2.0)
        y_true_a5 = torch.clamp(y_true.clone(), min=-2.0, max=2.0)
        y_pred_a5 = torch.round(y_pred_a5) + 2  # 转换为0-4的类别
        y_true_a5 = torch.round(y_true_a5) + 2
        acc_5 = self.accuracy_5(y_pred_a5, y_true_a5)

        # 二分类评估 (有0) - 使用比较操作生成新张量
        binary_preds = (y_pred >= 0).long()
        binary_truth = (y_true >= 0).long()
        acc_2_has0 = self.accuracy_2_has0(binary_preds, binary_truth)
        f1_score_has0 = self.f1_score_2_has0(binary_preds, binary_truth)

        # 二分类评估 (无0)
        non_zeros = torch.where(y_true != 0)[0]
        if len(non_zeros) > 0:
            non_zeros_binary_preds = (y_pred[non_zeros] > 0).long()
            non_zeros_binary_truth = (y_true[non_zeros] > 0).long()
            acc_2_non0 = self.accuracy_2_non0(non_zeros_binary_preds, non_zeros_binary_truth)
            f1_score_non0 = self.f1_score_2_non0(non_zeros_binary_preds, non_zeros_binary_truth)

        else:
            acc_2_non0 = torch.tensor(0.0)
            f1_score_non0 = torch.tensor(0.0)

        self.mae.update(y_pred, y_true)
        self.corr.update(y_pred, y_true)
        self.accuracy_7.update(y_pred_a7, y_true_a7)
        self.accuracy_5.update(y_pred_a5, y_true_a5)
        self.accuracy_2_non0.update(non_zeros_binary_preds, non_zeros_binary_truth)
        self.accuracy_2_has0.update(binary_preds, binary_truth)
        self.f1_score_2_non0.update(non_zeros_binary_preds, non_zeros_binary_truth)
        self.f1_score_2_has0.update(binary_preds, binary_truth)

        return {
            "Has0_acc_2": round(acc_2_has0.item(), 4),
            "Has0_F1_score": round(f1_score_has0.item(), 4),
            "Non0_acc_2": round(acc_2_non0.item(), 4),
            "Non0_F1_score": round(f1_score_non0.item(), 4),
            # "Mult_acc_5": round(acc_5.item(), 4),
            "Mult_acc_7": round(acc_7.item(), 4),
            "MAE": round(mae.item(), 4),
            "Corr": round(float(corr), 4),
        }

    def compute(self):
        """
        计算所有指标
        """
        return {
            "Has0_acc_2": self.accuracy_2_has0.compute(),
            "Has0_F1_score": self.f1_score_2_has0.compute(),
            "Non0_acc_2": self.accuracy_2_non0.compute(),
            "Non0_F1_score": self.f1_score_2_non0.compute(),
            # "Mult_acc_5": self.accuracy_5.compute(),
            "Mult_acc_7": self.accuracy_7.compute(),
            "MAE": self.mae.compute(),
            "Corr": self.corr.compute(),
        }

    def reset(self):
        """
        重置指标状态
        """
        self.mae.reset()
        self.corr.reset()
        self.accuracy_7.reset()
        self.accuracy_5.reset()
        self.accuracy_2_non0.reset()
        self.accuracy_2_has0.reset()
        self.f1_score_2_non0.reset()
        self.f1_score_2_has0.reset()
