import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoModel, RobertaModel
import transformers

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import sequence_mean
from utils.metrics import MOSIMetrics
from cmff.analysis.visualization import tsne_visualization
from thop import profile


class LightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.get_hparams())
        self.val1_metrics = MOSIMetrics()
        self.val2_metrics = MOSIMetrics()
        self.test_metrics = MOSIMetrics()

        self.test_prediction = []
        self.test_label = []
        self.test_feature = []
        self.test_attn_output_weights = []

        self.text_extractor = AutoModel.from_pretrained(
            config.text_extractor,
            output_hidden_states=True,
            add_pooling_layer=False,
        )
        self.audio_extractor = AutoModel.from_pretrained(
            config.audio_extractor,
            output_hidden_states=True,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.audio_extractor.config.hidden_size, self.text_extractor.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.text_extractor.config.hidden_size, self.text_extractor.config.hidden_size),
        )
        self.cmff = CMFF(num_layers=3, text_hidden_size=self.text_extractor.config.hidden_size, dropout=config.dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.text_extractor.config.hidden_size, self.text_extractor.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.text_extractor.config.hidden_size, 8),
        )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.text_extractor.config.hidden_size))
        # nn.init.xavier_normal_(self.cls_token)
        # self.encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=self.text_extractor.config.hidden_size,
        #         nhead=8,
        #         batch_first=True,
        #     ),
        #     num_layers=3,
        # )
        # self.fc_2 = nn.Sequential(
        #     nn.Linear(self.text_extractor.config.hidden_size * 2, self.text_extractor.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(self.text_extractor.config.hidden_size, 8),
        # )
        if config.platform == "Darwin":
            return
        #####################################################################################
        # text_pretrained = torch.load(
        #     "cmff/mlruns/0/0f149096a2cb4f7b8791b204b4d6334a/artifacts/checkpoints/best_model.ckpt",
        #     map_location=self.device,
        # )
        # for name, param in self.text_extractor.named_parameters():
        #     name = "text_extractor." + name
        #     if name in text_pretrained["state_dict"]:
        #         # print("load text", name)
        #         param.data = text_pretrained["state_dict"][name]
        #         # if "attention" not in name:
        #         # param.requires_grad = False
        # del text_pretrained
        # #####################################################################################
        # audio_pretrained = torch.load(
        #     "cmff/mlruns/0/791f0a2d7d7a45ef9e7942bd32c4fdaa/artifacts/checkpoints/best_model.ckpt",
        #     map_location=self.device,
        # )
        # for name, param in self.audio_extractor.named_parameters():
        #     name = "audio_extractor." + name
        #     if name in audio_pretrained["state_dict"]:
        #         # print("load audio", name)
        #         param.data = audio_pretrained["state_dict"][name]
        #         # if "attention" not in name:
        #         # param.requires_grad = False
        # del audio_pretrained
        #####################################################################################
        # del self.text_extractor
        # del self.audio_extractor

    def forward(self, batch):
        (
            file_name,
            text,
            text_input_ids,
            text_attention_mask,
            text_length,
            audio_input_values,
            audio_attention_mask,
            audio_output_attention_mask,
            label,
        ) = batch
        ###############################################################################################################
        # audio使用最后一层的mean作为特征
        # audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        # hidden_states = audio_output.hidden_states
        # audio_feature = hidden_states[-1]
        # del audio_output, hidden_states
        # audio_feature = sequence_mean(audio_feature, audio_output_attention_mask.sum(1))
        # audio_feature = self.projector(audio_feature)
        # logits = self.fc(audio_feature)
        # return logits, (audio_feature, 0)
        ###############################################################################################################
        # audio 使用 cmff 编码多层
        # audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        # hidden_states = audio_output.hidden_states
        # del audio_output
        # audio_feature = [sequence_mean(i, audio_output_attention_mask.sum(1)) for i in hidden_states]
        # audio_feature = torch.stack(audio_feature, dim=1)
        # audio_feature = self.projector(audio_feature)
        # audio_feature, attn_output_weights = self.cmff(text_feature=None, audio_feature=audio_feature)
        # logits = self.fc(audio_feature)
        # return logits, (audio_feature, attn_output_weights)
        ###############################################################################################################
        # text仅使用最后一层的特征
        # text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        # hidden_states = text_output.hidden_states
        # text_feature = hidden_states[-1][:, 0, :]  # 使用cls作为特征
        # del text_output, hidden_states
        # # text_feature = sequence_mean( hidden_states[-1], text_length)  # 使用mean作为特征
        # logits = self.fc(text_feature)
        # return logits, (text_feature, 0)
        ###############################################################################################################
        # text 使用 cmff 编码多层cls
        # text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        # hidden_states = text_output.hidden_states
        # text_feature = [i[:, 0, :] for i in hidden_states]
        # text_feature = torch.stack(text_feature, dim=1)
        # text_feature, attn_output_weights = self.cmff(text_feature=text_feature, audio_feature=None)
        # logits = self.fc(text_feature)
        # return logits, (text_feature, attn_output_weights)
        ###############################################################################################################
        # text + audio 使用cmff编码多层cls
        text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        text_feature = [i[:, 0, :] for i in text_output.hidden_states]
        del text_output
        text_feature = torch.stack(text_feature, dim=1)
        audio_input_values = audio_input_values.to(torch.float32)  # TODO:
        audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        audio_feature = [sequence_mean(i, audio_output_attention_mask.sum(1)) for i in audio_output.hidden_states]
        del audio_output
        audio_feature = torch.stack(audio_feature, dim=1)
        audio_feature = self.projector(audio_feature)
        feature, attn_output_weights = self.cmff(text_feature=text_feature, audio_feature=audio_feature)
        logit = self.fc(feature)
        return logit, (feature, attn_output_weights)
        ###############################################################################################################
        #  对比实验，直接cat last然后预测
        # text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        # hidden_states = text_output.hidden_states
        # text_feature = hidden_states[-1][:, 0, :]  # 使用cls作为特征
        # del text_output, hidden_states
        # audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        # hidden_states = audio_output.hidden_states
        # audio_feature = hidden_states[-1]
        # del audio_output, hidden_states
        # audio_feature = sequence_mean(audio_feature, audio_output_attention_mask.sum(1))
        # audio_feature = self.projector(audio_feature)
        # feature = torch.cat([text_feature, audio_feature], dim=1)
        # logits = self.fc_2(feature)
        # return logits, (feature, 0)
        ###############################################################################################################
        #  对比实验，使用多层的mean然后cat然后预测
        # audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        # hidden_states = audio_output.hidden_states
        # del audio_output
        # audio_feature = [sequence_mean(i, audio_output_attention_mask.sum(1)) for i in hidden_states]
        # audio_feature = torch.stack(audio_feature, dim=1)
        # audio_feature = torch.mean(audio_feature, dim=1)
        # text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        # hidden_states = text_output.hidden_states
        # text_feature = [i[:, 0, :] for i in hidden_states]
        # text_feature = torch.stack(text_feature, dim=1)
        # text_feature = torch.mean(text_feature, dim=1)
        # feature = torch.cat([text_feature, audio_feature], dim=1)
        # logits = self.fc_2(feature)
        # return logits, (feature, 0)
        ###############################################################################################################
        # 对比实验，使用transformer encoder替代 cmff 来编码多层c
        text_output = self.text_extractor(text_input_ids, attention_mask=text_attention_mask)
        text_feature = [i[:, 0, :] for i in text_output.hidden_states]
        del text_output
        text_feature = torch.stack(text_feature, dim=1)
        audio_output = self.audio_extractor(audio_input_values, attention_mask=audio_attention_mask)
        audio_feature = [sequence_mean(i, audio_output_attention_mask.sum(1)) for i in audio_output.hidden_states]
        del audio_output
        audio_feature = torch.stack(audio_feature, dim=1)
        audio_feature = self.projector(audio_feature)
        feature = torch.cat(
            [
                self.cls_token.expand(audio_feature.size(0), -1, -1),
                text_feature,
                audio_feature,
            ],
            dim=1,
        )
        feature = self.encoder(feature)
        feature = feature[:, 0, :]  # (batch_size, hidden_size)
        logits = self.fc(feature)
        return logits, (feature, 0)
        ###############################################################################################################

    def compute_loss(self, logits, label):
        #################################### l1 loss
        l1_logits = logits[:, 0].squeeze()
        l1_loss = F.l1_loss(l1_logits, label, reduction="none")
        return {"loss": l1_loss.mean()}
        hard_ratio = 0.3  # 硬样本的比例
        num_hard = max(int(l1_loss.size(0) * hard_ratio), 1)  # 至少选择1个样本
        _, hard_indices = torch.topk(l1_loss, num_hard)
        weighted = torch.ones_like(l1_loss)
        weighted[hard_indices] = weighted[hard_indices] + 2
        l1_loss = l1_loss * weighted  # easy sample loss * 1, hard sample loss * 3
        l1_loss = l1_loss.mean()
        return {"loss": l1_loss}
        #################################### cls loss
        discrete_label = torch.clamp(label.clone(), min=-3.0, max=3.0)
        discrete_label = torch.round(discrete_label) + 3
        cls_logits = logits[:, 1:]

        # 计算每个样本的交叉熵损失（不使用reduction）
        class_weights = torch.tensor([7.419, 1.784, 1.146, 0.344, 0.561, 1.504, 11.477]).to(self.device)
        individual_cls_loss = F.cross_entropy(
            cls_logits, discrete_label.long(), reduction="none", label_smoothing=0.2, weight=class_weights
        )
        # 对分类损失也应用相同的硬负样本挖掘策略
        _, cls_hard_indices = torch.topk(individual_cls_loss, num_hard)
        cls_weighted = torch.ones_like(individual_cls_loss)
        cls_weighted[cls_hard_indices] = cls_weighted[cls_hard_indices] + 2
        weighted_cls_loss = individual_cls_loss * cls_weighted
        # cls_hard_mask = torch.zeros_like(individual_cls_loss)
        # cls_hard_mask[cls_hard_indices] = hard_weight - 1.0
        # weighted_cls_loss = individual_cls_loss * (1.0 + cls_hard_mask)
        cls_loss = weighted_cls_loss.mean()
        ####################################
        loss = l1_loss + cls_loss
        losses = {
            "loss": loss,
            "l1_loss": l1_loss,
            "cls_loss": cls_loss,
        }
        return losses

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.weight_decay,
        #     betas=(0.9, 0.999),
        # )
        # scheduler = transformers.get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # lr_scheduler = {"scheduler": scheduler, "name": "learning_rate", "interval": "step", "frequency": 1}
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        params_text = []
        params_audio = []
        prrams_others = []
        for name, param in self.named_parameters():
            if "text_extractor" in name:
                params_text.append(param)
            elif "audio_extractor" in name:
                params_audio.append(param)
            else:
                prrams_others.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": params_text, "lr": 2e-5 / 4, "weight_decay": self.config.weight_decay},
                {"params": params_audio, "lr": 1e-4 / 2, "weight_decay": self.config.weight_decay},
                {"params": prrams_others, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
            ],
            betas=(0.9, 0.999),
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler = {"scheduler": scheduler, "name": "learning_rate", "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # def on_train_start(self):
    # self.logger.log_hyperparams(self.hparams, {"val/ZZZ": 0})
    # self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        # log learning rate
        # for i, param_group in enumerate(self.optimizers().param_groups):
        #     self.log(
        #         f"train/lr_{i}",
        #         param_group["lr"],
        #         on_step=False,
        #         on_epoch=True,
        #         batch_size=batch[-1].size(0),
        #         sync_dist=True if self.config.num_GPU > 1 else False,
        #     )
        logits, _ = self(batch)
        losses = self.compute_loss(logits, batch[-1])
        self.log_dict(
            {f"losses/{k}_train": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch[-1].size(0),
            sync_dist=True if self.config.num_GPU > 1 else False,
        )
        return losses

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, _ = self(batch)
        losses = self.compute_loss(logits, batch[-1])
        if dataloader_idx == 0:
            self.val1_metrics.update(logits[:, 0].squeeze(), batch[-1])
            self.log_dict(
                {f"losses/{k}_val": v for k, v in losses.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch[-1].size(0),
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )

        elif dataloader_idx == 1:
            self.val2_metrics.update(logits[:, 0].squeeze(), batch[-1])
            self.log_dict(
                {f"losses/{k}_test": v for k, v in losses.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch[-1].size(0),
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )
        else:
            raise ValueError("dataloader_idx must be 0 or 1")

    def on_validation_epoch_end(self):
        val1_metrics = self.val1_metrics.compute()
        self.val1_metrics.reset()
        val2_metrics = self.val2_metrics.compute()
        self.val2_metrics.reset()

        self.log_dict(
            {f"metrics/{k}_val": v for k, v in val1_metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.config.num_GPU > 1 else False,
            add_dataloader_idx=False,
        )
        self.log_dict(
            {f"metrics/{k}_test": v for k, v in val2_metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.config.num_GPU > 1 else False,
            add_dataloader_idx=False,
        )

    def on_test_start(self):
        self.test_prediction = []
        self.test_label = []
        self.test_feature = []
        self.test_attention_weight = []

    def test_step(self, batch):
        logits, (features, attn_output_weights) = self(batch)
        self.test_metrics.update(logits[:, 0].squeeze(), batch[-1])

        self.test_prediction.append(logits[:, 0].squeeze().detach().cpu())
        self.test_label.append(batch[-1].detach().cpu())
        self.test_feature.append(features.detach().cpu())
        if type(attn_output_weights) is not int:
            self.test_attn_output_weights.append(attn_output_weights.detach().cpu())
            print("-----", attn_output_weights.shape)  # TODO:

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(
            {"metrics/" + k: v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True if self.config.num_GPU > 1 else False,
        )

        prediction = torch.cat(self.test_prediction, dim=0).float().numpy()
        label = torch.cat(self.test_label, dim=0).float().numpy()
        feature = torch.cat(self.test_feature, dim=0).float().numpy()
        os.makedirs("cmff/temp", exist_ok=True)
        np.save(prediction_path := "cmff/temp/prediction.npy", prediction)
        np.save(label_path := "cmff/temp/label.npy", label)
        np.save(feature_path := "cmff/temp/feature.npy", feature)
        self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=prediction_path, artifact_path="")
        self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=label_path, artifact_path="")
        self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=feature_path, artifact_path="")
        os.remove(prediction_path)
        os.remove(label_path)
        os.remove(feature_path)

        if self.test_attn_output_weights != []:
            # attn_output_weights: [num_test_samples, num_layers, batch_size, seq_len, seq_len]
            attn_output_weights = torch.cat(self.test_attn_output_weights, dim=0).float().numpy()
            np.save(attn_output_weights_path := "cmff/temp/attn_output_weights.npy", attn_output_weights)
            # print("attn_output_weights", attn_output_weights.shape)  # TODO:
            self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=attn_output_weights_path, artifact_path="")
            os.remove(attn_output_weights_path)

        # tsne_path = tsne_visualization(feature, label)
        # self.logger.experiment.log_artifact(run_id=self.logger.run_id, local_path=tsne_path, artifact_path="")
        fig = tsne_visualization(feature, label)
        self.logger.experiment.log_figure(run_id=self.logger.run_id, figure=fig, artifact_file="tsne_visualization.png")

        self.test_prediction = []
        self.test_label = []
        self.test_feature = []
        self.test_attn_output_weights = []

    def backward(self, loss):
        loss.backward()

    def predict_step(self, batch):
        logits, (features, attn_output_weights) = self(batch)
        label = batch[-1]
        logits = logits[0, 0]
        with torch.no_grad():
            mae = torch.abs(logits - label)
        if mae > 0.3:
            return
        # print("attn_output_weights", attn_output_weights.shape)
        # attn_output_weights.shape = [batch_size, num_layers, seq_len, seq_len]
        # seq_len = 51, cls + 25text + 25audio
        # num_layers = 3 代表了有3层
        # batch_size = 1

        try:
            with torch.no_grad():
                # Ensure label is also a scalar tensor on the same device for comparison
                if not isinstance(label, torch.Tensor):
                    label_tensor = torch.tensor(label, device=logits.device, dtype=logits.dtype)
                else:
                    label_tensor = label.to(logits.device).squeeze()  # Ensure scalar

                if logits.shape == label_tensor.shape:
                    mae = torch.abs(logits - label_tensor)
                else:
                    print(f"Shape mismatch: Predicted {logits.shape}, Label {label_tensor.shape}")

        except Exception as e:
            print(f"Error calculating MAE: {e}")

        # --- Visualization Call ---
        # Condition for visualization (e.g., low MAE, specific sample index, etc.)
        # Using your original condition: mae <= 0.3
        # Ensure attn_output_weights is not None before calling
        if attn_output_weights is not None and mae <= 0.3:
            print(
                f"\nVisualizing attention for sample with low MAE ({mae.item():.3f}). Label: {label.item():.3f}, Pred: {logits.item():.3f}"
            )
            print("attn_output_weights shape:", attn_output_weights.shape)

            # Call the visualization function
            # Pass the correct number of text/audio layers your model uses in the sequence
            # Based on your comment seq_len=51 -> 25 text + 25 audio
            visualize_attention(attn_output_weights, num_text_layers=25, num_audio_layers=25)

        # --- Return necessary values for your evaluation loop ---
        # return {'logits': predicted_value, 'label': label, 'mae': mae} # Example return dict


class MoEExpert(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size * 4)
        self.linear2 = nn.Linear(input_size, input_size * 4)
        self.linear3 = nn.Linear(input_size * 4, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = F.gelu(self.linear1(x))
        x1 = self.dropout(x1)
        x2 = F.sigmoid(self.linear2(x))
        x = x1 * x2
        x = self.linear3(x)
        return x


class CMFFLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_att = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cls_expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.text_expert = MoEExpert(hidden_size, dropout)
        self.audio_expert = MoEExpert(hidden_size, dropout)

    def forward(self, feature, token_type_ids):
        """
        feature: (batch_size, seq_len, hidden_size)
        token_type_ids: (seq_len), 0 for cls, 1 for text, 2 for audio, torch.tensor([0, 1, 1, 2, 2])
        """
        attn_mask = torch.zeros(feature.shape[1], feature.shape[1], device=feature.device, dtype=torch.bool)
        # attn_mask[1:, 0] = True

        residual = feature
        feature = self.norm1(feature)
        feature, attn_output_weights = self.self_att(feature, feature, feature, attn_mask=attn_mask)
        feature = self.dropout1(feature)
        feature = feature + residual

        residual = feature
        feature = self.norm2(feature)
        expert_output = self.cls_expert(feature[:, 0:1, :])
        text_index = torch.where(token_type_ids == 1)[0]
        audio_index = torch.where(token_type_ids == 2)[0]
        # 判断text_index是否为空
        if text_index.shape != 0:
            text_output = self.text_expert(feature[:, text_index, :])
            expert_output = torch.cat([expert_output, text_output], dim=1)
        if audio_index.shape != 0:
            audio_output = self.audio_expert(feature[:, audio_index, :])
            expert_output = torch.cat([expert_output, audio_output], dim=1)
        feature = self.dropout2(expert_output)
        feature = feature + residual

        return feature, attn_output_weights


class CMFF(nn.Module):
    def __init__(self, num_layers, text_hidden_size, dropout):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, text_hidden_size))
        nn.init.xavier_normal_(self.cls_token)

        self.token_type_embeddings = nn.Embedding(2, text_hidden_size)
        self.position_embeddings = nn.Embedding(25, text_hidden_size)

        self.cmff_layers = nn.ModuleList([CMFFLayer(text_hidden_size, dropout=dropout) for _ in range(num_layers)])
        # self.cmff_layers = nn.ModuleList(
        #     [
        #         nn.TransformerEncoderLayer(
        #             d_model=text_hidden_size,
        #             nhead=8,
        #             dim_feedforward=text_hidden_size * 4,
        #             dropout=dropout,
        #             activation="gelu",
        #             batch_first=True,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )

    def forward(self, text_feature=None, audio_feature=None):
        """
        text_feature: (batch_size, num_layer, hidden_size)
        audio_feature: (batch_size, num_layer, hidden_size)
        """
        if text_feature is not None:
            cls = self.cls_token.expand(text_feature.shape[0], -1, -1)
        else:
            cls = self.cls_token.expand(audio_feature.shape[0], -1, -1)

        if text_feature is not None:
            text_feature += self.token_type_embeddings(
                torch.zeros(text_feature.shape[0], text_feature.shape[1], dtype=torch.long, device=text_feature.device)
            )
            text_feature += self.position_embeddings(
                torch.arange(text_feature.shape[1], device=text_feature.device).unsqueeze(0).expand(text_feature.shape[0], -1)
            )
        if audio_feature is not None:
            audio_feature += self.token_type_embeddings(
                torch.ones(audio_feature.shape[0], audio_feature.shape[1], dtype=torch.long, device=audio_feature.device)
            )
            audio_feature += self.position_embeddings(
                torch.arange(audio_feature.shape[1], device=audio_feature.device).unsqueeze(0).expand(audio_feature.shape[0], -1)
            )

        if text_feature is None and audio_feature is None:
            raise ValueError("text_feature and audio_feature cannot be None at the same time")
        elif text_feature is None and audio_feature is not None:
            feature = torch.cat([cls, audio_feature], dim=1)
            token_type_ids = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=feature.device),
                    torch.ones(audio_feature.shape[1], dtype=torch.long, device=feature.device) * 2,
                ],
                dim=0,
            )
        elif text_feature is not None and audio_feature is None:
            feature = torch.cat([cls, text_feature], dim=1)
            token_type_ids = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=feature.device),
                    torch.ones(text_feature.shape[1], dtype=torch.long, device=feature.device),
                ],
                dim=0,
            )
        else:
            feature = torch.cat([cls, text_feature, audio_feature], dim=1)
            token_type_ids = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=feature.device),
                    torch.ones(text_feature.shape[1], dtype=torch.long, device=feature.device),
                    torch.ones(audio_feature.shape[1], dtype=torch.long, device=feature.device) * 2,
                ],
                dim=0,
            )

        # if self.training:
        #     mask_prob = 0.3
        #     mask = torch.rand(feature.shape[1], device=feature.device)
        #     mask = mask > mask_prob  # 需要mask的部分为 False
        #     mask[0] = True
        #     mask[1] = True
        #     feature = feature[:, mask, :]
        #     token_type_ids = token_type_ids[:, mask]

        attn_output_weights_list = []
        for layer in self.cmff_layers:
            feature, attn_output_weights = layer(feature, token_type_ids)
            attn_output_weights_list.append(attn_output_weights)

        feature = feature[:, 0, :]
        attn_output_weights = torch.stack(attn_output_weights_list, dim=1)  # [batch_size, num_layers, seq_len, seq_len]
        return feature, attn_output_weights


import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_attention(attn_weights, num_text_layers=25, num_audio_layers=25):
    """
    Visualizes attention weights from the CMFF module.

    Args:
        attn_weights (torch.Tensor): Attention weights tensor with shape
                                     [batch_size, num_layers, seq_len, seq_len].
                                     Expected seq_len = 1 (CLS) + num_text_layers + num_audio_layers.
        num_text_layers (int): Number of text layer features included in the sequence.
        num_audio_layers (int): Number of audio layer features included in the sequence.
    """
    if attn_weights is None or not isinstance(attn_weights, torch.Tensor):
        print("Invalid attention weights provided.")
        return

    if attn_weights.dim() != 4:
        raise ValueError(f"Expected attn_weights to be a 4D tensor, but got {attn_weights.dim()}D")

    # --- Configuration ---
    batch_size, num_layers, seq_len, _ = attn_weights.shape
    expected_seq_len = 1 + num_text_layers + num_audio_layers

    if seq_len != expected_seq_len:
        print(
            f"Warning: Sequence length ({seq_len}) doesn't match expected "
            f"({expected_seq_len} = 1 CLS + {num_text_layers} Text + {num_audio_layers} Audio). "
            f"Adjust num_text_layers/num_audio_layers parameters if needed."
        )
        # Attempt to proceed, but labels might be incorrect if counts are wrong.
        # Recalculate based on actual seq_len if possible, assuming 1 CLS
        # This part is heuristic - adjust if your structure is different
        if seq_len > 1:
            # Try to infer based on expected total length
            if num_text_layers + num_audio_layers == seq_len - 1:
                pass  # User params match reality
            else:
                # Default assumption if mismatch: split remaining equally
                print("Attempting to infer layer counts based on seq_len...")
                assumed_layers_per_modality = (seq_len - 1) // 2
                num_text_layers = assumed_layers_per_modality
                num_audio_layers = seq_len - 1 - num_text_layers  # Assign remainder to audio
                print(f"Inferred: num_text_layers={num_text_layers}, num_audio_layers={num_audio_layers}")
        else:
            print("Error: Cannot infer layer counts for seq_len <= 1.")
            return

    if batch_size > 1:
        print(f"Warning: Batch size is {batch_size}. Visualizing only the first sample.")

    # Ensure tensor is on CPU and detached for numpy conversion
    attn_weights_np = attn_weights[0].detach().cpu().numpy()  # Shape: [num_layers, seq_len, seq_len]

    print(f"Visualizing attention for {num_layers} layers with sequence length {seq_len}")

    # --- Create Labels ---
    # Adjust range if your layers are indexed differently (e.g., 1-24 instead of 0-23)
    labels = ["CLS"] + [f"T{i}" for i in range(num_text_layers)] + [f"A{i}" for i in range(num_audio_layers)]

    # Check if label count matches seq_len after potential inference
    if len(labels) != seq_len:
        print(f"Error: Number of generated labels ({len(labels)}) does not match sequence length ({seq_len}). Cannot plot.")
        return

    # --- Plotting ---
    # Determine overall min/max for consistent color scaling across layers
    vmin = np.min(attn_weights_np)
    vmax = np.max(attn_weights_np)

    # Create a figure with subplots (1 row, num_layers columns)
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5.5))  # Adjust figsize as needed
    if num_layers == 1:  # Handle case of single layer for proper indexing
        axes = [axes]

    for i in range(num_layers):
        ax = axes[i]
        sns.heatmap(
            attn_weights_np[i],
            ax=ax,
            cmap="viridis",  # Or 'hot', 'YlGnBu', etc.
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,  # Show color bar
            square=True,  # Make cells square
            linewidths=0.5,  # Add lines between cells
            cbar_kws={"shrink": 0.5},  # Make color bar smaller
            vmin=vmin,
            vmax=vmax,
        )  # Use consistent scaling

        ax.set_title(f"CMFF Layer {i + 1} Attention")
        ax.set_xlabel("Key (Attended To)")
        ax.set_ylabel("Query (Attending From)")

        # Adjust label appearance
        ax.tick_params(axis="x", labelsize=8, labelrotation=90)
        ax.tick_params(axis="y", labelsize=8, labelrotation=0)
        # Ensure all ticks are shown (matplotlib might skip some if too dense)
        ax.set_xticks(np.arange(seq_len) + 0.5)
        ax.set_yticks(np.arange(seq_len) + 0.5)

    plt.suptitle(f"Attention Weights Visualization (Seq Len: {seq_len})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlap and make space for suptitle
    plt.show()


if __name__ == "__main__":
    from config import Config
    from lightning_data import LightningData
    from lightning.pytorch.loggers import MLFlowLogger

    config = Config()
    logger = MLFlowLogger(
        tracking_uri="file:./mlruns",
        experiment_name="Default",  # config.Dataset, Default
        run_name=config.run_name,
        # log_model=True,
    )

    config.batch_size_eval = 1
    dm = LightningData(config)
    dm.setup(stage="test")
    for batch in dm.test_dataloader():
        print(len(batch))
        print(batch)
        break

    model = LightningModel(config)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params", total_params)

    # model = LightningModel.load_from_checkpoint(
    #     "/root/autodl-tmp/code1/mlruns/0/98742fe45fc741389c6685db84db4154/artifacts/checkpoints/best_model.ckpt",
    # config=config,
    # )
    # trainer = L.Trainer(
    #     logger=logger,
    #     precision="16-mixed",
    #     # accelerator="cpu",
    # )
    # trainer.test(model=model, datamodule=dm)
    # trainer.predict(model=model, datamodule=dm)

    for batch in dm.test_dataloader():
        print(len(batch))
        print(batch)

        # 计算FLOPs
        # 只用第一个batch做profile
        input_args = (batch,)
        flops, params = profile(model, inputs=input_args, verbose=False)
        print(f"模型FLOPs: {flops / 1e9:.2f} GFLOPs, 参数量: {params / 1e6:.2f} M")
        break
