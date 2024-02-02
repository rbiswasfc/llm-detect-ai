import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10034742


class SupContrastiveLoss(nn.Module):
    def __init__(self, temperature, device):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, outputs, labels):
        N = outputs.size()[0]
        labels = labels.reshape(N, 1)
        self_similarity_mask = torch.ones((N, N)).fill_diagonal_(0).to(self.device)

        pos_mask = torch.eq(labels, labels.T).float()
        neg_mask = torch.abs(pos_mask - 1)

        H = torch.matmul(outputs, outputs.T) * self_similarity_mask
        H_pos = H * pos_mask
        H_neg = H * neg_mask

        v_pos = torch.mean(torch.exp(torch.div(H_pos, self.temperature)), dim=1)
        v_neg = torch.mean(torch.exp(torch.div(H_neg, self.temperature)), dim=1)

        loss = (-1/N) * torch.sum(torch.log(v_pos/(v_pos + v_neg)))

        return loss

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Rank Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class AiModel(nn.Module):
    """
    The LLM Detect AI Generated Text Model
    """

    def __init__(self, cfg, device):
        print("initializing the Rank Model...")

        super(AiModel, self).__init__()
        self.cfg = cfg

        # ----------------------------- Backbone -----------------------------------------#
        backbone_config = AutoConfig.from_pretrained(self.cfg.model.backbone_path)
        backbone_config.update(
            {
                "use_cache": False,
            }
        )

        self.backbone = AutoModel.from_pretrained(
            self.cfg.model.backbone_path,
            config=backbone_config
        )
        if self.cfg.model.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(self.cfg.model.dropout_rate)

        hidden_size = self.backbone.config.hidden_size
        project_dim = self.cfg.model.projection_dim
        self.pool = MeanPooling()

        self.projection_head = nn.Sequential(
            nn.Dropout(self.cfg.model.dropout_rate),
            nn.Linear(hidden_size, project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )

        # loss function
        self.loss_fn = SupContrastiveLoss(
            temperature=self.cfg.model.temperature,
            device=device,
        )

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        encoder_layer = outputs.last_hidden_state
        embeddings = self.pool(encoder_layer, attention_mask)
        embeddings = self.projection_head(embeddings)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # features
        embeddings = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (bs, num_features)

        # loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(embeddings, labels)

        return loss
