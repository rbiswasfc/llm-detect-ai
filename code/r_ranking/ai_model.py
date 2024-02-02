import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel


def get_ranking_loss(logits, labels, margin=0.7):
    logits = torch.sigmoid(logits)
    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)

    logits1 = logits.unsqueeze(1)
    logits2 = logits.unsqueeze(0)

    y_ij = torch.sign(labels1 - labels2)
    r_ij = logits1 - logits2

    loss = torch.clamp(-r_ij*y_ij + margin, min=0.0).mean()
    return loss


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

        # classifier
        num_features = self.backbone.config.hidden_size
        self.classifier = nn.Linear(num_features, 1)

        self.pool = MeanPooling()

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

        return embeddings

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # features
        features = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        features = self.dropout(features)
        logits = self.classifier(features).reshape(-1)

        # loss
        loss = None
        labels = labels.reshape(-1)
        if labels is not None:
            loss = get_ranking_loss(logits, labels)

        return logits, loss
