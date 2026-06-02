# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    random.seed(seed)

    def read_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                if "text" in item and "label" in item and item["text"].strip():
                    data.append({"text" : item["text"], "label" : int(item["label"])})
        return data

    train_all = read_jsonl(train_tsv_path)
    random.shuffle(train_all)

    val_size = int(len(train_all) * val_ratio)
    train_data = train_all[:val_size]
    val_data = train_all[val_size:]

    test_data = read_jsonl(test_tsv_path) if test_tsv_path else []

    return train_data, val_data, test_data

    #raise NotImplementedError("make_sentiment_dataset을 구현하세요.")


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoded_text = self.tokenizer.encode(text)
        if len(encoded_text) > self.max_length:
            encoded_text = encoded_text[:self.max_length]
        else:
            padding = self.max_length - len(encoded_text)
            encoded_text.extend([self.pad_id] * padding)

        input_ids = torch.tensor(encoded_text, dtype=torch.long)

        return input_ids, label

        #raise NotImplementedError("ReviewSentimentDataset.__getitem__을 구현하세요.")


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.gpt.config["emb_dim"], self.num_labels)
        #raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        output = self.gpt(input_ids)
        vector = output[:, -1, :]
        logits = self.classifier(self.dropout(vector))
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return (loss, logits)
        return logits
        
        #raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    
    total_loss = 0.0
    count = 0
    correct = 0
    total_size = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total_size += labels.size(0)

    loss_mean = total_loss / len(train_loader)
    accuracy = correct / total_size

    return loss_mean, accuracy

    #raise NotImplementedError("train_epoch_sentiment를 구현하세요.")


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()

    total_loss = 0.0
    total_size = 0
    correct = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total_size += labels.size(0)

    loss_mean = total_loss / len(data_loader)
    accuracy = correct / total_size

    return loss_mean, accuracy
            

    #raise NotImplementedError("evaluate_sentiment를 구현하세요.")
