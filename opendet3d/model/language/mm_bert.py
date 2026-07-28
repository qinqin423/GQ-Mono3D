"""BERT model from mmdetection."""

import logging
import os
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig
from transformers import BertModel as HFBertModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized, special_tokens_list
):
    """Generate attention mask between each pair of special tokens."""
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    special_tokens_mask = torch.zeros(
        (bs, num_token), device=input_ids.device
    ).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = torch.nonzero(special_tokens_mask)

    # 初始为对角阵
    attention_mask = (
        torch.eye(num_token, device=input_ids.device)
        .bool()
        .unsqueeze(0)
        .repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)

    for b in range(bs):
        curr_idxs = idxs[idxs[:, 0] == b, 1]
        previous_col = -1
        for col in curr_idxs:
            if col == 0 or col == num_token - 1:
                attention_mask[b, col, col] = True
                position_ids[b, col] = 0
            else:
                # 填充句子内部的注意力矩阵块
                attention_mask[
                    b, previous_col + 1 : col + 1, previous_col + 1 : col + 1
                ] = True
                position_ids[b, previous_col + 1 : col + 1] = torch.arange(
                    0, col - previous_col, device=input_ids.device
                )
            previous_col = col

    return attention_mask, position_ids.to(torch.long)


class BertModel(nn.Module):
    """BERT model for language embedding only encoder."""

    def __init__(
        self,
        name: str = "bert-base-uncased",
        max_tokens: int = 256,
        pad_to_max: bool = True,
        use_sub_sentence_represent: bool = False,
        special_tokens_list: list = None,
        add_pooling_layer: bool = False,
        num_layers_of_embedded: int = 1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        # Use the fast tokenizer implementation.
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

        self.language_backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "body",
                        BertEncoder(
                            name,
                            add_pooling_layer=add_pooling_layer,
                            num_layers_of_embedded=num_layers_of_embedded,
                            use_checkpoint=use_checkpoint,
                        ),
                    )
                ]
            )
        )

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert (
                special_tokens_list is not None
            ), "special_tokens should not be None if use_sub_sentence_represent is True"

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list
            )

    def forward(self, captions: Sequence[str]) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device

        if isinstance(captions, str):
            captions = [captions]
        else:
            captions = list(captions)

        tokenized = self.tokenizer(
            captions,
            max_length=self.max_tokens,
            padding="max_length" if self.pad_to_max else "longest",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        input_ids = tokenized.input_ids
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = (
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens
                )
            )
        else:
            attention_mask = tokenized.attention_mask
            position_ids = None

        token_type_ids = tokenized.get("token_type_ids", torch.zeros_like(input_ids))

        tokenizer_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }
        language_dict_features = self.language_backbone(tokenizer_input)

        if self.use_sub_sentence_represent:
            language_dict_features["position_ids"] = position_ids
            language_dict_features["text_token_mask"] = (
                tokenized.attention_mask.bool()
            )
        return language_dict_features


class BertEncoder(nn.Module):
    """BERT encoder for language embedding."""

    def __init__(
        self,
        name: str,
        add_pooling_layer: bool = False,
        num_layers_of_embedded: int = 1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint

        # 屏蔽 transformers 的冗余日志
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.model = HFBertModel.from_pretrained(
            name,
            add_pooling_layer=add_pooling_layer,
            config=config,
            # 关键：使用 eager 模式以手动兼容 3D-MOOD 的自定义 mask
            attn_implementation="eager",
        )

        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        input_ids = x["input_ids"]
        mask = x["attention_mask"]

        # --- 核心修正代码段 ---
        # 如果 mask 是 [Batch, Seq, Seq]，transformers 会报 expand 维度错误
        # 此时需要转换为浮点数并增加维度，使其符合 transformers 对自定义 mask 的 4D 期望 [Batch, 1, Seq, Seq]
        if mask is not None and mask.dim() == 3:
            # 将 bool 转换为 float，并扩展维度
            extended_attention_mask = mask.unsqueeze(1).to(dtype=self.model.dtype)
            # 转换 mask：0 为关注，非常小的负数为屏蔽
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.model.dtype).min
        else:
            extended_attention_mask = mask

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            position_ids=x["position_ids"],
            token_type_ids=x["token_type_ids"],
            output_hidden_states=True,
        )

        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(
            encoded_layers[-self.num_layers_of_embedded :], 1
        ).mean(1)

        # 计算最终嵌入，考虑原始掩码
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        elif mask.dim() == 3:
            # 如果是矩阵 mask，使用其对角线（即 token 是否存在的 mask）
            token_mask = torch.diagonal(mask, dim1=1, dim2=2)
            embedded = features * token_mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1],
        }
        return results