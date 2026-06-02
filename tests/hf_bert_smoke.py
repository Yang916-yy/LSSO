from __future__ import annotations

import torch

from examples.models.bert import iter_bert_lsso_layers, replace_bert_self_attention_with_lsso


def main() -> None:
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        num_labels=3,
    )
    model = BertForSequenceClassification(config)
    replace_bert_self_attention_with_lsso(
        model,
        rank=8,
        gamma_max=0.3,
        theta_gamma_init=-4.0,
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -4:] = 0
    labels = torch.tensor([0, 2])

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out.loss.backward()

    layers = iter_bert_lsso_layers(model)
    assert len(layers) == config.num_hidden_layers
    assert layers[0].last_diagnostics is not None
    print("hf bert lsso smoke passed")


if __name__ == "__main__":
    main()
