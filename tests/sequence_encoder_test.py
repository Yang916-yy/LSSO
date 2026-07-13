import pytest
import torch

from examples.models import ProteinFitnessModel, SequenceMixerEncoder
from experiments.flip_aav import ProteinTokenizer


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_sequence_encoder_masked_forward_backward(mixer):
    model = SequenceMixerEncoder(
        32,
        max_length=12,
        pad_token_id=0,
        dim=32,
        depth=2,
        num_heads=4,
        mixer=mixer,
        rank=8,
        projection_dim=16,
        dropout=0.0,
    )
    ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]])
    mask = ids.ne(0)
    output = model(ids, mask)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.token_embedding.weight.grad is not None


def test_padding_does_not_change_sequence_embedding():
    model = SequenceMixerEncoder(
        16, max_length=8, pad_token_id=0, dim=16, depth=1,
        num_heads=4, mixer="rrlsso", rank=8, dropout=0.0,
    ).eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, 0, 0]])
    torch.testing.assert_close(model(short), model(padded), rtol=2e-5, atol=2e-5)


def test_protein_tokenizer_and_fitness_head():
    tokenizer = ProteinTokenizer()
    ids, mask = tokenizer.batch(["ACDX", "MNPQRST"], max_length=6)
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size, max_length=6, pad_token_id=0, dim=16,
        depth=1, num_heads=4, mixer="lsso", rank=8, dropout=0.0,
    )
    prediction = ProteinFitnessModel(encoder)(ids, mask)
    assert prediction.shape == (2,)
    assert mask.sum(dim=1).tolist() == [4, 6]
