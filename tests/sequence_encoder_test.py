import pytest
import torch

from examples.models import ProteinFitnessModel, SequenceMixerEncoder
from experiments.beir_retrieval import chunked_topk
from experiments.flip_aav import (
    FitnessDataset,
    LengthBucketBatchSampler,
    ProteinTokenizer,
    collate_proteins,
)


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


def test_chunked_topk_matches_full_matrix():
    generator = torch.Generator().manual_seed(0)
    queries = torch.randn(7, 12, generator=generator)
    documents = torch.randn(23, 12, generator=generator)
    expected = (queries @ documents.T).topk(5, dim=1).indices
    actual = chunked_topk(
        queries, documents, k=5, device=torch.device("cpu"),
        query_chunk=3, document_chunk=7,
    )
    assert torch.equal(actual, expected)


def test_length_bucket_sampler_groups_similar_lengths():
    lengths = [90, 10, 80, 20, 70, 30, 60, 40]
    batches = list(LengthBucketBatchSampler(lengths, batch_size=2, seed=0))
    assert sorted(index for batch in batches for index in batch) == list(range(8))
    assert all(max(lengths[i] for i in batch) - min(lengths[i] for i in batch) <= 10
               for batch in batches)


def test_pretokenized_protein_collate():
    class Rows:
        def __getitem__(self, key):
            return {"aa_seq": ["ACD", "MNPQRST"], "label": [1.0, 2.0]}[key]

    tokenizer = ProteinTokenizer()
    dataset = FitnessDataset(Rows(), "aa_seq", "label", tokenizer, max_length=5)
    ids, mask, targets = collate_proteins([dataset[0], dataset[1]])
    assert ids.shape == (2, 5)
    assert mask.sum(dim=1).tolist() == [3, 5]
    assert targets.tolist() == [1.0, 2.0]
