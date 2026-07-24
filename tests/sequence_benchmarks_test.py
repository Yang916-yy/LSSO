from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from examples.models import (
    ReverseComplementSequenceClassifier,
    SequenceClassifier,
    SequenceMixerEncoder,
    SequencePairClassifier,
    SequenceValueEncoder,
)
from experiments.genomic_benchmarks import (
    NucleotideTokenizer,
    StringSequenceDataset,
    apply_training_profile,
)
from experiments.sequence_benchmarks.common import (
    _scheduler_lambda,
    _classification_metrics,
    LengthBucketBatchSampler,
    TrainingConfig,
    collate_token_pairs,
    collate_tokens,
    collate_values,
    make_loader,
    stratified_fold_indices,
    stratified_split_indices,
    stratified_subset_indices,
    train_classifier,
)
from experiments.run_rrlsso_dna_program import choose_with_margin
from experiments.lra_benchmark import prepare_retrieval
from experiments.sequence_benchmarks.lra_data import (
    KAGGLE_LRA_HANDLE,
    CharacterVocabulary,
    PathfinderDataset,
    build_packed_pairs,
    build_packed_tokens,
    download_kaggle_lra,
    iter_aan,
    iter_listops,
    resolve_listops_files,
    resolve_pathfinder_directory,
)
from experiments.summarize_sequence_benchmarks import aggregate, average_ranks


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_continuous_sequence_classifier_forward_backward(mixer: str):
    encoder = SequenceValueEncoder(
        3,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=4,
        mixer=mixer,
        rank=8,
        dropout=0.0,
    )
    model = SequenceClassifier(encoder, 4)
    values = torch.randn(2, 6, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    logits = model(values, mask)
    assert logits.shape == (2, 4)
    logits.square().mean().backward()
    assert encoder.input_projection.weight.grad is not None


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_spatial_value_encoder_uses_complete_2d_grid(mixer: str):
    encoder = SequenceValueEncoder(
        1,
        max_length=16,
        spatial_shape=(4, 4),
        dim=16,
        depth=1,
        num_heads=4,
        mixer=mixer,
        rank=8,
        dropout=0.0,
    )
    values = torch.randn(2, 16, 1, requires_grad=True)
    output = encoder(values)
    assert output.shape == (2, 16)
    output.square().mean().backward()
    assert values.grad is not None
    mask = torch.tensor([[1] * 12 + [0] * 4, [1] * 16], dtype=torch.bool)
    changed_padding = values.detach().clone()
    changed_padding[0, 12:] += 1000
    torch.testing.assert_close(
        encoder(values.detach(), mask)[0],
        encoder(changed_padding, mask)[0],
        rtol=1e-5,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="complete grid"):
        encoder(values[:, :-1])


def test_spatial_value_encoder_rejects_mismatched_grid():
    with pytest.raises(ValueError, match="does not match"):
        SequenceValueEncoder(1, max_length=16, spatial_shape=(3, 4))


def test_spatial_local_blocks_are_interleaved_and_receive_gradients():
    encoder = SequenceValueEncoder(
        1,
        max_length=64,
        spatial_shape=(8, 8),
        dim=16,
        depth=3,
        num_heads=4,
        mixer="rrlsso",
        rank=8,
        local_spatial_kernel=3,
        local_spatial_dilations=(1, 2, 3),
        dropout=0.0,
    )
    values = torch.randn(2, 64, 1, requires_grad=True)
    output = encoder(values)
    assert output.shape == (2, 16)
    assert [block.depthwise.dilation for block in encoder.local_spatial_blocks] == [
        (1, 1), (2, 2), (3, 3)
    ]
    output.square().mean().backward()
    assert encoder.local_spatial_blocks[0].depthwise.weight.grad is not None
    assert encoder.local_spatial_blocks[-1].pointwise.weight.grad is not None


def test_spatial_local_configuration_rejects_ambiguous_depth():
    with pytest.raises(ValueError, match="match depth"):
        SequenceValueEncoder(
            1,
            max_length=16,
            spatial_shape=(4, 4),
            dim=16,
            depth=2,
            num_heads=4,
            local_spatial_kernel=3,
            local_spatial_dilations=(1,),
        )


def test_low_rank_learned_position_embedding_reduces_long_table():
    full = SequenceValueEncoder(
        3, max_length=1000, dim=64, depth=1, num_heads=4,
        mixer="rrlsso", rank=8, position_rank=0, dropout=0.0,
    )
    low_rank = SequenceValueEncoder(
        3, max_length=1000, dim=64, depth=1, num_heads=4,
        mixer="rrlsso", rank=8, position_rank=16, dropout=0.0,
    )
    full_position = full.position_embedding.numel()
    low_rank_position = (
        low_rank.position_embedding.numel()
        + low_rank.position_projection.weight.numel()
    )
    assert low_rank_position < full_position
    values = torch.randn(2, 20, 3)
    mask = torch.ones(2, 20, dtype=torch.bool)
    assert low_rank(values, mask).shape == (2, 64)


def test_multiscale_temporal_stem_is_padding_invariant_and_differentiable():
    encoder = SequenceValueEncoder(
        3,
        max_length=16,
        dim=16,
        depth=1,
        num_heads=4,
        mixer="rrlsso",
        rank=8,
        pooling="meanmax",
        local_stem_kernels=(3, 7),
        dropout=0.0,
    ).eval()
    values = torch.randn(2, 8, 3, requires_grad=True)
    valid = torch.ones(2, 8, dtype=torch.bool)
    padded = torch.cat((values.detach(), torch.randn(2, 5, 3)), dim=1)
    padded_valid = torch.cat((valid, torch.zeros(2, 5, dtype=torch.bool)), dim=1)
    expected = encoder(values, valid)
    actual = encoder(padded, padded_valid)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward()
    assert encoder.local_temporal_stem.depthwise[0].weight.grad is not None


def test_multiscale_temporal_stem_rejects_even_kernels():
    with pytest.raises(ValueError, match="positive odd"):
        SequenceValueEncoder(
            3,
            max_length=8,
            dim=16,
            depth=1,
            num_heads=4,
            local_stem_kernels=(3, 4),
        )


def test_pair_classifier_and_collates():
    encoder = SequenceMixerEncoder(
        16,
        max_length=8,
        pad_token_id=0,
        dim=16,
        depth=1,
        num_heads=4,
        mixer="rrlsso",
        rank=8,
        dropout=0.0,
    )
    model = SequencePairClassifier(encoder)
    batch = collate_token_pairs(
        [
            (torch.tensor([2, 3]), torch.tensor([4]), 1),
            (torch.tensor([5]), torch.tensor([6, 7, 8]), 0),
        ]
    )
    logits = model(batch["first"], batch["first_mask"], batch["second"], batch["second_mask"])
    assert logits.shape == (2, 2)
    assert batch["first_mask"].sum(dim=1).tolist() == [2, 1]
    assert batch["second_mask"].sum(dim=1).tolist() == [1, 3]
    assert batch["first_padding_ratio"] == pytest.approx(0.25)
    assert batch["second_padding_ratio"] == pytest.approx(1.0 / 3.0)

    token_batch = collate_tokens([(torch.tensor([2, 3]), 0), (torch.tensor([4]), 1)])
    assert token_batch["inputs"].tolist() == [[2, 3], [4, 0]]
    assert token_batch["padding_ratio"] == pytest.approx(0.25)
    value_batch = collate_values([(torch.ones(2, 3), 0), (torch.ones(4, 3), 1)])
    assert value_batch["mask"].sum(dim=1).tolist() == [2, 4]
    assert value_batch["padding_ratio"] == pytest.approx(0.25)


def test_stratified_split_is_disjoint_and_keeps_each_class_in_train():
    labels = [0] * 5 + [1] * 4 + [2]
    train, validation = stratified_split_indices(labels, 0.25, seed=7)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(len(labels)))
    assert {labels[index] for index in train} == {0, 1, 2}


def test_stratified_folds_are_disjoint_balanced_and_cover_validation_once():
    labels = [0] * 6 + [1] * 6 + [2] * 6
    validation_sets = []
    for fold in range(3):
        train, validation = stratified_fold_indices(labels, 3, fold, seed=11)
        assert set(train).isdisjoint(validation)
        assert sorted(train + validation) == list(range(len(labels)))
        assert [labels[index] for index in validation].count(0) == 2
        assert [labels[index] for index in validation].count(1) == 2
        assert [labels[index] for index in validation].count(2) == 2
        validation_sets.append(set(validation))
    assert set.union(*validation_sets) == set(range(len(labels)))
    assert sum(map(len, validation_sets)) == len(labels)


def test_classification_metrics_include_multiclass_matthews_correlation():
    perfect = _classification_metrics([0, 1, 2, 1], [0, 1, 2, 1], 3)
    assert perfect["matthews_correlation"] == pytest.approx(1.0)
    constant = _classification_metrics([0, 0, 0, 0], [0, 1, 0, 1], 2)
    assert constant["matthews_correlation"] == 0.0


def test_recipe_selection_prefers_simpler_candidate_inside_margin():
    scores = {"none": 0.800, "rc_train": 0.802, "rc_train_eval": 0.803}
    assert choose_with_margin(
        scores, ("none", "rc_train", "rc_train_eval"), margin=0.0025
    ) == "rc_train"


def test_cosine_scheduler_respects_nonzero_floor():
    assert _scheduler_lambda(100, 0, 100, 0.1) == pytest.approx(0.1)
    assert _scheduler_lambda(0, 0, 100, 0.1) == pytest.approx(1.0)


def test_hyenadna_flavor_profile_maps_only_generic_training_choices():
    from argparse import Namespace

    args = Namespace(
        training_profile="hyenadna-flavor", epochs=60, patience=12,
        batch_size=64, eval_batch_size=64, lr=3e-4, weight_decay=0.01,
        warmup_ratio=0.05, min_lr_ratio=0.0, dropout=0.1,
        embedding_dropout=None, pooling="max",
        reverse_complement_probability=0.5, reverse_complement_eval=True,
        mutation_probability=0.0, mutation_clean_epochs=0,
    )
    apply_training_profile(args)
    assert (args.epochs, args.lr, args.warmup_ratio) == (100, 6e-4, 0.01)
    assert (args.dropout, args.embedding_dropout) == (0.0, 0.1)
    assert args.weight_decay == 0.01
    assert args.min_lr_ratio == 0.1
    assert args.reverse_complement_probability == 0.0

    args.training_profile = "hyenadna-flavor-rc"
    apply_training_profile(args)
    assert args.reverse_complement_probability == 0.5
    assert args.reverse_complement_eval is False

    args.training_profile = "hyenadna-flavor-rc-mutation"
    apply_training_profile(args)
    assert args.mutation_probability == 0.002
    assert args.mutation_clean_epochs == 20


def test_stratified_subset_is_exact_deterministic_and_order_unbiased():
    labels = [0] * 10 + [1] * 6 + [2] * 4
    selected = stratified_subset_indices(labels, 10, seed=5)
    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert selected == stratified_subset_indices(labels, 10, seed=5)
    assert [labels[index] for index in selected].count(0) == 5
    assert [labels[index] for index in selected].count(1) == 3
    assert [labels[index] for index in selected].count(2) == 2


def test_length_bucket_sampler_state_resumes_next_epoch():
    sampler = LengthBucketBatchSampler([5, 1, 4, 2, 3], batch_size=2, seed=11)
    list(sampler)
    state = sampler.state_dict()
    expected_next = list(sampler)
    resumed = LengthBucketBatchSampler([5, 1, 4, 2, 3], batch_size=2, seed=11)
    resumed.load_state_dict(state)
    assert list(resumed) == expected_next


def test_listops_vocabulary_and_memory_mapped_cache(tmp_path: Path):
    source = tmp_path / "basic_train.tsv"
    source.write_text("Source\tTarget\n[ MAX 1 2 ]\t2\n[ MIN 7 4 ]\t4\n", encoding="utf-8")
    rows = list(iter_listops(source))
    vocabulary = CharacterVocabulary.from_listops(text for text, _ in rows)
    prefix = tmp_path / "cache" / "train-l32"
    dataset = build_packed_tokens(
        prefix, rows, lambda text: vocabulary.encode_listops(text, max_length=32)
    )
    assert len(dataset) == 2
    assert dataset[0][1] == 2
    assert dataset[0][0][-1].item() == vocabulary.eos_token_id
    assert prefix.with_suffix(".tokens.bin").exists()

    pairs = build_packed_pairs(
        tmp_path / "cache" / "pairs-l16",
        [("paper a", "paper b", 1), ("paper c", "paper d", 0)],
        lambda text: vocabulary.encode_chars(text, max_length=16),
    )
    assert len(pairs) == 2
    assert pairs[1][2] == 0


def test_kaggle_lra_mirror_layout_resolves_without_copying_large_files(tmp_path: Path):
    listops = tmp_path / "_kaggle" / "listops" / "listops"
    listops.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (listops / f"{split}.tsv").write_text("Source\tTarget\n[ MAX 1 2 ]\t2\n")
    files = resolve_listops_files(tmp_path)
    assert files["train"] == listops / "train.tsv"

    pathfinder = (
        tmp_path / "_kaggle" / "pathfinder" / "pathfinder" / "pathfinder32"
    )
    pathfinder.mkdir(parents=True)
    assert resolve_pathfinder_directory(tmp_path, 32) == pathfinder


def test_kaggle_lra_download_is_version_pinned_and_writes_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []

    class FakeKaggleHub:
        @staticmethod
        def dataset_download(handle, **kwargs):
            calls.append((handle, kwargs))
            return kwargs["output_dir"]

    monkeypatch.setitem(__import__("sys").modules, "kagglehub", FakeKaggleHub)
    resolved = download_kaggle_lra(tmp_path)
    assert resolved == tmp_path / "_kaggle"
    assert calls[0][0] == KAGGLE_LRA_HANDLE
    manifest = json.loads((resolved / ".lsso-source.json").read_text())
    assert manifest["handle"] == KAGGLE_LRA_HANDLE


def test_packed_cache_manifest_invalidates_stale_tokens(tmp_path: Path):
    vocabulary = CharacterVocabulary(list("abc"))
    prefix = tmp_path / "cache" / "rows"
    first = build_packed_tokens(
        prefix,
        [("a", 0)],
        lambda text: vocabulary.encode_chars(text, 8),
        manifest={"schema": 1, "source": "first"},
    )
    assert first[0][1] == 0
    rebuilt = build_packed_tokens(
        prefix,
        [("bc", 1)],
        lambda text: vocabulary.encode_chars(text, 8),
        manifest={"schema": 1, "source": "second"},
    )
    assert rebuilt[0][1] == 1
    assert len(rebuilt[0][0]) == 3


def test_aan_reader_accepts_integral_float_labels(tmp_path: Path):
    source = tmp_path / "aan.tsv"
    source.write_text(
        "1.0\tA\tB\tb'first paper'\tb'second paper'\n"
        "0\tC\tD\tb'third paper'\tb'fourth paper'\n"
        "0.5\tE\tF\tb'invalid'\tb'label'\n",
        encoding="utf-8",
    )
    assert list(iter_aan(source)) == [
        ("b'first paper'", "b'second paper'", 1),
        ("b'third paper'", "b'fourth paper'", 0),
    ]


def test_fixed_byte_vocabulary_handles_utf8_without_training_scan():
    vocabulary = CharacterVocabulary.from_bytes()
    encoded = vocabulary.encode_bytes("Aé", max_length=8)
    assert encoded[:-1] == [
        vocabulary.lookup[chr(value)] for value in "Aé".encode("utf-8")
    ]
    assert encoded[-1] == vocabulary.eos_token_id


def test_retrieval_preprocessing_builds_nonempty_byte_cache(tmp_path: Path):
    source = tmp_path / "data" / "aan"
    source.mkdir(parents=True)
    for split in ("train", "eval", "test"):
        (source / f"new_aan_pairs.{split}.tsv").write_text(
            "1.0\tA\tB\tb'first paper'\tb'second paper'\n",
            encoding="utf-8",
        )
    train, validation, test, vocabulary = prepare_retrieval(
        tmp_path / "data", tmp_path / "cache", max_length=32, download=False
    )
    assert (len(train), len(validation), len(test)) == (1, 1, 1)
    assert vocabulary.vocab_size == 259
    assert train[0][0][-1] == vocabulary.eos_token_id


def test_pathfinder_reader_uses_official_metadata_layout(tmp_path: Path):
    root = tmp_path / "pathfinder32"
    metadata = root / "curv_contour_length_14" / "metadata"
    image_dir = root / "curv_contour_length_14" / "imgs" / "0"
    metadata.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.fromarray(np.full((32, 32), 127, dtype=np.uint8)).save(image_dir / "sample.png")
    (metadata / "0.npy").write_text("imgs/0 sample.png 0 1\n", encoding="utf-8")
    dataset = PathfinderDataset(root)
    values, label = dataset[0]
    assert values.shape == (1024, 1)
    assert label == 1


def test_genomic_tokenizer_and_dataset():
    tokenizer = NucleotideTokenizer()
    dataset = StringSequenceDataset([("ACGTN", 1), ("TGCA", 0)], tokenizer, max_length=4)
    tokens, label = dataset[0]
    assert tokens.shape == (4,)
    assert label == 1
    assert tokens.min().item() > tokenizer.pad_token_id


def test_local_motif_stem_preserves_shape_padding_and_gradients():
    encoder = SequenceMixerEncoder(
        8,
        max_length=12,
        pad_token_id=0,
        dim=16,
        depth=1,
        num_heads=4,
        mixer="rrlsso",
        rank=8,
        dropout=0.0,
        local_motif_kernel=7,
    )
    short = torch.tensor([[2, 3, 4, 5]])
    padded = torch.tensor([[2, 3, 4, 5, 0, 0]])
    output = encoder(short)
    assert output.shape == (1, 16)
    torch.testing.assert_close(output, encoder(padded), rtol=2e-5, atol=2e-5)
    output.square().mean().backward()
    assert encoder.local_motif_stem[0].weight.grad is not None


@pytest.mark.parametrize("kernel", [-1, 2, 6])
def test_local_motif_stem_rejects_invalid_kernel(kernel: int):
    with pytest.raises(ValueError, match="positive odd"):
        SequenceMixerEncoder(
            8, max_length=12, pad_token_id=0, dim=16, depth=1,
            num_heads=4, mixer="rrlsso", rank=8, local_motif_kernel=kernel,
        )


def test_gated_dilated_motif_blocks_preserve_padding_and_receive_gradients():
    encoder = SequenceMixerEncoder(
        8,
        max_length=24,
        pad_token_id=0,
        dim=16,
        depth=2,
        num_heads=4,
        mixer="rrlsso",
        rank=8,
        dropout=0.0,
        position_rank=4,
        local_motif_dilations=(1, 1, 4, 16, 64),
    )
    assert [block.feature_depthwise.dilation[0] for block in encoder.local_motif_blocks] == [
        1, 1, 4, 16, 64
    ]
    short = torch.tensor([[2, 3, 4, 5, 2, 3]])
    padded = torch.tensor([[2, 3, 4, 5, 2, 3, 0, 0]])
    output = encoder(short)
    torch.testing.assert_close(output, encoder(padded), rtol=2e-5, atol=2e-5)
    output.square().mean().backward()
    assert encoder.local_motif_blocks[0].feature_depthwise.weight.grad is not None
    assert encoder.local_motif_blocks[-1].gate_pointwise.weight.grad is not None


def test_local_motif_stems_are_mutually_exclusive():
    with pytest.raises(ValueError, match="different local stems"):
        SequenceMixerEncoder(
            8, max_length=12, pad_token_id=0, dim=16, depth=2,
            num_heads=4, mixer="rrlsso", rank=8, local_motif_kernel=7,
            local_motif_dilations=(1, 4),
        )


def test_reverse_complement_preserves_right_padding_and_is_an_involution():
    tokenizer = NucleotideTokenizer()
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size, max_length=8, pad_token_id=tokenizer.pad_token_id,
        dim=16, depth=1, num_heads=4, mixer="rrlsso", rank=8, dropout=0.0,
    )
    model = ReverseComplementSequenceClassifier(
        encoder, 2, complement_ids=tokenizer.complement_ids,
    )
    ids = torch.tensor([[2, 2, 3, 4, 0, 0], [2, 3, 6, 0, 0, 0]])
    mask = ids.ne(0)
    reverse, reverse_mask = model.reverse_complement(ids, mask)
    assert reverse.tolist() == [[3, 4, 5, 5, 0, 0], [6, 4, 5, 0, 0, 0]]
    restored, restored_mask = model.reverse_complement(reverse, reverse_mask)
    assert torch.equal(restored, ids)
    assert torch.equal(restored_mask, mask)


def test_point_mutation_changes_only_canonical_bases_before_clean_stage():
    tokenizer = NucleotideTokenizer()
    encoder = SequenceMixerEncoder(
        tokenizer.vocab_size, max_length=8, pad_token_id=0,
        dim=16, depth=1, num_heads=4, mixer="rrlsso", rank=8, dropout=0.0,
    )
    model = ReverseComplementSequenceClassifier(
        encoder,
        2,
        complement_ids=tokenizer.complement_ids,
        mutation_probability=1.0,
        mutation_stop_epoch=80,
    )
    inputs = torch.tensor([[2, 3, 4, 5, 6, 1, 0]])
    mask = inputs.ne(0)
    mutated = model.mutate(inputs, mask)
    assert torch.all(mutated[:, :4] != inputs[:, :4])
    assert torch.equal(mutated[:, 4:], inputs[:, 4:])
    model.set_augmentation_epoch(80)
    assert torch.equal(model.mutate(inputs, mask), inputs)


@pytest.mark.parametrize("pooling", ["mean", "max", "meanmax"])
def test_sequence_pooling_ignores_right_padding(pooling: str):
    encoder = SequenceMixerEncoder(
        16, max_length=8, pad_token_id=0, dim=16, depth=1, num_heads=4,
        mixer="rrlsso", rank=8, dropout=0.0, pooling=pooling,
    ).eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, 0, 0]])
    torch.testing.assert_close(encoder(short), encoder(padded), rtol=2e-5, atol=2e-5)


def test_shared_trainer_writes_and_resumes_checkpoints(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = [
        (torch.tensor([2, 3]), 0),
        (torch.tensor([3, 4]), 1),
        (torch.tensor([2, 4]), 0),
        (torch.tensor([4, 3]), 1),
    ]
    train = make_loader(
        rows,
        batch_size=2,
        workers=0,
        device=torch.device("cpu"),
        collate_fn=collate_tokens,
        train=True,
        seed=0,
    )
    evaluation = make_loader(
        rows,
        batch_size=2,
        workers=0,
        device=torch.device("cpu"),
        collate_fn=collate_tokens,
        train=False,
        seed=0,
    )
    assert train.persistent_workers is False  # workers=0 always disables persistence
    assert evaluation.persistent_workers is False
    encoder = SequenceMixerEncoder(
        8, max_length=4, pad_token_id=0, dim=8, depth=1,
        num_heads=2, mixer="rrlsso", rank=4, dropout=0.0,
    )
    model = SequenceClassifier(encoder, 2)
    config = TrainingConfig(
        output=str(tmp_path / "run"),
        epochs=1,
        seed=0,
    )
    result = train_classifier(
        model,
        train,
        evaluation,
        evaluation,
        num_classes=2,
        config=config,
        metadata={"suite": "test", "dataset": "tiny", "mixer": "rrlsso"},
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "test_metrics.json").exists()
    state = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    assert state["run_config"]["source_revision"]["git_commit"]
    assert isinstance(state["run_config"]["argv"], list)
    assert {group["weight_decay"] for group in state["optimizer"]["param_groups"]} == {
        0.0,
        0.01,
    }
    assert "alpha_mean" in state["metrics"]
    # A completed one-epoch run can be loaded and evaluated without retraining.
    train_classifier(
        model,
        train,
        evaluation,
        evaluation,
        num_classes=2,
        config=config,
        metadata={"suite": "test", "dataset": "tiny", "mixer": "rrlsso"},
    )


def test_gradient_accumulation_matches_effective_large_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = [
        (torch.tensor([2, 3]), 0),
        (torch.tensor([3, 4]), 1),
        (torch.tensor([2, 4]), 0),
        (torch.tensor([4, 3]), 1),
        (torch.tensor([2, 2]), 0),
    ]
    large_loader = make_loader(
        rows, batch_size=4, workers=0, device=torch.device("cpu"),
        collate_fn=collate_tokens, train=False, seed=0,
    )
    micro_loader = make_loader(
        rows, batch_size=2, workers=0, device=torch.device("cpu"),
        collate_fn=collate_tokens, train=False, seed=0,
    )
    encoder = SequenceMixerEncoder(
        8, max_length=4, pad_token_id=0, dim=8, depth=1,
        num_heads=2, mixer="rrlsso", rank=4, dropout=0.0,
    )
    large_model = SequenceClassifier(encoder, 2)
    micro_model = copy.deepcopy(large_model)
    metadata = {"suite": "test", "dataset": "tiny", "mixer": "rrlsso"}
    train_classifier(
        large_model, large_loader, large_loader, None, num_classes=2,
        config=TrainingConfig(output=str(tmp_path / "large"), epochs=1, seed=0),
        metadata=metadata,
    )
    train_classifier(
        micro_model, micro_loader, micro_loader, None, num_classes=2,
        config=TrainingConfig(
            output=str(tmp_path / "micro"), epochs=1, seed=0, grad_accum=2
        ),
        metadata=metadata,
    )
    for large_parameter, micro_parameter in zip(
        large_model.parameters(), micro_model.parameters(), strict=True
    ):
        torch.testing.assert_close(large_parameter, micro_parameter, rtol=1e-5, atol=1e-6)


def test_validation_only_training_never_writes_test_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = [(torch.tensor([2, 3]), 0), (torch.tensor([3, 4]), 1)]
    loader = make_loader(
        rows, batch_size=2, workers=0, device=torch.device("cpu"),
        collate_fn=collate_tokens, train=False, seed=0,
    )
    encoder = SequenceMixerEncoder(
        8, max_length=4, pad_token_id=0, dim=8, depth=1,
        num_heads=2, mixer="rrlsso", rank=4, dropout=0.0,
    )
    model = SequenceClassifier(encoder, 2)
    output = tmp_path / "validation-only"
    train_classifier(
        model, loader, loader, None, num_classes=2,
        config=TrainingConfig(output=str(output), epochs=1, seed=0),
        metadata={"suite": "test", "dataset": "tiny", "mixer": "rrlsso"},
    )
    assert (output / "validation_metrics.json").exists()
    assert not (output / "test_metrics.json").exists()


def test_summary_uses_only_complete_model_dataset_intersection():
    runs = [
        {"suite": "genomic", "dataset": "a", "model": "rrlsso", "accuracy": 0.8,
         "macro_f1": 0.7, "parameters": 10},
        {"suite": "genomic", "dataset": "a", "model": "rrlsso", "accuracy": 1.0,
         "macro_f1": 0.9, "parameters": 10},
        {"suite": "genomic", "dataset": "b", "model": "rrlsso", "accuracy": 0.6,
         "macro_f1": 0.5, "parameters": 10},
    ]
    summary = aggregate(runs)
    reported = [
        {"suite": "genomic", "dataset": "a", "model": "baseline", "accuracy": 0.7},
        {"suite": "genomic", "dataset": "b", "model": "baseline", "accuracy": 0.7},
    ]
    ranks = average_ranks(summary, reported)
    by_model = {row["model"]: row for row in ranks}
    assert by_model["rrlsso"]["complete_datasets"] == 2
    assert by_model["rrlsso"]["mean_rank"] == 1.5
    assert by_model["baseline"]["mean_rank"] == 1.5
