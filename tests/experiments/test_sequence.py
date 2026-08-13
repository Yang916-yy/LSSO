from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import json
from pathlib import Path

import torch
import torch.nn as nn
import pytest
from torch.utils.data import Dataset, TensorDataset

import experiments.train_transformers as train_transformers
from experiments.sequence_data import (
    DatasetBundle,
    NucleotideTokenizer,
    PATHFINDER_SPLIT_PROTOCOL,
    TokenVocabulary,
    build_packed_tokens,
    collate_tokens,
    collate_values,
    pathfinder_split_indices,
    prepare_genomic_benchmarks,
    prepare_lra,
    stratified_split_indices,
    validate_formal_source_provenance,
)
from experiments.train_transformers import (
    SequenceClassifier,
    SequenceEncoder,
    SequencePairClassifier,
    TrainingConfig,
    _build_collate,
    _build_run_payload,
    _runtime_metadata,
    _is_better_validation_checkpoint,
    _makes_early_stop_progress,
    _validate_formal_data_source,
    build_model,
    parse_args,
    resolve_args,
    train,
)
from experiments.sequence_data import make_loader
from lsso import CoreMode
from lsso.ball import cuda


pytestmark = pytest.mark.experiment


def _encoder(mixer: str) -> SequenceEncoder:
    return SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=2,
        num_heads=2,
        rank=4,
        mixer=mixer,  # type: ignore[arg-type]
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )


def _grid_encoder(mixer: str) -> SequenceEncoder:
    return SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=6,
        dim=16,
        depth=2,
        num_heads=2,
        rank=4,
        mixer=mixer,  # type: ignore[arg-type]
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
        grid_shape=(2, 3),
    )


def test_learned_position_embedding_uses_small_normal_initialization() -> None:
    torch.manual_seed(31)
    weights = _encoder("mha").position_embedding.weight.detach()
    assert abs(float(weights.mean())) < 0.006
    assert float(weights.std(unbiased=False)) == pytest.approx(0.02, abs=0.004)


def test_factorized_grid_positions_use_flat_position_scale() -> None:
    torch.manual_seed(31)
    encoder = _grid_encoder("mha")
    assert encoder.position_embedding is None
    assert encoder.row_embedding is not None
    assert encoder.column_embedding is not None
    assert not any(isinstance(module, nn.Conv2d) for module in encoder.modules())
    coordinates = encoder._position_features(length=6, device=torch.device("cpu"))
    torch.testing.assert_close(
        coordinates[5], encoder.row_embedding.weight[1] + encoder.column_embedding.weight[2]
    )
    assert float(coordinates.detach().std(unbiased=False)) == pytest.approx(0.02, abs=0.006)


def test_factorized_grid_positions_reject_invalid_grids() -> None:
    with pytest.raises(ValueError, match="value inputs"):
        SequenceEncoder(
            input_kind="tokens",
            vocab_size=8,
            pad_token_id=0,
            max_length=6,
            dim=8,
            depth=1,
            num_heads=2,
            rank=4,
            mixer="mha",
            core_mode=CoreMode.DYNAMIC,
            rank_rotary=False,
            implementation="reference",
            mlp_ratio=2.0,
            dropout=0.0,
            bias=True,
            grid_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="exactly cover"):
        SequenceEncoder(
            input_kind="values",
            vocab_size=None,
            pad_token_id=None,
            max_length=8,
            dim=8,
            depth=1,
            num_heads=2,
            rank=4,
            mixer="mha",
            core_mode=CoreMode.DYNAMIC,
            rank_rotary=False,
            implementation="reference",
            mlp_ratio=2.0,
            dropout=0.0,
            bias=True,
            grid_shape=(2, 3),
        )
    with pytest.raises(ValueError, match="full 2x3 grid"):
        _grid_encoder("mha")(torch.ones(1, 5, 1), torch.ones(1, 5, dtype=torch.bool))


def test_pathx_length_is_covered_by_the_learned_position_table() -> None:
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=128**2,
        dim=8,
        depth=1,
        num_heads=2,
        rank=4,
        mixer="mha",
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=False,
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )
    last_position = torch.tensor([128**2 - 1])
    assert encoder.position_embedding(last_position).shape == (1, 8)


def test_pathx_full_length_reference_forward_and_backward_are_finite() -> None:
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=128**2,
        dim=8,
        depth=1,
        num_heads=2,
        rank=4,
        mixer="lsso",
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="reference",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    )
    inputs = torch.rand(1, 128**2, 1, requires_grad=True)
    valid_mask = torch.ones(1, 128**2, dtype=torch.bool)
    loss = encoder(inputs, valid_mask).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_byte_vocabulary_preserves_utf8_bytes_and_eos() -> None:
    vocabulary = TokenVocabulary.byte_level()
    encoded = vocabulary.encode_bytes("e\u0301", max_length=4)
    assert encoded == [
        vocabulary.lookup["e"],
        vocabulary.lookup[chr(0xCC)],
        vocabulary.lookup[chr(0x81)],
        vocabulary.eos_token_id,
    ]


def test_cuda_runtime_metadata_records_the_current_native_contract() -> None:
    metadata = _runtime_metadata(torch.device("cpu"), cuda_enabled=True)
    assert metadata["lsso_cuda_contract"] == cuda._NATIVE_CONTRACT_VERSION


def test_nucleotide_tokenizer_does_not_confuse_unknown_with_padding() -> None:
    tokenizer = NucleotideTokenizer()
    assert tokenizer.encode("ACGTNZ", 8).tolist() == [2, 3, 4, 5, 6, 1]
    with pytest.raises(ValueError, match="empty genomic"):
        tokenizer.encode("", 8)


def test_stratified_split_is_deterministic_and_keeps_every_class_in_train() -> None:
    labels = [0] * 8 + [1] * 4 + [2]
    train, validation = stratified_split_indices(labels, 0.25, 19)
    assert (train, validation) == stratified_split_indices(labels, 0.25, 19)
    assert set(labels[index] for index in train) == {0, 1, 2}
    assert not set(train).intersection(validation)
    assert sorted(train + validation) == list(range(len(labels)))


def test_pathfinder_split_is_deterministic_80_10_10() -> None:
    keys = [f"folder/file_{index}.png/{index}" for index in range(100)]
    train, validation, test = pathfinder_split_indices(keys)
    assert (len(train), len(validation), len(test)) == (80, 10, 10)
    assert (train, validation, test) == pathfinder_split_indices(keys)
    assert not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test))


def test_pathfinder_split_matches_tfds_hard_md5_order() -> None:
    keys = [f"sample-{index}" for index in range(10)]
    train, validation, test = pathfinder_split_indices(keys)
    assert train + validation + test == [7, 4, 2, 5, 0, 1, 3, 9, 6, 8]


def test_collate_uses_lengths_not_the_token_value() -> None:
    batch = collate_tokens(
        [(torch.tensor([0, 4]), 1), (torch.tensor([3]), 0)], pad_token_id=0
    )
    assert batch["mask"].tolist() == [[True, True], [True, False]]


@pytest.mark.parametrize("mixer", ("mha", "lsso"))
@pytest.mark.parametrize("pooling", ("mean", "meanmax"))
def test_sequence_classifier_masks_padding_for_both_mixers(
    mixer: str, pooling: str
) -> None:
    torch.manual_seed(4)
    model = SequenceClassifier(_encoder(mixer), 3, pooling=pooling).eval()  # type: ignore[arg-type]
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    first = torch.tensor([[2, 3, 0, 0], [4, 0, 0, 0]])
    second = torch.tensor([[2, 3, 10, 9], [4, 8, 7, 6]])
    torch.testing.assert_close(model(first, mask), model(second, mask), rtol=0, atol=0)


@pytest.mark.parametrize("mixer", ("mha", "lsso"))
def test_factorized_grid_positions_mask_invalid_pixels_for_both_mixers(mixer: str) -> None:
    torch.manual_seed(17)
    model = SequenceClassifier(_grid_encoder(mixer), 3, pooling="meanmax").eval()  # type: ignore[arg-type]
    valid = torch.tensor([[True, False, True, False, True, False]])
    first = torch.tensor([[[1.0], [0.0], [2.0], [0.0], [3.0], [0.0]]])
    second = torch.tensor([[[1.0], [8.0], [2.0], [7.0], [3.0], [6.0]]])
    torch.testing.assert_close(model(first, valid), model(second, valid), rtol=0, atol=0)


def test_meanmax_readout_handles_an_empty_valid_set() -> None:
    model = SequenceClassifier(_encoder("mha"), 3, pooling="meanmax")
    encoded = torch.randn(2, 3, 16)
    valid = torch.tensor([[False, False, False], [True, False, False]])
    pooled = model._pool(encoded, valid)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(pooled).all()
    assert model.meanmax_projection is not None
    assert model.readout_projection is not None


def test_mean_readout_keeps_the_existing_parameterization() -> None:
    model = SequenceClassifier(_encoder("mha"), 3)
    assert model.meanmax_projection is None
    assert model.readout_projection is None


def test_pair_classifier_accepts_independent_lengths() -> None:
    model = SequencePairClassifier(_encoder("lsso"), 2).eval()
    first = torch.tensor([[2, 3, 0], [4, 0, 0]])
    second = torch.tensor([[7, 6, 5], [3, 2, 0]])
    first_mask = first.ne(0)
    second_mask = second.ne(0)
    logits = model(first, first_mask, second, second_mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_meanmax_pooling_is_rejected_for_paired_tasks() -> None:
    args = resolve_args(
        parse_args(["--suite", "lra", "--task", "retrieval", "--pooling", "meanmax"])
    )
    dataset = TensorDataset(
        torch.ones(1, 4, dtype=torch.long), torch.zeros(1, dtype=torch.long)
    )
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="tokens",
        num_classes=2,
        max_length=4,
        metadata={},
        vocab_size=8,
        pad_token_id=0,
        paired=True,
    )
    with pytest.raises(ValueError, match="not supported for paired"):
        build_model(args, bundle)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mha_baseline_accepts_fp16_autocast_with_padding() -> None:
    model = SequenceClassifier(_encoder("mha"), 2).cuda().train()
    inputs = torch.tensor([[2, 3, 0], [4, 0, 0]], device="cuda")
    mask = inputs.ne(0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(inputs, mask).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_lsso_meanmax_readout_is_finite_with_masked_values() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="values",
        vocab_size=None,
        pad_token_id=None,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="lsso",
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
        grid_shape=(2, 4),
    )
    model = SequenceClassifier(encoder, 2, pooling="meanmax").cuda().train()
    inputs = torch.rand(2, 8, 1, device="cuda")
    valid = torch.tensor(
        [[True, True, False, False, False, False, False, False],
         [True, True, True, False, False, False, False, False]],
        device="cuda",
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(inputs, valid).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_lsso_sequence_path_receives_fp16_amp_activations() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="lsso",
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    ).cuda()
    observed: list[torch.dtype] = []
    hook = encoder.blocks[0].mixer.register_forward_pre_hook(
        lambda _module, arguments: observed.append(arguments[0].dtype)
    )
    inputs = torch.tensor([[2, 3, 0]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = encoder(inputs, inputs.ne(0))
    hook.remove()
    assert torch.isfinite(outputs).all()
    assert observed == [torch.float16]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_lsso_sequence_path_rejects_bf16_amp() -> None:
    cuda.load()
    encoder = SequenceEncoder(
        input_kind="tokens",
        vocab_size=12,
        pad_token_id=0,
        max_length=8,
        dim=16,
        depth=1,
        num_heads=2,
        rank=16,
        mixer="lsso",
        core_mode=CoreMode.DYNAMIC,
        rank_rotary=True,
        implementation="cuda",
        mlp_ratio=2.0,
        dropout=0.0,
        bias=True,
    ).cuda()
    inputs = torch.tensor([[2, 3, 0]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="FP16 AMP"):
            encoder(inputs, inputs.ne(0))


def test_genomic_folder_protocol_uses_official_test_without_resplitting(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split, records in {
        "train": {"negative": ["AC", "ACG"], "positive": ["TTTT", "G"]},
        "test": {"negative": ["A"], "positive": ["GGGGGG"]},
    }.items():
        for label, sequences in records.items():
            directory = task_root / split / label
            directory.mkdir(parents=True)
            for index, sequence in enumerate(sequences):
                (directory / f"{index}.txt").write_text(sequence, encoding="utf-8")
    bundle = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
    )
    assert bundle.max_length == 6
    assert len(bundle.train) + len(bundle.validation) == 4
    assert len(bundle.test) == 2
    assert bundle.metadata["split_protocol"] == "official-train-test-plus-stratified-validation-v1"


def test_formal_local_genomic_provenance_hashes_source_contents(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split, records in {
        "train": {"negative": ["AC", "ACG"], "positive": ["TT", "TTT"]},
        "test": {"negative": ["A"], "positive": ["G"]},
    }.items():
        for label, sequences in records.items():
            directory = task_root / split / label
            directory.mkdir(parents=True)
            for index, sequence in enumerate(sequences):
                (directory / f"{index}.txt").write_text(sequence, encoding="utf-8")

    first = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
        formal=True,
    )
    validate_formal_source_provenance(first)
    first_hash = first.metadata["source"]["content_sha256"]
    (task_root / "train" / "negative" / "0.txt").write_text("GG", encoding="utf-8")
    second = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
        formal=True,
    )
    assert second.metadata["source"]["content_sha256"] != first_hash


def test_formal_provenance_rejects_an_unhashed_local_source(tmp_path) -> None:
    task_root = tmp_path / "demo_human_or_worm"
    for split in ("train", "test"):
        for label, sequence in (("negative", "AC"), ("positive", "GT")):
            directory = task_root / split / label
            directory.mkdir(parents=True)
            (directory / "0.txt").write_text(sequence, encoding="utf-8")
            if split == "train":
                (directory / "1.txt").write_text(sequence + "A", encoding="utf-8")
    bundle = prepare_genomic_benchmarks(
        "demo_human_or_worm",
        data_root=tmp_path,
        max_length=0,
        validation_fraction=0.25,
        split_seed=7,
        allow_download=False,
        revision=None,
    )
    with pytest.raises(ValueError, match="content_sha256"):
        validate_formal_source_provenance(bundle)


def test_listops_cache_is_current_data_only(tmp_path) -> None:
    source = tmp_path / "listops"
    source.mkdir()
    for split in ("train", "val", "test"):
        (source / f"{split}.tsv").write_text(
            "Source\tTarget\n[ MAX 1 2 ]\t2\n[ MIN 9 3 ]\t3\n",
            encoding="utf-8",
        )
    bundle = prepare_lra(
        "listops",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=16,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
        formal=True,
    )
    assert len(bundle.train) == len(bundle.validation) == len(bundle.test) == 2
    tokens, label = bundle.train[0]
    assert tokens.dtype == torch.long
    assert label in (2, 3)
    assert bundle.metadata["tokenization"] == "official-listops-token-stream-with-eos"
    validate_formal_source_provenance(bundle)
    assert len(bundle.metadata["source"]["content_sha256"]) == 64


def test_packed_token_cache_serializes_a_concurrent_first_build(tmp_path) -> None:
    prefix = tmp_path / "cache" / "tokens"
    rows = [("first", 0), ("second", 1)]
    manifest = {"schema": 1, "case": "concurrent"}

    def build():
        return build_packed_tokens(
            prefix,
            rows,
            lambda text: [len(text)],
            manifest,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        datasets = [
            future.result()
            for future in (executor.submit(build), executor.submit(build))
        ]
    assert [len(dataset) for dataset in datasets] == [2, 2]
    assert not list(prefix.parent.glob("*.tmp"))


def test_retrieval_pair_cache_preserves_independent_documents(tmp_path) -> None:
    source = tmp_path / "aan"
    source.mkdir()
    for split, name in {
        "train": "new_aan_pairs.train.tsv",
        "val": "new_aan_pairs.eval.tsv",
        "test": "new_aan_pairs.test.tsv",
    }.items():
        del split
        (source / name).write_text(
            "1\tfirst\tsecond\tleft abstract\tright abstract\n",
            encoding="utf-8",
        )
    bundle = prepare_lra(
        "retrieval",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=16,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
    )
    first, second, label = bundle.train[0]
    assert bundle.paired
    assert first.dtype == second.dtype == torch.long
    assert first.numel() and second.numel()
    assert label == 1


def test_pathfinder_bundle_uses_value_tokens_and_official_split(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder2" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (2, 2), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    bundle = prepare_lra(
        "pathfinder",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=2,
        allow_download=False,
        revision=None,
    )
    values, label = bundle.train[0]
    assert bundle.input_kind == "values"
    assert bundle.value_masking == "nonzero"
    assert bundle.metadata["value_masking"] == "nonzero-pixels"
    assert bundle.max_length == 4
    assert values.shape == (4, 1)
    assert label in (0, 1)


def test_collate_values_can_exclude_zero_valued_pixels() -> None:
    batch = collate_values(
        [
            (torch.tensor([[0.0], [1.0], [0.0]]), 0),
            (torch.tensor([[2.0], [0.0]]), 1),
        ],
        value_masking="nonzero",
    )
    assert torch.equal(
        batch["mask"],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    assert batch["padding_ratio"] == pytest.approx(2.0 / 3.0)


def test_value_masking_contract_reaches_the_training_loader() -> None:
    dataset = TensorDataset(torch.zeros(1, 2, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=2,
        metadata={},
        value_masking="nonzero",
    )
    batch = _build_collate(bundle)([(torch.tensor([[0.0], [3.0]]), 1)])
    assert torch.equal(batch["mask"], torch.tensor([[False, True]]))


def test_pathfinder_manifest_is_scoped_to_its_resolution(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder2" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (2, 2), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "source-manifest.json").write_text(
        json.dumps(
            {
                "pathfinder32_hard": {
                    "metadata_files": 200,
                    "metadata_sha256": "32-only-signature",
                    "usable_rows": 200000,
                }
            }
        ),
        encoding="utf-8",
    )
    bundle = prepare_lra(
        "pathfinder",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=2,
        allow_download=False,
        revision=None,
    )
    assert bundle.metadata["resolution"] == 2


def test_pathx_is_fixed_to_pathfinder_128_with_its_own_identity(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    root = tmp_path / "pathfinder" / "pathfinder128" / "curv_contour_length_14"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    lines = []
    for index in range(10):
        folder = root / f"group_{index}"
        folder.mkdir()
        filename = f"sample_{index}.png"
        pil_image.new("L", (128, 128), color=index).save(folder / filename)
        lines.append(f"group_{index} {filename} unused {index % 2}")
    (metadata / "0.npy").write_text("\n".join(lines), encoding="utf-8")
    bundle = prepare_lra(
        "pathx",
        data_root=tmp_path,
        cache_root=tmp_path / "cache",
        max_length=None,
        validation_fraction=None,
        split_seed=None,
        pathfinder_resolution=None,
        allow_download=False,
        revision=None,
    )
    values, label = bundle.train[0]
    assert bundle.metadata["task"] == "pathx"
    assert bundle.metadata["resolution"] == 128
    assert bundle.metadata["split_protocol"] == PATHFINDER_SPLIT_PROTOCOL
    assert Path(bundle.metadata["source"]["path"]).name == "pathfinder128"
    assert (len(bundle.train), len(bundle.validation), len(bundle.test)) == (8, 1, 1)
    assert bundle.max_length == 128**2
    assert values.shape == (128**2, 1)
    assert label in (0, 1)


def test_resolved_pathx_delegates_its_fixed_resolution_to_the_data_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    marker = object()
    observed: dict[str, object] = {}

    def fake_prepare_lra(task: str, **kwargs: object) -> object:
        observed["task"] = task
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(train_transformers, "prepare_lra", fake_prepare_lra)
    assert train_transformers.build_bundle(args) is marker
    assert observed["task"] == "pathx"
    assert observed["pathfinder_resolution"] is None


def test_lra_defaults_preserve_the_requested_learning_rates_and_burn_in() -> None:
    pathfinder = resolve_args(parse_args(["--suite", "lra", "--task", "pathfinder"]))
    pathfinder_custom = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "pathfinder",
                "--pathfinder-resolution",
                "16",
            ]
        )
    )
    pathx = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    listops = resolve_args(parse_args(["--suite", "lra", "--task", "listops"]))
    text = resolve_args(parse_args(["--suite", "lra", "--task", "text"]))
    assert pathfinder.lr == 2e-4
    assert (
        pathfinder.dim,
        pathfinder.depth,
        pathfinder.heads,
        pathfinder.mlp_ratio,
        pathfinder.dropout,
        pathfinder.batch_size,
        pathfinder.eval_batch_size,
        pathfinder.grad_accum,
        pathfinder.pooling,
    ) == (
        256,
        6,
        4,
        2.0,
        0.0,
        64,
        256,
        2,
        "meanmax",
    )
    assert pathfinder.early_stop_min_epochs == 150
    assert pathfinder.pathfinder_resolution == 32
    assert pathfinder_custom.pathfinder_resolution == 16
    assert pathfinder.max_length is None
    assert pathfinder.validation_fraction is None
    assert pathfinder.split_seed is None
    assert pathx.pathfinder_resolution == 128
    assert pathx.max_length is None
    assert (pathx.dim, pathx.depth, pathx.batch_size, pathx.grad_accum) == (128, 6, 12, 10)
    assert listops.lr == 5e-4
    assert (
        listops.dim,
        listops.depth,
        listops.batch_size,
        listops.eval_batch_size,
        listops.grad_accum,
        listops.mlp_ratio,
        listops.pooling,
    ) == (
        128,
        6,
        25,
        25,
        2,
        4.0,
        "mean",
    )
    assert (
        text.dim,
        text.depth,
        text.batch_size,
        text.eval_batch_size,
        text.grad_accum,
    ) == (256, 6, 16, 16, 2)
    assert listops.early_stop_min_epochs == 30


def test_pilot_epochs_preserves_the_scheduler_horizon_and_is_not_formal() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "pathfinder",
                "--epochs",
                "200",
                "--pilot-epochs",
                "50",
                "--validation-only",
            ]
        )
    )
    assert (args.epochs, args.pilot_epochs) == (200, 50)
    with pytest.raises(ValueError, match="requires --validation-only"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--pilot-epochs",
                    "1",
                ]
            )
        )
    with pytest.raises(ValueError, match="rejects --pilot-epochs"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--pilot-epochs",
                    "1",
                    "--validation-only",
                    "--formal",
                ]
            )
        )
    with pytest.raises(ValueError, match="must not exceed epochs"):
        resolve_args(
            parse_args(
                [
                    "--suite",
                    "lra",
                    "--task",
                    "pathfinder",
                    "--epochs",
                    "2",
                    "--pilot-epochs",
                    "3",
                    "--validation-only",
                ]
            )
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--max-length", "1024"), "--max-length is not supported for LRA Pathfinder"),
        (
            ("--validation-fraction", "0.2"),
            "--validation-fraction is not supported for LRA Pathfinder",
        ),
        (("--split-seed", "7"), "--split-seed is not supported for LRA Pathfinder"),
    ),
)
def test_pathfinder_rejects_inactive_data_options(
    arguments: tuple[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_args(
            parse_args(["--suite", "lra", "--task", "pathfinder", *arguments])
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--max-length", "1024"), "--max-length is not supported for LRA Path-X"),
        (
            ("--pathfinder-resolution", "32"),
            "--pathfinder-resolution is not supported for LRA Path-X",
        ),
        (
            ("--validation-fraction", "0.2"),
            "--validation-fraction is not supported for LRA Path-X",
        ),
        (("--split-seed", "7"), "--split-seed is not supported for LRA Path-X"),
    ),
)
def test_pathx_rejects_data_overrides(
    arguments: tuple[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_args(parse_args(["--suite", "lra", "--task", "pathx", *arguments]))


def test_pathfinder_metadata_excludes_inactive_data_options() -> None:
    args = resolve_args(
        parse_args(["--suite", "lra", "--task", "pathfinder", "--pooling", "meanmax"])
    )
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=32**2,
        metadata={},
    )
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        torch.nn.Linear(1, 1),
        torch.device("cpu"),
        cuda_enabled=False,
    )
    resolved = payload["resolved_arguments"]
    assert resolved["pathfinder_resolution"] == 32
    assert resolved["pooling"] == "meanmax"
    assert payload["model"]["pooling"] == "meanmax"
    assert {"max_length", "validation_fraction", "split_seed"}.isdisjoint(resolved)


def test_pathfinder_model_uses_factorized_grid_positions_and_records_its_contract() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite", "lra", "--task", "pathfinder", "--pathfinder-resolution", "2",
                "--mixer", "mha",
            ]
        )
    )
    dataset = TensorDataset(torch.zeros(1, 4, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=4,
        metadata={"resolution": 2},
    )
    model = build_model(args, bundle)
    assert isinstance(model, SequenceClassifier)
    assert model.encoder.grid_shape == (2, 2)
    assert model.encoder.position_embedding is None
    assert model.encoder.row_embedding is not None
    assert model.encoder.column_embedding is not None
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        model,
        torch.device("cpu"),
        cuda_enabled=False,
    )
    assert payload["model"]["mlp_ratio"] == 2.0
    assert payload["model"]["dropout"] == 0.0
    assert payload["model"]["bias"] is True
    assert payload["model"]["position_encoding"] == "factorized-grid-absolute"
    assert payload["model"]["position_initialization"] == "factorized-normal-0.02"


def test_pathx_run_metadata_keeps_the_fixed_task_identity() -> None:
    args = resolve_args(parse_args(["--suite", "lra", "--task", "pathx"]))
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    bundle = DatasetBundle(
        train=dataset,
        validation=dataset,
        test=dataset,
        input_kind="values",
        num_classes=2,
        max_length=128**2,
        metadata={"task": "pathx", "resolution": 128},
    )
    payload = _build_run_payload(
        args,
        bundle,
        {"train": 1, "validation": 1, "test": 1},
        torch.nn.Linear(1, 1),
        torch.device("cpu"),
        cuda_enabled=False,
    )
    resolved = payload["resolved_arguments"]
    assert resolved["task"] == "pathx"
    assert resolved["pathfinder_resolution"] == 128
    assert {"max_length", "validation_fraction", "split_seed"}.isdisjoint(resolved)
    assert payload["data_contract"]["value_masking"] == "length"
    assert payload["model"]["position_initialization"] == "normal-0.02"


def test_formal_download_requires_an_immutable_revision() -> None:
    args = resolve_args(
        parse_args(
            [
                "--suite",
                "lra",
                "--task",
                "text",
                "--formal",
                "--allow-download",
            ]
        )
    )
    with pytest.raises(ValueError, match="immutable full-SHA"):
        _validate_formal_data_source(args)
    args.data_revision = "release-1"
    with pytest.raises(ValueError, match="immutable full-SHA"):
        _validate_formal_data_source(args)
    args.data_revision = "0123456789abcdef0123456789abcdef01234567"
    _validate_formal_data_source(args)


def test_checkpoint_selection_is_independent_of_early_stop_tolerances() -> None:
    assert _is_better_validation_checkpoint(0.7001, 0.9, 0.7, 0.9)
    assert _is_better_validation_checkpoint(0.7, 0.89, 0.7, 0.9)
    assert _makes_early_stop_progress(
        0.7001,
        0.9,
        0.7,
        0.9,
        accuracy_delta=0.01,
        loss_relative_delta=0.01,
    ) == (False, False)


def test_shared_lra_config_resolves_the_selected_task_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    pathfinder = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/lra.toml"),
                "--task",
                "pathfinder",
            ]
        )
    )
    assert (
        pathfinder.lr,
        pathfinder.epochs,
        pathfinder.dim,
        pathfinder.batch_size,
        pathfinder.grad_accum,
        pathfinder.pooling,
    ) == (
        2e-4,
        200,
        256,
        64,
        2,
        "meanmax",
    )
    retrieval = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/lra.toml"),
                "--task",
                "retrieval",
            ]
        )
    )
    assert (retrieval.dim, retrieval.depth, retrieval.batch_size, retrieval.grad_accum) == (
        128,
        6,
        16,
        4,
    )


def test_sequence_toml_template_supplies_required_task_selection() -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(parse_args(["--config", str(root / "experiments/configs/lra.toml")]))
    assert (args.suite, args.task, args.rank, args.lr) == ("lra", "listops", 32, 5e-4)


def test_genomic_toml_template_matches_the_formal_rank() -> None:
    root = Path(__file__).resolve().parents[2]
    args = resolve_args(
        parse_args(["--config", str(root / "experiments/configs/genomic.toml")])
    )
    assert (args.suite, args.dim, args.depth, args.heads, args.rank) == (
        "genomic",
        128,
        4,
        4,
        32,
    )
    assert (args.batch_size, args.grad_accum) == (64, 2)

    ordinary = resolve_args(
        parse_args(
            [
                "--config",
                str(root / "experiments/configs/genomic.toml"),
                "--task",
                "demo_human_or_worm",
            ]
        )
    )
    assert (ordinary.batch_size, ordinary.grad_accum, ordinary.rank) == (128, 1, 32)


class _TinyTokenDataset(Dataset):
    labels = [0, 1, 0, 1]
    lengths = [3, 2, 3, 2]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        values = (torch.tensor([2, 3, 4]), torch.tensor([3, 4]))[index % 2]
        return values, self.labels[index]


def test_pilot_epochs_caps_the_outer_loop_without_shortening_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=17,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=17,
    )
    recorded_total_steps: list[int] = []
    scheduler_lambda = train_transformers._scheduler_lambda

    def record_scheduler_horizon(
        step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float
    ) -> float:
        recorded_total_steps.append(total_steps)
        return scheduler_lambda(step, warmup_steps, total_steps, min_lr_ratio)

    monkeypatch.setattr(train_transformers, "_scheduler_lambda", record_scheduler_horizon)
    config = TrainingConfig(
        output=tmp_path / "run",
        epochs=4,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=4,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=17,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
        pilot_epochs=2,
    )
    result = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={"case": "pilot"},
        device=device,
    )
    metrics = (config.output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 2
    assert set(recorded_total_steps) == {len(loader) * config.epochs}
    assert result["target"] == "validation"


def test_checkpoint_refuses_configuration_mismatch_without_rewriting_manifest(tmp_path) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=3,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=3,
    )
    output = tmp_path / "run"
    config = TrainingConfig(
        output=output,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=0,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=3,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
    )
    train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={"case": "first"},
        device=device,
    )
    manifest = (output / "config.json").read_text(encoding="utf-8")
    mismatch = TrainingConfig(**{**config.__dict__, "resume": True})
    with pytest.raises(RuntimeError, match="configuration does not match"):
        train(
            SequenceClassifier(_encoder("mha"), 2),
            loader,
            validation,
            validation,
            num_classes=2,
            config=mismatch,
            run_payload={"case": "different"},
            device=device,
        )
    assert (output / "config.json").read_text(encoding="utf-8") == manifest


def test_checkpoint_allows_explicit_resume_for_the_same_run(tmp_path) -> None:
    dataset = _TinyTokenDataset()
    device = torch.device("cpu")
    loader = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=True,
        seed=5,
    )
    validation = make_loader(
        dataset,
        batch_size=2,
        workers=0,
        device=device,
        collate_fn=functools.partial(collate_tokens, pad_token_id=0),
        train=False,
        seed=5,
    )
    config = TrainingConfig(
        output=tmp_path / "run",
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        min_lr_ratio=0.0,
        grad_accum=1,
        grad_clip=0.0,
        patience=0,
        early_stop_min_epochs=0,
        early_stop_accuracy_delta=0.0,
        early_stop_loss_relative_delta=0.0,
        seed=5,
        resume=False,
        validation_only=True,
        max_train_batches=0,
        max_eval_batches=0,
        amp=False,
    )
    first = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=config,
        run_payload={
            "case": "resume",
            "resolved_arguments": {"resume": False, "task": "test"},
        },
        device=device,
    )
    metrics_before = (config.output / "metrics.jsonl").read_text(encoding="utf-8")
    manifest_before = (config.output / "config.json").read_text(encoding="utf-8")
    result = train(
        SequenceClassifier(_encoder("mha"), 2),
        loader,
        validation,
        validation,
        num_classes=2,
        config=TrainingConfig(**{**config.__dict__, "resume": True}),
        run_payload={
            "case": "resume",
            "resolved_arguments": {"resume": True, "task": "test"},
        },
        device=device,
    )
    assert result == first
    assert (config.output / "metrics.jsonl").read_text(encoding="utf-8") == metrics_before
    assert (config.output / "config.json").read_text(encoding="utf-8") == manifest_before
    state = torch.load(config.output / "last.pt", map_location="cpu", weights_only=False)
    assert state["completed"]
    assert state["format_version"] == 2
    assert "early_stop_best_accuracy" in state
