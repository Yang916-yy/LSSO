import json

from tools.kaggle_imagenet_tree_to_wds import (
    build_part,
    class_bounds,
    load_local_labels,
    locate_imagenet,
)


def _write_jpeg(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_partition_bounds_cover_classes_once():
    bounds = [class_bounds(part, 3, 10) for part in range(3)]
    assert bounds == [(0, 3), (3, 6), (6, 10)]
    assert [index for start, end in bounds for index in range(start, end)] == list(range(10))


def test_locate_flat_kaggle_imagenet_tree(tmp_path):
    flat = tmp_path / "mounted-dataset" / "ILSVRC2012"
    for index in range(1000):
        (flat / f"n{index:08d}").mkdir(parents=True)
    data_root, metadata_root = locate_imagenet(tmp_path)
    assert data_root == flat
    assert metadata_root is None
    synsets, validation = load_local_labels(metadata_root, data_root)
    assert synsets == [f"n{index:08d}" for index in range(1000)]
    assert validation == {}


def test_build_train_and_validation_parts_with_unique_shard_names(tmp_path):
    data = tmp_path / "ILSVRC" / "Data" / "CLS-LOC"
    synsets = ["n00000001", "n00000002", "n00000003", "n00000004"]
    for index, synset in enumerate(synsets):
        _write_jpeg(data / "train" / synset / f"{synset}_1.JPEG", f"jpeg-{index}".encode())
    validation = {
        "ILSVRC2012_val_00000001": 1,
        "ILSVRC2012_val_00000002": 3,
    }
    for key in validation:
        _write_jpeg(data / "val" / f"{key}.JPEG", key.encode())

    train_output = tmp_path / "part-00"
    train = build_part(
        data,
        train_output,
        part_id=0,
        num_train_parts=2,
        synsets=synsets,
        validation_labels=validation,
        maxcount=1,
    )
    assert train["samples"] == 2
    assert train["class_index_range"] == [0, 2]
    assert [item["name"] for item in train["shards"]] == [
        "imagenet1k-train-p00-00000.tar",
        "imagenet1k-train-p00-00001.tar",
    ]
    assert all(len(item["sha256"]) == 64 for item in train["shards"])
    assert json.loads((train_output / "manifest.json").read_text()) == train
    assert build_part(
        data,
        train_output,
        part_id=0,
        num_train_parts=2,
        synsets=synsets,
        validation_labels=validation,
        maxcount=1,
    ) == train

    val_output = tmp_path / "part-02"
    val = build_part(
        data,
        val_output,
        part_id=2,
        num_train_parts=2,
        synsets=synsets,
        validation_labels=validation,
        maxcount=8,
    )
    assert val["split"] == "validation"
    assert val["samples"] == 2
    assert [item["name"] for item in val["shards"]] == [
        "imagenet1k-validation-val-00000.tar"
    ]
