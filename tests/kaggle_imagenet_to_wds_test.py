import gzip
import io
import json
import tarfile

import webdataset as wds

from experiments.imagenet_wds_train import local_webdataset
from tools.kaggle_imagenet_to_wds import convert_tar_stream


def _source_archive() -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            members = {
                "ILSVRC/Data/CLS-LOC/train/n00000001/n00000001_1.JPEG": b"train-1",
                "ILSVRC/Annotations/CLS-LOC/train/n00000001/n00000001_1.xml": b"skip",
                "ILSVRC/Data/CLS-LOC/train/n00000002/n00000002_1.JPEG": b"train-2",
                "ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG": b"val-1",
                "ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000001.JPEG": b"skip",
            }
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _samples(pattern):
    return list(wds.WebDataset(str(pattern), shardshuffle=False).decode().to_tuple("jpg", "cls"))


def test_stream_conversion_filters_labels_shards_and_resumes(tmp_path):
    payload = _source_archive()
    classes = {"n00000001": 0, "n00000002": 1}
    validation = {"ILSVRC2012_val_00000001": 7}

    counts = convert_tar_stream(
        io.BytesIO(payload), tmp_path, classes, validation, maxcount=1
    )
    assert counts == (2, 1)
    train = _samples(tmp_path / "imagenet1k-train-{00000..00001}.tar")
    val = _samples(tmp_path / "imagenet1k-validation-00000.tar")
    assert train == [(b"train-1", 0), (b"train-2", 1)]
    assert val == [(b"val-1", 7)]
    assert len(list(tmp_path.glob("*.complete.json"))) == 3
    assert not list(tmp_path.glob("*.partial"))

    # A restarted conversion scans the source but does not rewrite committed shards.
    before = {path: path.stat().st_mtime_ns for path in tmp_path.glob("*.tar")}
    assert convert_tar_stream(
        io.BytesIO(payload), tmp_path, classes, validation, maxcount=1
    ) == (2, 1)
    after = {path: path.stat().st_mtime_ns for path in tmp_path.glob("*.tar")}
    assert before == after
    for marker in tmp_path.glob("*.complete.json"):
        assert json.loads(marker.read_text())["samples"] == 1

    local = local_webdataset(
        sorted(str(path) for path in tmp_path.glob("imagenet1k-train-*.tar")),
        shardshuffle=False,
        seed=0,
    ).compose(wds.to_tuple("jpg", "cls"), wds.map_tuple(bytes, int))
    assert list(local) == [(b"train-1", 0), (b"train-2", 1)]
