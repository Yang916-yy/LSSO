import lsso
import lsso.backends
import lsso.nn
import lsso.ops


def test_public_namespaces_export_supported_symbols() -> None:
    assert lsso.LSSO is lsso.nn.LSSO
    assert lsso.RRLSSO is lsso.nn.RRLSSO
    assert lsso.lsso is lsso.ops.lsso
    assert callable(lsso.backends.is_available)


def test_package_version_matches_project_metadata() -> None:
    assert lsso.__version__ == "0.2.0.dev0"
