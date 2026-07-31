from __future__ import annotations

import pytest

import lsso
from lsso import CoreMode, LSSO, LSSOConfig
from lsso.ball import CoreMode as BallCoreMode
from lsso.ball import LSSO as BallLSSO
from lsso.ball import LSSOConfig as BallConfig


pytestmark = pytest.mark.core


def test_public_api_is_narrow() -> None:
    assert CoreMode is BallCoreMode
    assert LSSO is BallLSSO
    assert LSSOConfig is BallConfig
    assert set(lsso.__all__) == {
        "CoreMode",
        "LSSO",
        "LSSOConfig",
        "__version__",
    }
    assert lsso.__version__ == "0.6.1"
