import hypermatrix
from frontier import REGISTRY


def test_toolkit_registers_cleanly():
    assert "hypermatrix" in REGISTRY.load_toolkits(["hypermatrix"])
    errors = [i for i in REGISTRY.validate() if i.level == "error" and i.key.startswith("hypermatrix.")]
    assert not errors, errors
    assert hypermatrix.DIRECTION == "hypermatrix"
