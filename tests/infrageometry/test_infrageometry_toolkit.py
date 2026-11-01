from frontier import REGISTRY

import infrageometry


def test_toolkit_registers_cleanly():
    assert "infrageometry" in REGISTRY.load_toolkits(["infrageometry"])
    errors = [i for i in REGISTRY.validate() if i.level == "error" and i.key.startswith("infrageometry.")]
    assert not errors, errors
    assert infrageometry.DIRECTION == "infrageometry"
