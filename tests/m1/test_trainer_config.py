import pytest

from trace_bench.runner import _normalize_trainers


def test_trainer_unknown_key_raises():
    with pytest.raises(ValueError) as exc:
        _normalize_trainers([{"name": "PrioritySearch", "unknown": 1}], {})
    assert "unknown keys" in str(exc.value).lower()
