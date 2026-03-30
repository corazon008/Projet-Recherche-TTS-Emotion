from models.baseline import Baseline


def test_baseline_model():
    base = Baseline()

    assert type(base) == Baseline
