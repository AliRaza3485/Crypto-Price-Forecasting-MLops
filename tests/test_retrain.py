from src.models import retrain

# NOTE: retrain.py (like model_training.py) imports mlflow LAZILY -- only
# inside promote()/log_rejection(), not at module level. That means this
# whole module, and every pure function tested below (decide_promotion,
# should_retrain, restart_serving), imports and runs fine in the slim
# requirements-ci.txt env with NO mlflow installed -- verified directly.
# promote()/log_rejection() themselves aren't covered here since they need
# real mlflow; that's an accepted gap, not the reason to skip everything
# else. Previously this entire file was skipped in CI via
# `pytest.importorskip("mlflow")`, which meant NONE of these tests ever
# actually ran in GitHub Actions -- only locally, with the full training
# env installed. Removing that guard is what makes CI's green checkmark
# mean something for this file.


def make_metrics(rmse: float) -> dict:
    """Build a metrics dict shaped like evaluate()'s output; only rmse varies
    in these tests, the rest are placeholders.
    """
    return {"rmse": rmse, "mae": 0.0, "r2": 0.0, "directional_acc": 0.0}


# --- decide_promotion ---------------------------------------------------


def test_decide_promotion_no_champion_promotes():
    # First-ever run: nothing to lose to, so promote unconditionally.
    assert retrain.decide_promotion(make_metrics(1.0), None, margin=0.0) is True


def test_decide_promotion_candidate_strictly_better():
    cand = make_metrics(0.8)
    champ = make_metrics(1.0)
    assert retrain.decide_promotion(cand, champ, margin=0.0) is True


def test_decide_promotion_candidate_worse():
    cand = make_metrics(1.2)
    champ = make_metrics(1.0)
    assert retrain.decide_promotion(cand, champ, margin=0.0) is False


def test_decide_promotion_equal_rmse_zero_margin_promotes():
    # <= boundary: equal RMSE with margin 0.0 still counts as "beats" the champion.
    cand = make_metrics(1.0)
    champ = make_metrics(1.0)
    assert retrain.decide_promotion(cand, champ, margin=0.0) is True


def test_decide_promotion_better_but_less_than_margin_rejected():
    # Candidate improves by 0.05, but the required margin is 0.1 -> not enough.
    cand = make_metrics(0.95)
    champ = make_metrics(1.0)
    assert retrain.decide_promotion(cand, champ, margin=0.1) is False


def test_decide_promotion_better_by_exactly_margin_promotes():
    # Candidate improves by exactly the margin -> boundary case, should promote.
    cand = make_metrics(0.9)
    champ = make_metrics(1.0)
    assert retrain.decide_promotion(cand, champ, margin=0.1) is True


# --- should_retrain -------------------------------------------------------


def _forbid_network_calls(monkeypatch):
    """Monkeypatch the network-facing drift functions to blow up if called,
    so a passing test proves the "no network" shortcuts really skip them.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("get_current_features/compute_drift should not be called")

    monkeypatch.setattr("src.models.retrain.get_current_features", _boom)
    monkeypatch.setattr("src.models.retrain.compute_drift", _boom)


def test_should_retrain_force_skips_drift_check(monkeypatch):
    _forbid_network_calls(monkeypatch)

    do_retrain, report = retrain.should_retrain(force=True)

    assert (do_retrain, report) == (True, None)


def test_should_retrain_drift_gate_off_skips_drift_check(monkeypatch):
    monkeypatch.setattr("src.models.retrain.DRIFT_GATE", False)
    _forbid_network_calls(monkeypatch)

    do_retrain, report = retrain.should_retrain(force=False)

    assert (do_retrain, report) == (True, None)


def test_should_retrain_gate_on_drift_detected(monkeypatch):
    monkeypatch.setattr("src.models.retrain.DRIFT_GATE", True)
    fake_report = {
        "drift_detected": True,
        "n_drifted": 4,
        "n_features": 14,
        "features": [],
    }

    monkeypatch.setattr("src.models.retrain.get_current_features", lambda: object())
    monkeypatch.setattr("src.models.retrain.compute_drift", lambda current: fake_report)
    monkeypatch.setattr("src.models.retrain.format_report", lambda r: "")

    do_retrain, report = retrain.should_retrain(force=False)

    assert (do_retrain, report) == (True, fake_report)


def test_should_retrain_gate_on_no_drift(monkeypatch):
    monkeypatch.setattr("src.models.retrain.DRIFT_GATE", True)
    fake_report = {
        "drift_detected": False,
        "n_drifted": 0,
        "n_features": 14,
        "features": [],
    }

    monkeypatch.setattr("src.models.retrain.get_current_features", lambda: object())
    monkeypatch.setattr("src.models.retrain.compute_drift", lambda current: fake_report)
    monkeypatch.setattr("src.models.retrain.format_report", lambda r: "")

    do_retrain, report = retrain.should_retrain(force=False)

    assert (do_retrain, report) == (False, fake_report)


# --- restart_serving --------------------------------------------------------


class _FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def test_restart_serving_success_does_not_raise(monkeypatch):
    monkeypatch.setattr(
        "src.models.retrain.subprocess.run",
        lambda *args, **kwargs: _FakeCompletedProcess(returncode=0),
    )

    retrain.restart_serving()  # completing without raising is the assertion


def test_restart_serving_failure_does_not_raise(monkeypatch):
    # Non-fatal by design: a failed restart just logs a warning, the new
    # model is already saved -- restart_serving must never raise.
    monkeypatch.setattr(
        "src.models.retrain.subprocess.run",
        lambda *args, **kwargs: _FakeCompletedProcess(returncode=1, stderr="boom"),
    )

    retrain.restart_serving()  # completing without raising is the assertion
