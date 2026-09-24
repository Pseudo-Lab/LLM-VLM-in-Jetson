from src.common import eval_kmmlu


def test_extract_accuracy_parses_kmmlu_results():
    fake_results = {"results": {"kmmlu": {"acc,none": 0.4123, "acc_stderr,none": 0.01}}}
    acc = eval_kmmlu.extract_accuracy(fake_results, task="kmmlu")
    assert abs(acc - 0.4123) < 1e-6


def test_run_kmmlu_uses_simple_evaluate(monkeypatch):
    captured = {}

    def fake_simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {"results": {"kmmlu": {"acc,none": 0.5}}}

    monkeypatch.setattr(eval_kmmlu, "simple_evaluate", fake_simple_evaluate)
    acc = eval_kmmlu.run_kmmlu("dummy/path", limit=8, device="cpu")
    assert acc == 0.5
    assert captured["tasks"] == ["kmmlu"]
    assert captured["limit"] == 8
