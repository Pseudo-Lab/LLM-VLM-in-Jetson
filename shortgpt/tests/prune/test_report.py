from src.prune.report import format_prune_report


def test_format_prune_report_contains_key_numbers():
    info = {"params_before": 1000, "params_after": 700, "ratio_actual": 0.30, "ratio_target": 0.30}
    text = format_prune_report(info, bits=4)
    assert "1,000" in text
    assert "700" in text
    assert "30" in text          # 감축률 %
    assert "GB" in text          # 메모리 추정 포함
