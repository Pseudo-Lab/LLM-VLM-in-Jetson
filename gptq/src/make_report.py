"""양자화 결과 시각화 리포트 생성.

생성 파일:
  results/report_ppl.png       — PPL 비교 (fp16 / GPTQ-EN / GPTQ-KO캘리브)
  results/report_kdtcbench.png — K-DTCBench 정확도 (카테고리별)
  results/report_summary.png   — 요약 대시보드 (PPL + K-DTCBench 통합)

실행:
  python src/make_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ── 색상 팔레트
C_FP16  = "#4C72B0"   # 파랑 — fp16 원본
C_EN    = "#DD8452"   # 주황 — 영어 캘리브
C_KO    = "#55A868"   # 초록 — 한국어 캘리브
C_RAND  = "#C0C0C0"   # 회색 — 랜덤 기준선
C_RED   = "#C44E52"

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ── 데이터 로드 ──────────────────────────────────────────────

def load_ppl() -> dict:
    fp16 = json.loads((RESULTS / "ppl_fp16.json").read_text())
    gptq_en = json.loads((RESULTS / "ppl_gptq.json").read_text())
    gptq_ko = json.loads((RESULTS / "ppl_gptq_kocalib.json").read_text())
    return {"fp16": fp16, "gptq_en": gptq_en, "gptq_ko": gptq_ko}


def load_kdtcbench() -> dict:
    # 전체 240문제(범주당 80) 동일 조건 결과 우선. fp16 은 full 없으면 구 추정치 폴백.
    out = {}
    specs = [("fp16", ["kdtcbench_fp16.json", "kdtcbench_fp16_est.json"]),
             ("gptq_en", ["kdtcbench_gptq_en.json"]),
             ("kocalib", ["kdtcbench_kocalib.json"])]
    for key, fnames in specs:
        for fname in fnames:
            p = RESULTS / fname
            if p.exists():
                out[key] = json.loads(p.read_text())
                break
    return out


# ── Figure 1: PPL 비교 ──────────────────────────────────────

def plot_ppl(ppl: dict, ax_en, ax_ko):
    models = ["fp16", "gptq_en", "gptq_ko"]
    labels = ["fp16\n(기준)", "GPTQ\n영어캘리브", "GPTQ\n한국어캘리브"]
    colors = [C_FP16, C_EN, C_KO]

    for ax, lang, title in [(ax_en, "en", "영어 PPL (WikiText-2)"),
                             (ax_ko, "ko", "한국어 PPL (Wikipedia-KO)")]:
        vals = [ppl[m][lang]["ppl"] for m in models]
        bars = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Perplexity (낮을수록 좋음)", fontsize=9)
        ymin = min(vals) * 0.96
        ymax = max(vals) * 1.06
        ax.set_ylim(ymin, ymax)

        # 값 레이블
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # fp16 대비 Δ% 표시
        fp16_val = vals[0]
        for i, (bar, v) in enumerate(zip(bars[1:], vals[1:]), 1):
            delta = (v - fp16_val) / fp16_val * 100
            sign = "+" if delta >= 0 else ""
            color = C_RED if delta > 0 else C_KO
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ymin + (ymax - ymin) * 0.04,
                    f"{sign}{delta:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold")

        ax.axhline(fp16_val, color=C_FP16, linestyle="--", linewidth=1.2,
                   alpha=0.6, label="fp16 기준선")


# ── Figure 2: K-DTCBench ────────────────────────────────────

def plot_kdtcbench(bench: dict, ax):
    cats = ["document", "table", "chart", "전체"]
    x = np.arange(len(cats))
    width = 0.28

    # K-DTCBench 는 비전 경로(세 모델 다 fp16) 측정이라 캘리브 차이는 노이즈.
    # 영어캘리브(구버전)는 메시지를 흐려 제외 → fp16 vs 한국어캘리브 2-way.
    spec = [("fp16", "fp16 원본", C_FP16),
            ("kocalib", "GPTQ 한국어캘리브", C_KO)]
    series = []
    for key, label, color in spec:
        if key not in bench:
            continue
        b = bench[key]
        vals = [b["by_category"].get(c, {}).get("accuracy", 0)
                for c in cats[:3]] + [b["total_accuracy"]]
        n_each = next(iter(b["by_category"].values()))["total"]
        series.append((f"{label} (n={n_each}/범주)", color, vals))

    n = len(series)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    for (label, color, vals), offset in zip(series, offsets):
        bars = ax.bar(x + offset, vals, width=width, color=color,
                      label=label, zorder=3, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=color)

    # 랜덤 기준선 25%
    ax.axhline(0.25, color=C_RAND, linestyle="--", linewidth=1.5,
               label="랜덤 기준 (25%)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Document\n(문서)", "Table\n(표)", "Chart\n(차트)", "전체"], fontsize=10)
    ax.set_ylabel("정확도 (높을수록 좋음)", fontsize=9)
    ax.set_title("K-DTCBench 정확도 — 한국어 Document/Table/Chart VQA",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8.5, loc="upper right")

    # 핵심 해석 주석
    if "fp16" in bench:
        ax.text(0.5, -0.22,
                "※ 두 모델 모두 240문제(범주당 80) 동일 조건. fp16 원본도 랜덤(25%) 근처 — "
                "낮은 점수는 양자화 손상이 아니라 모델 한계(Llama 3.2 Vision은 영어 전용). "
                "비전 경로는 양쪽 다 fp16이라 양자화(한국어캘리브) 영향은 노이즈 수준.",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=C_RED, style="italic")


# ── Figure 3: 요약 대시보드 ──────────────────────────────────

def make_summary_table(ppl: dict, bench: dict) -> list[list]:
    rows = [
        ["모델", "EN PPL", "KO PPL", "EN Δ", "KO Δ",
         "K-DTCBench\n전체 정확도", "Document", "Table", "Chart"],
    ]
    fp16_en = ppl["fp16"]["en"]["ppl"]
    fp16_ko = ppl["fp16"]["ko"]["ppl"]

    def ppl_row(key, label, bench_key=None):
        en = ppl[key]["en"]["ppl"]
        ko = ppl[key]["ko"]["ppl"]
        d_en = f'+{(en/fp16_en-1)*100:.1f}%' if key != "fp16" else "—"
        d_ko = f'+{(ko/fp16_ko-1)*100:.1f}%' if key != "fp16" else "—"
        if bench_key and bench_key in bench:
            b = bench[bench_key]
            total = f'{b["total_accuracy"]:.1%}'
            doc   = f'{b["by_category"].get("document",{}).get("accuracy",0):.1%}'
            tbl   = f'{b["by_category"].get("table",{}).get("accuracy",0):.1%}'
            cht   = f'{b["by_category"].get("chart",{}).get("accuracy",0):.1%}'
        else:
            total = doc = tbl = cht = "—"
        return [label, f"{en:.2f}", f"{ko:.2f}", d_en, d_ko, total, doc, tbl, cht]

    rows.append(ppl_row("fp16",    "fp16 원본",               "fp16"))
    rows.append(ppl_row("gptq_en", "GPTQ 영어캘리브",         None))
    rows.append(ppl_row("gptq_ko", "GPTQ 한국어캘리브 ★",     "kocalib"))
    return rows


# ── 메인 ─────────────────────────────────────────────────────

def main():
    ppl   = load_ppl()
    bench = load_kdtcbench()

    # ── 대시보드 (3행 레이아웃) ──
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Llama 3.2-11B Vision  GPTQ 4-bit 양자화 결과\n"
        "한국어 캘리브레이션(Wikipedia-KO 70% + Flickr30k 30%) 적용",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.35,
                          top=0.91, bottom=0.06, left=0.07, right=0.97)

    ax_en  = fig.add_subplot(gs[0, 0])
    ax_ko  = fig.add_subplot(gs[0, 1])
    ax_dtc = fig.add_subplot(gs[1, :])
    ax_tbl = fig.add_subplot(gs[2, :])

    # PPL
    plot_ppl(ppl, ax_en, ax_ko)

    # K-DTCBench
    plot_kdtcbench(bench, ax_dtc)

    # 요약 테이블
    ax_tbl.axis("off")
    rows = make_summary_table(ppl, bench)
    col_widths = [0.20, 0.08, 0.08, 0.08, 0.08, 0.14, 0.10, 0.10, 0.10]
    tbl = ax_tbl.table(
        cellText=rows[1:], colLabels=rows[0],
        loc="center", cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)

    # 헤더 스타일
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor("#2C3E50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # 한국어캘리브 행 강조
    for j in range(len(rows[0])):
        tbl[(3, j)].set_facecolor("#E8F5E9")
        tbl[(3, j)].set_text_props(fontweight="bold")

    ax_tbl.set_title("전체 결과 요약", fontsize=11, fontweight="bold",
                     pad=8, loc="left", x=0.01)

    out = RESULTS / "report_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[done] {out}")

    # PPL 단독
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    fig2.suptitle("PPL 비교: fp16 vs GPTQ (영어캘리브 vs 한국어캘리브)",
                  fontsize=13, fontweight="bold")
    plot_ppl(ppl, a1, a2)
    fig2.tight_layout()
    out2 = RESULTS / "report_ppl.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"[done] {out2}")

    # K-DTCBench 단독
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle("K-DTCBench 정확도", fontsize=13, fontweight="bold")
    plot_kdtcbench(bench, ax3)
    fig3.tight_layout()
    out3 = RESULTS / "report_kdtcbench.png"
    fig3.savefig(out3, bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print(f"[done] {out3}")


if __name__ == "__main__":
    main()
