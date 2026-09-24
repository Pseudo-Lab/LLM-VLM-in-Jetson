"""압축 시도 결과 시각화 — '8GB Jetson 적재' 목표 대비 한계 정리.

밤사이 실험(3-bit 재양자화 + depth pruning)의 핵심 메시지를 한 장으로:
  (A) 모델 크기: fp16 → 4bit → 3bit. 3-bit도 ~10GB라 8GB 목표 미달.
      이유 = 덩치의 ~4.5GB가 양자화 안 된 fp16 비전/cross-attn/임베딩.
  (B) Depth pruning: self-attn 층 드롭 시 KO PPL — 힐링 없으면 2층 넘어가며 붕괴.

결론: one-shot 3-bit·프루닝으론 8GB 불가. 비전경로 양자화(멀티모달 캘리브)+
프루닝 힐링(LoRA)+sub-4bit(AQLM 등) 조합이라야 → Phase-2.

실행: python src/make_compression_chart.py  → results/report_compression.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

C_FP16, C_4BIT, C_3BIT, C_RED, C_TGT = "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#888888"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})


def main():
    sweep = json.loads((RESULTS / "pruning_sweep.json").read_text(encoding="utf-8"))["sweep"]
    ko3 = json.loads((RESULTS / "ppl_gptq_3bit_kocalib.json").read_text())["ko"]["ppl"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("압축 시도 결과 — 8GB Jetson 적재 목표 대비 한계",
                 fontsize=14, fontweight="bold")

    # ── (A) 모델 크기 ──
    names = ["fp16 원본", "GPTQ 4-bit\n(한국어캘리브)", "GPTQ 3-bit\n(한국어캘리브)"]
    sizes = [21.34, 11.01, 10.13]
    colors = [C_FP16, C_4BIT, C_3BIT]
    bars = axA.bar(names, sizes, color=colors, width=0.55, zorder=3)
    for b, s in zip(bars, sizes):
        axA.text(b.get_x() + b.get_width() / 2, s + 0.3, f"{s:.1f}GB",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    axA.axhline(8, color=C_RED, linestyle="--", linewidth=1.6, zorder=2)
    axA.text(2.45, 8.2, "Jetson 8GB 한계", color=C_RED, ha="right",
             va="bottom", fontsize=9, fontweight="bold")
    axA.set_ylabel("모델 크기 (GB, 낮을수록 좋음)", fontsize=10)
    axA.set_title("① 크기 — 3-bit도 ~10GB (목표 미달)", fontsize=11.5, fontweight="bold", pad=8)
    axA.set_ylim(0, 23)
    axA.text(0.5, -0.30,
             "3-bit는 4-bit 대비 0.9GB만 절감 — 덩치의 ~4.5GB가 양자화 안 된 fp16 비전/cross-attn/임베딩이라.",
             transform=axA.transAxes, ha="center", va="top", fontsize=8.5,
             color=C_RED, style="italic")

    # ── (B) Depth pruning PPL 절벽 ──
    ks = [r["k"] for r in sweep]
    kos = [r["ko"] for r in sweep]
    axB.plot(ks, kos, "o-", color=C_4BIT, linewidth=2, markersize=7, zorder=3)
    for k, v in zip(ks, kos):
        axB.annotate(f"{v:.0f}" if v >= 100 else f"{v:.1f}",
                     (k, v), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, fontweight="bold")
    axB.axhline(kos[0], color=C_TGT, linestyle=":", linewidth=1.3, zorder=2)
    axB.text(8, kos[0] * 1.15, "프루닝 0층(기준)", color=C_TGT, ha="right", fontsize=8.5)
    axB.set_yscale("log")
    axB.set_xlabel("드롭한 self-attn 층 수 (×0.11GB/층)", fontsize=10)
    axB.set_ylabel("KO Perplexity (log, 낮을수록 좋음)", fontsize=10)
    axB.set_title("② Depth pruning — 2층 넘으면 붕괴(힐링 없음)", fontsize=11.5, fontweight="bold", pad=8)
    axB.set_xticks(ks)
    axB.text(0.5, -0.30,
             "fine-tune 힐링 없이 one-shot 드롭은 ~2층이 한계(+28%). 4층부터 사용불가. "
             "절감도 층당 0.11GB뿐.",
             transform=axB.transAxes, ha="center", va="top", fontsize=8.5,
             color=C_RED, style="italic")

    fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.28)
    out = RESULTS / "report_compression.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
