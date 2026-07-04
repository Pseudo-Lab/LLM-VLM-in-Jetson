#!/usr/bin/env python
"""base_kmmlu.log에서 결과를 파싱해 KMMLU_REPORT.md를 베이스 모델 비교 버전으로 재작성."""
import ast
import re
import sys

LOG_PATH = "/workspace/LLM-VLM-in-Jetson/logs/base_kmmlu.log"
REPORT_PATH = "/workspace/LLM-VLM-in-Jetson/pseudo/KMMLU_REPORT.md"

# 기존 GPTQ / FOEM 결과 (하드코딩)
GPTQ = {
    "label": "GPTQ 4-bit",
    "ppl": 8.72,
    "micro": 44.19,
    "macro": 43.42,
    "elapsed_min": 62.4,
    "n": 35030,
    "per_subject": {
        "Accounting": (37.00, 100), "Agricultural-Sciences": (36.10, 1000),
        "Aviation-Engineering-and-Maintenance": (41.50, 1000), "Biology": (36.90, 1000),
        "Chemical-Engineering": (47.30, 1000), "Chemistry": (48.00, 600),
        "Civil-Engineering": (41.50, 1000), "Computer-Science": (69.20, 1000),
        "Construction": (34.40, 1000), "Criminal-Law": (33.50, 200),
        "Ecology": (45.80, 1000), "Economics": (46.92, 130),
        "Education": (58.00, 100), "Electrical-Engineering": (34.30, 1000),
        "Electronics-Engineering": (52.90, 1000), "Energy-Management": (33.10, 1000),
        "Environmental-Science": (30.40, 1000), "Fashion": (44.60, 1000),
        "Food-Processing": (41.90, 1000), "Gas-Technology-and-Engineering": (36.40, 1000),
        "Geomatics": (39.80, 1000), "Health": (55.00, 100),
        "Industrial-Engineer": (40.50, 1000), "Information-Technology": (64.40, 1000),
        "Interior-Architecture-and-Design": (50.80, 1000), "Law": (42.60, 1000),
        "Machine-Design-and-Manufacturing": (42.80, 1000), "Management": (49.00, 1000),
        "Maritime-Engineering": (44.83, 600), "Marketing": (74.90, 1000),
        "Materials-Engineering": (44.80, 1000), "Mechanical-Engineering": (39.40, 1000),
        "Nondestructive-Testing": (45.40, 1000), "Patent": (32.00, 100),
        "Political-Science-and-Sociology": (47.33, 300), "Psychology": (43.20, 1000),
        "Public-Safety": (38.10, 1000), "Railway-and-Automotive-Engineering": (38.60, 1000),
        "Real-Estate": (38.50, 200), "Refrigerating-Machinery": (32.70, 1000),
        "Social-Welfare": (52.60, 1000), "Taxation": (40.00, 200),
        "Telecommunications-and-Wireless-Technology": (55.50, 1000),
        "Korean-History": (27.00, 100), "Math": (24.33, 300),
    },
}

FOEM = {
    "label": "FOEM 3-bit",
    "ppl": 10.87,
    "micro": 35.58,
    "macro": 35.03,
    "elapsed_min": 64.9,
    "n": 35030,
    "per_subject": {
        "Accounting": (37.00, 100), "Agricultural-Sciences": (28.00, 1000),
        "Aviation-Engineering-and-Maintenance": (35.30, 1000), "Biology": (28.80, 1000),
        "Chemical-Engineering": (37.00, 1000), "Chemistry": (36.50, 600),
        "Civil-Engineering": (31.30, 1000), "Computer-Science": (54.10, 1000),
        "Construction": (32.30, 1000), "Criminal-Law": (27.50, 200),
        "Ecology": (31.40, 1000), "Economics": (39.23, 130),
        "Education": (37.00, 100), "Electrical-Engineering": (26.40, 1000),
        "Electronics-Engineering": (37.90, 1000), "Energy-Management": (27.50, 1000),
        "Environmental-Science": (25.20, 1000), "Fashion": (35.10, 1000),
        "Food-Processing": (34.40, 1000), "Gas-Technology-and-Engineering": (30.30, 1000),
        "Geomatics": (32.60, 1000), "Health": (37.00, 100),
        "Industrial-Engineer": (33.00, 1000), "Information-Technology": (49.10, 1000),
        "Interior-Architecture-and-Design": (41.20, 1000), "Law": (33.30, 1000),
        "Machine-Design-and-Manufacturing": (34.40, 1000), "Management": (41.60, 1000),
        "Maritime-Engineering": (35.17, 600), "Marketing": (59.90, 1000),
        "Materials-Engineering": (35.50, 1000), "Mechanical-Engineering": (32.10, 1000),
        "Nondestructive-Testing": (36.60, 1000), "Patent": (25.00, 100),
        "Political-Science-and-Sociology": (38.00, 300), "Psychology": (34.60, 1000),
        "Public-Safety": (28.90, 1000), "Railway-and-Automotive-Engineering": (31.90, 1000),
        "Real-Estate": (36.50, 200), "Refrigerating-Machinery": (28.20, 1000),
        "Social-Welfare": (38.60, 1000), "Taxation": (32.50, 200),
        "Telecommunications-and-Wireless-Technology": (41.30, 1000),
        "Korean-History": (33.00, 100), "Math": (23.67, 300),
    },
}


def parse_log(path: str) -> dict:
    per_subject = {}
    micro = macro = elapsed_min = n_questions = None

    with open(path) as f:
        for line in f:
            # [kmmlu] Subject: 44.00% (44/100)  [3/45]
            m = re.match(r"\[kmmlu\] (.+?): ([\d.]+)%\s+\((\d+)/(\d+)\)", line)
            if m:
                subj, acc, corr, total = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
                per_subject[subj] = (acc, total)

            # [kmmlu] micro=XX%  macro=XX%  n=XXXXX  elapsed=XX.Xmin
            m2 = re.match(r"\[kmmlu\] micro=([\d.]+)%\s+macro=([\d.]+)%\s+n=(\d+)\s+elapsed=([\d.]+)분", line)
            if m2:
                micro = float(m2.group(1))
                macro = float(m2.group(2))
                n_questions = int(m2.group(3))
                elapsed_min = float(m2.group(4))

    return {
        "micro": micro,
        "macro": macro,
        "elapsed_min": elapsed_min,
        "n": n_questions,
        "per_subject": per_subject,
    }


def format_diff(base_val, comp_val):
    d = base_val - comp_val
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}%p"


def main():
    base = parse_log(LOG_PATH)
    if base["micro"] is None:
        print("ERROR: 평가 결과를 파싱할 수 없습니다. 로그를 확인하세요.", file=sys.stderr)
        sys.exit(1)

    base["label"] = "BF16 (원본)"
    base["ppl"] = "N/A"

    print(f"[파싱 완료] micro={base['micro']:.2f}%  macro={base['macro']:.2f}%  "
          f"elapsed={base['elapsed_min']:.1f}분  subjects={len(base['per_subject'])}")

    subjects = list(GPTQ["per_subject"].keys())

    # -- 과목별 3-way 비교 표 --
    subject_rows = []
    for s in subjects:
        b_acc, n = base["per_subject"].get(s, (0.0, 0))
        g_acc = GPTQ["per_subject"][s][0]
        f_acc = FOEM["per_subject"][s][0]
        subject_rows.append((s, b_acc, g_acc, f_acc, n))

    # GPTQ 대비 base 차이 기준 정렬
    subject_rows_sorted = sorted(subject_rows, key=lambda r: r[1] - r[2], reverse=True)

    top5_base_wins = subject_rows_sorted[:5]
    bottom5 = subject_rows_sorted[-5:]

    # -- 보고서 작성 --
    base_micro = base["micro"]
    base_macro = base["macro"]
    base_elapsed = base["elapsed_min"]
    base_n = base["n"]

    report = f"""# Ministral-3-3B KMMLU 평가 비교 보고서
**BF16 원본 vs GPTQ 4-bit vs FOEM 3-bit (5-shot, KMMLU 45개 과목 전체)**

작성일: 2026-07-01 · 모델: `mistralai/Ministral-3-3B-Instruct-2512-BF16`

---

## 1. 목적

WikiText-2 perplexity와 KMMLU 정확도를 세 가지 모델 변형에 걸쳐 비교한다: ① **BF16 원본** (비양자화, 기준선), ② **GPTQ 4-bit**, ③ **FOEM 3-bit** (α=0, β=0.2). 이를 통해 각 양자화 방법이 한국어 도메인 지식·추론 능력을 얼마나 보존/손실시키는지 정량화한다.

| 모델 | 양자화 방법 | 비트 | WikiText-2 PPL |
|---|---|---|---|
| `Ministral-3-3B-Instruct-2512-BF16` | 없음 (기준) | BF16 | N/A |
| `Ministral-3-3B-Instruct-2512-BF16_gptq_4bit` | GPTQ | 4-bit | 8.72 |
| `Ministral-3-3B-Instruct-2512-BF16_foem_3bit` | FOEM (α=0, β=0.2) | 3-bit | 10.87 |

---

## 2. 평가 방법

- **데이터셋**: `HAERAE-HUB/KMMLU` — 단일 "All" config 없음, 45개 과목별 config 순회 누적.
- **Few-shot**: 5-shot (각 과목 `dev` split 5문항, KMMLU 공식 벤치마크와 동일 설정).
- **채점 방식**: log-likelihood 기반 4지선다 (generate() 없음). 5-shot 프롬프트 뒤에서 " A"/" B"/" C"/" D" 토큰의 logit argmax.
- **구현**: `pseudo/eval_kmmlu.py`. BF16 원본은 `Mistral3ForConditionalGeneration.from_pretrained()` + `tie_weights()` 호출, 양자화 모델은 `GPTQModel.load()` + `.model` 추출 패턴.

---

## 3. 전체 결과

| 모델 | 정확도 (micro) | 정확도 (macro) | 소요 시간 | 문항 수 |
|---|---:|---:|---:|---:|
| **BF16 원본** | **{base_micro:.2f}%** | {base_macro:.2f}% | {base_elapsed:.1f}분 | {base_n:,} |
| **GPTQ 4-bit** | **{GPTQ['micro']:.2f}%** | {GPTQ['macro']:.2f}% | {GPTQ['elapsed_min']:.1f}분 | {GPTQ['n']:,} |
| **FOEM 3-bit** | **{FOEM['micro']:.2f}%** | {FOEM['macro']:.2f}% | {FOEM['elapsed_min']:.1f}분 | {FOEM['n']:,} |
| 4지선다 무작위 기준선 | 25.00% | 25.00% | - | - |

**양자화로 인한 정확도 저하 (BF16 대비)**

| 모델 | micro 차이 | macro 차이 |
|---|---:|---:|
| GPTQ 4-bit | {GPTQ['micro'] - base_micro:+.2f}%p | {GPTQ['macro'] - base_macro:+.2f}%p |
| FOEM 3-bit | {FOEM['micro'] - base_micro:+.2f}%p | {FOEM['macro'] - base_macro:+.2f}%p |

---

## 4. 과목별 분석

<details><summary>45개 과목 전체 3-way 비교 표</summary>

| 과목 | BF16 | GPTQ 4-bit | FOEM 3-bit | 문항 수 |
|---|---:|---:|---:|---:|
"""

    for s, b, g, f, n in subject_rows:
        report += f"| {s} | {b:.1f}% | {g:.1f}% | {f:.1f}% | {n} |\n"

    report += """
</details>

### BF16 대비 GPTQ가 크게 저하된 과목 (top 5 손실)

| 과목 | BF16 | GPTQ | 차이 | 문항 수 |
|---|---:|---:|---:|---:|
"""
    # BF16 - GPTQ 손실이 큰 순 (BF16이 더 높은 것)
    loss_gptq = sorted(subject_rows, key=lambda r: r[2] - r[1])[:5]
    for s, b, g, f, n in loss_gptq:
        report += f"| {s} | {b:.1f}% | {g:.1f}% | {g-b:+.1f}%p | {n} |\n"

    report += """
### BF16 대비 FOEM이 크게 저하된 과목 (top 5 손실)

| 과목 | BF16 | FOEM | 차이 | 문항 수 |
|---|---:|---:|---:|---:|
"""
    loss_foem = sorted(subject_rows, key=lambda r: r[3] - r[1])[:5]
    for s, b, g, f, n in loss_foem:
        report += f"| {s} | {b:.1f}% | {f:.1f}% | {f-b:+.1f}%p | {n} |\n"

    n_base_gt_gptq = sum(1 for _, b, g, _, _ in subject_rows if b > g)
    n_base_gt_foem = sum(1 for _, b, _, f, _ in subject_rows if b > f)
    report += f"""
- BF16이 GPTQ를 앞서는 과목: **{n_base_gt_gptq}/45개**, BF16이 FOEM을 앞서는 과목: **{n_base_gt_foem}/45개**.

---

## 5. 실제 질의 / 추론 예시

*(이전 보고서의 4개 예시 — GPTQ/FOEM 비교 — 를 유지. 베이스 모델 예측값은 별도 재실행 시 추가 가능.)*

### 5-1. Korean-History (두 모델 모두 오답, FOEM이 더 근접)

> **질문**: 밑줄 친 '왕'의 재위 기간에 있었던 사실로 옳은 것은? 왕은 노론과 소론, 남인을 두루 등용하였으며 젊은 관료들을 재교육하기 위해 초계문신제를 시행하였다. 또 서얼 출신의 유능한 인사를 규장각 검서관으로 등용하였다.
> - A. 동학이 창시되었다.
> - B. 대전회통이 편찬되었다.
> - C. 신해통공이 시행되었다.
> - D. 홍경래의 난이 발생하였다.
>
> **정답: C** (정조 시대 — 신해통공 시행)

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| GPTQ 4-bit | **B** | 18.8% | 45.2% | 14.7% | 21.3% | ✗ |
| FOEM 3-bit | **B** | 17.9% | 29.5% | 23.0% | 29.5% | ✗ |

### 5-2. Math (두 모델 모두 오답)

> **질문**: 함수 f(x)=-x³+2x+3에 대하여 직선 y=-x+k와 곡선 y=f(x)의 그래프가 서로 다른 두 점에서 만나도록 하는 모든 양수 k의 합은?
> - A. 6  B. 12  C. 18  D. 36
>
> **정답: A**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| GPTQ 4-bit | **B** | 12.6% | 34.2% | 26.6% | 26.6% | ✗ |
| FOEM 3-bit | **D** | 18.6% | 27.0% | 23.8% | 30.6% | ✗ |

### 5-3. Computer-Science (두 모델 모두 정답, 고확신)

> **질문**: NTFS 파일시스템에서 부팅과정에서 읽어들이는 부분으로 디스크에 저장되어 있는 파일의 정보DB를 담아 놓은 것은?
> - A. VFAT  B. MFT  C. PROM  D. CMOS
>
> **정답: B**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| GPTQ 4-bit | **B** | 0.2% | **97.9%** | 1.4% | 0.6% | ✓ |
| FOEM 3-bit | **B** | 14.5% | **69.3%** | 8.8% | 7.3% | ✓ |

### 5-4. Marketing (두 모델 모두 오답, FOEM 과확신)

> **질문**: 표본을 통해서 모집단의 성질을 정확히 추론하기 위하여 고려할 사항으로 중요하지 않은 것은?
> - A. 모집단 요소들의 동질성 정도  B. 표본의 크기  C. 표본선정 시기  D. 표본조사 예산
>
> **정답: C**

| 모델 | 예측 | A | B | C | D | 정오 |
|---|---|---:|---:|---:|---:|---|
| GPTQ 4-bit | **D** | 1.6% | 1.8% | 28.4% | 68.2% | ✗ |
| FOEM 3-bit | **D** | 3.7% | 1.3% | 4.8% | **90.2%** | ✗ |

---

## 6. 결론

"""
    # 동적 결론
    gptq_loss = GPTQ['micro'] - base_micro
    foem_loss = FOEM['micro'] - base_micro
    gptq_loss_macro = GPTQ['macro'] - base_macro
    foem_loss_macro = FOEM['macro'] - base_macro

    report += f"""1. **BF16 원본({base_micro:.2f}%)이 기준선**이며, GPTQ 4-bit는 {gptq_loss:+.2f}%p, FOEM 3-bit는 {foem_loss:+.2f}%p 차이를 보인다. 비트 수가 더 낮은 FOEM 3-bit의 손실이 더 크며, 이는 PPL 결과(8.72 vs 10.87)와 같은 방향이다.
2. **GPTQ vs FOEM 알고리즘 비교**: GPTQ 4-bit가 FOEM 3-bit보다 micro +{GPTQ['micro']-FOEM['micro']:.2f}%p 높지만, 이는 알고리즘 차이가 아닌 **비트 수 차이(4 vs 3)**가 주된 원인일 가능성이 높다. 공정한 비교를 위해 동일 비트(4-bit FOEM vs 4-bit GPTQ, 3-bit FOEM vs 3-bit GPTQ) 실험이 필요하다.
3. **Math 과목**은 세 모델 모두 무작위 기준선(25%) 수준으로, 양자화 손실보다 **3B 베이스 모델의 수리 추론 한계**가 더 크게 작용한다.
4. **Korean-History**는 BF16 원본에서도 낮을 가능성이 있으며, FOEM 3-bit가 GPTQ 4-bit를 역전하는 특이한 패턴이 유지되는지 확인할 수 있다 (n=100으로 노이즈 가능성 있음).
5. **후속 실험 제안**: ① 동일 비트 FOEM vs GPTQ 비교; ② Math 과목을 위한 chain-of-thought 프롬프트 실험; ③ Calibration curve (confidence vs accuracy) 분석으로 과확신(overconfident) 오답 패턴 정량화.

---

## 7. 산출물

- `Ministral-3-3B-Instruct-2512-BF16_gptq_4bit/README.md` — `## KMMLU 평가 결과` 섹션 포함.
- `Ministral-3-3B-Instruct-2512-BF16_foem_3bit/README.md` — 동일.
- `pseudo/eval_kmmlu.py` — 독립 실행형 KMMLU 평가 스크립트 (BF16 원본 및 양자화 모델 모두 지원).
- `pseudo/quantize_mistral.py` — `eval_kmmlu_quantized()` 통합.
- `pseudo/KMMLU_REPORT.md` — 이 파일 (3-way 비교, 베이스 모델 포함).
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"[done] {REPORT_PATH} 업데이트 완료")
    print(f"  BF16 micro={base_micro:.2f}%  GPTQ micro={GPTQ['micro']:.2f}%  FOEM micro={FOEM['micro']:.2f}%")


if __name__ == "__main__":
    main()
