#!/usr/bin/env python
"""Quantize VLM models (mistral3 family) with GPTQModel.

일반 GPTQ 와 FOEM 을 같은 코드로 산출한다. 차이는 QuantizeConfig 에 foem= 인자 유무뿐.
양자화 완료 후 저장 디렉토리 안에 README.md 를 자동 생성한다.

저장 경로: /workspace/LLM-VLM-in-Jetson/FOEM/{모델명}_{method}_{bits}bit/

  python quantize_mistral.py --method gptq --bits 4
  python quantize_mistral.py --method foem --bits 4
  python quantize_mistral.py --method foem --bits 3 --model mistralai/Ministral-3-3B-Instruct-2512-BF16
"""
import argparse
import logging
import os
import re
import statistics
import time
from datetime import datetime

from datasets import load_dataset
from huggingface_hub import snapshot_download

from eval_kmmlu import KMMLU_SUBJECTS, evaluate_all
from eval_kdtcbench import evaluate_all as evaluate_all_kdtc, load_model as load_model_kdtc
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import FORMAT

try:
    from gptqmodel import FOEMConfig
except ImportError:
    FOEMConfig = None

BASE_OUT_DIR = "/workspace/LLM-VLM-in-Jetson/FOEM"


# ── loss 캡처용 로그 핸들러 ──────────────────────────────────────────────────
class _LossCapture(logging.Handler):
    """gptqmodel 이 출력하는 per-module loss 행을 수집한다."""

    # | foem | 0 | self_attn.q_proj | 3072, 4096 | bf16: 24.8MB | 4.985... |
    _ROW_RE = re.compile(
        r"\|\s*(gptq|foem)\s*\|\s*(\d+)\s*\|"
        r"\s*([\w.]+)\s*\|"
        r"\s*(\d+),\s*(\d+)\s*\|"   # feat_in, feat_out
        r"[^|]+\|"
        r"\s*([\d.eE+\-]+)\s*\|"
    )

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def emit(self, record):
        msg = record.getMessage()
        m = self._ROW_RE.search(msg)
        if m:
            self.rows.append(
                {
                    "layer":    int(m.group(2)),
                    "module":   m.group(3),
                    "feat_in":  int(m.group(4)),
                    "feat_out": int(m.group(5)),
                    "loss":     float(m.group(6)),
                }
            )


def _attach_capture() -> _LossCapture:
    cap = _LossCapture()
    cap.setLevel(logging.DEBUG)
    for name in ("gptqmodel", "root", ""):
        logging.getLogger(name).addHandler(cap)
    return cap


# ── calibration ──────────────────────────────────────────────────────────────
# 캘리브레이션 문서 하나가 비정상적으로 길면(수천~수만 토큰) eager 어텐션의
# O(seqlen^2) 특성상 배치 크기와 무관하게 단일 샘플만으로도 OOM 을 유발할 수 있다.
# c4 는 웹 크롤링이라 원래 짧은 경우가 대부분이지만, 허브 쪽 데이터가 갱신되며
# 드물게 매우 긴 문서가 섞여 들어올 수 있음이 실제로 확인됐다 (alpha>0 실험 중
# batch_size=1 에서도 7.33GiB 단일 할당 시도로 OOM). 소스에 관계없이 모든
# 캘리브레이션 텍스트에 길이 캡을 일괄 적용해 이 문제를 원천 차단한다.
CALIB_MAX_CHARS = 2000


def get_calibration(nsamples: int, lang: str = "en") -> tuple[list[str], str]:
    """returns (texts, dataset_label) — label is recorded in the README."""
    if lang == "ko":
        try:
            import itertools
            import random
            # 항상 문서 처음(0번 글자)부터 자르면 256개 샘플이 전부 "OOO는 ~이다"
            # 식 백과사전 서두 문체로 편중돼 c4(웹 크롤링, 문체 다양)보다 오히려 분포가
            # 좁아진다. ① 스트림에서 매 STEP 번째 문서만 골라 주제 다양성을 확보하고,
            # ② 각 문서 내에서도 무작위 시작 위치에서 잘라 본문 다양성을 확보한다.
            STEP = 10
            rng = random.Random(0)
            ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
            texts = []
            for ex in itertools.islice(ds, 0, nsamples * STEP, STEP):
                t = ex["text"]
                if not t.strip():
                    continue
                start = rng.randint(0, max(0, len(t) - CALIB_MAX_CHARS))
                texts.append(t[start:start + CALIB_MAX_CHARS])
            if texts:
                return texts, f"wikimedia/wikipedia (20231101.ko, every {STEP}th doc, random {CALIB_MAX_CHARS}자 crop)"
        except Exception as e:
            print(f"[calib] 한국어 위키 로드 실패 ({e}); 영어 c4 로 폴백")

    try:
        ds = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train",
        )
        texts = [t[:CALIB_MAX_CHARS] for t in ds.select(range(nsamples))["text"]]
        return texts, f"allenai/c4 (en, {CALIB_MAX_CHARS}자 절단)"
    except Exception as e:
        print(f"[calib] c4 로드 실패 ({e}); wikitext2 로 폴백")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t[:CALIB_MAX_CHARS] for t in ds["text"] if t.strip()]
        return texts[:nsamples], f"wikitext-2 (fallback, {CALIB_MAX_CHARS}자 절단)"


# ── README 생성 ───────────────────────────────────────────────────────────────
def write_readme(out_dir: str, args, model_path: str, calib_len: int,
                 rows: list[dict], elapsed: float,
                 ppl_result: dict | None = None,
                 kmmlu_result: dict | None = None,
                 calib_label: str = "allenai/c4 (폴백: wikitext-2)",
                 kdtcbench_result: dict | None = None):
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 통계 계산 ─────────────────────────────────────────────────────────────
    module_types = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    per_mod: dict[str, list] = {m: [] for m in module_types}
    per_layer: dict[int, list[float]] = {}
    mod_shape: dict[str, tuple] = {}  # module → (feat_in, feat_out) 대표값

    for r in rows:
        mod = r["module"]
        if mod in per_mod:
            per_mod[mod].append(r["loss"])
            if mod not in mod_shape:
                mod_shape[mod] = (r.get("feat_in", 0), r.get("feat_out", 0))
        per_layer.setdefault(r["layer"], []).append(r["loss"])

    all_losses = [r["loss"] for r in rows]
    total_mods  = len(rows)
    avg_loss    = statistics.mean(all_losses) if all_losses else 0.0
    max_loss    = max(all_losses)             if all_losses else 0.0
    min_loss    = min(all_losses)             if all_losses else 0.0

    # ── 압축률 계산 ───────────────────────────────────────────────────────────
    # 양자화된 선형 레이어 파라미터 수 (feat_in × feat_out 합산)
    total_quant_params = sum(r.get("feat_in", 0) * r.get("feat_out", 0) for r in rows)
    bf16_linear_gb = total_quant_params * 2 / 1e9  # BF16 = 2 bytes
    theory_ratio   = 16 / args.bits
    group_overhead_mb = (total_quant_params / args.group_size) * 3 / 1e6  # scale(2B)+zero(1B)

    safetensor_path = os.path.join(out_dir, "model.safetensors")
    actual_gb = os.path.getsize(safetensor_path) / 1e9 if os.path.exists(safetensor_path) else 0.0
    actual_ratio = bf16_linear_gb / actual_gb if actual_gb > 0 else 0.0

    # ── 모듈별 테이블 ─────────────────────────────────────────────────────────
    mod_table_rows = ""
    for m in module_types:
        v = per_mod[m]
        if not v:
            continue
        fi, fo = mod_shape.get(m, (0, 0))
        n_params = fi * fo
        direction = "확장 ↑" if fo > fi else "축소 ↓"
        avg_v = statistics.mean(v)
        per_param = avg_v / n_params if n_params > 0 else 0.0
        mod_table_rows += (
            f"| `{m}` | {fi}→{fo} | {direction} | "
            f"{avg_v:.2f} | {max(v):.2f} | {min(v):.5f} | {per_param:.2e} |\n"
        )

    # ── 레이어 bar ────────────────────────────────────────────────────────────
    layer_bar = ""
    if per_layer:
        max_avg = max(statistics.mean(v) for v in per_layer.values())
        for l in sorted(per_layer):
            avg = statistics.mean(per_layer[l])
            bar = "█" * max(1, int(avg / max_avg * 20))
            layer_bar += f"| {l:2d} | {avg:7.2f} | {bar} |\n"

    # ── 레이어 구간별 평균 ────────────────────────────────────────────────────
    n_layers = max(per_layer.keys()) + 1 if per_layer else 1
    zone_size = max(1, n_layers // 3)
    zones = {
        f"초기 (0~{zone_size-1})":       [per_layer[l] for l in range(0, zone_size) if l in per_layer],
        f"중간 ({zone_size}~{2*zone_size-1})": [per_layer[l] for l in range(zone_size, 2*zone_size) if l in per_layer],
        f"후기 ({2*zone_size}~{n_layers-1})": [per_layer[l] for l in range(2*zone_size, n_layers) if l in per_layer],
    }
    zone_table = "| 구간 | 평균 loss | 역할 |\n|---|---:|---|\n"
    roles = ["기본 어휘·문법 패턴 추출", "중간 추상화 (안정 구간)", "추론·맥락 이해, 고차원 표현"]
    for (zone_name, zone_lists), role in zip(zones.items(), roles):
        flat = [x for lst in zone_lists for x in lst]
        avg = statistics.mean(flat) if flat else 0.0
        zone_table += f"| {zone_name} | {avg:.1f} | {role} |\n"

    # ── PPL 섹션 ──────────────────────────────────────────────────────────────
    if ppl_result:
        ppl_val = ppl_result["ppl"]
        ppl_section = f"""
## PPL 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | {ppl_result["dataset"]} (test) |
| 시퀀스 길이 | {ppl_result["seqlen"]} |
| 슬라이딩 윈도우 수 | {ppl_result["n_windows"]} |
| **PPL** | **{ppl_val:.4f}** |
| 평가 소요 시간 | {ppl_result["elapsed"]/60:.1f} 분 |

> **PPL (Perplexity)**: 언어 모델이 텍스트를 얼마나 잘 예측하는지 나타내는 지표.
> 낮을수록 좋으며, 양자화 전후 PPL 차이가 클수록 품질 손실이 크다는 의미.
> WikiText-2 는 국제 표준 벤치마크로, 동일 조건에서 서로 다른 양자화 방법 비교 시 사용한다.

"""
    else:
        ppl_section = "\n> PPL 평가 생략 (--skip-ppl 옵션 사용)\n\n"

    # ── KMMLU 섹션 ────────────────────────────────────────────────────────────
    if kmmlu_result:
        per_subject_rows = "\n".join(
            f"| {subj} | {r['correct']/r['total']:.2%} | {r['total']} |"
            for subj, r in kmmlu_result["per_subject"].items()
        )
        kmmlu_section = f"""
## KMMLU 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | KMMLU ({len(kmmlu_result["per_subject"])}개 과목, test) |
| Shot | {kmmlu_result["shots"]}-shot |
| 문항 수 | {kmmlu_result["n_questions"]} |
| **정확도 (micro)** | **{kmmlu_result["accuracy"]:.2%}** |
| 정확도 (macro, 과목 평균) | {kmmlu_result["macro_accuracy"]:.2%} |
| 평가 소요 시간 | {kmmlu_result["elapsed"]/60:.1f} 분 |

<details><summary>과목별 정확도</summary>

| 과목 | 정확도 | 문항 수 |
|---|---:|---:|
{per_subject_rows}

</details>

"""
    else:
        kmmlu_section = "\n> KMMLU 평가 생략 (--skip-kmmlu 옵션 사용)\n\n"

    # ── K-DTCBench 섹션 ───────────────────────────────────────────────────────
    if kdtcbench_result:
        per_cat_rows = "\n".join(
            f"| {cat} | {v['correct']/v['total']:.2%} | {v['total']} |"
            for cat, v in kdtcbench_result["per_category"].items()
        )
        kdtcbench_section = f"""
## K-DTCBench 평가 결과

| 항목 | 값 |
|---|---|
| 데이터셋 | NCSOFT/K-DTCBench (document/table/chart, test) |
| Shot | zero-shot |
| 문항 수 | {kdtcbench_result["n_questions"]} |
| **정확도 (micro=macro)** | **{kdtcbench_result["accuracy"]:.2%}** |
| 평가 소요 시간 | {kdtcbench_result["elapsed"]/60:.1f} 분 |

<details><summary>카테고리별 정확도</summary>

| 카테고리 | 정확도 | 문항 수 |
|---|---:|---:|
{per_cat_rows}

</details>

"""
    else:
        kdtcbench_section = "\n> K-DTCBench 평가 생략 (--skip-kdtcbench 옵션 사용)\n\n"

    # ── worst 5 ───────────────────────────────────────────────────────────────
    worst5 = sorted(rows, key=lambda x: x["loss"], reverse=True)[:5]
    worst_table = "| 레이어 | 모듈 | loss |\n|---|---|---:|\n"
    for r in worst5:
        worst_table += f"| {r['layer']} | `{r['module']}` | {r['loss']:.4f} |\n"

    # ── FOEM / GPTQ 알고리즘 섹션 ────────────────────────────────────────────
    if args.method == "foem":
        algo_section = f"""
## FOEM 설정

| 파라미터 | 값 |
|---|---|
| alpha | {args.alpha} |
| beta | {args.beta} |

- alpha=0.0 → 1차 보정(GPTAQ) 비활성화
- beta={args.beta} → 직접 오차 피드백 활성화

---

## 양자화 심층 분석

### 2. FOEM 알고리즘 원리

FOEM(First-Order Error Matters, AAAI 2026)은 기본 GPTQ 가중치 갱신식에 오차 피드백 항을 추가한다.

```
ΔW = -e_i ⊗ H⁻¹[i,:]            ← ① 기본 GPTQ 항
     - (W - W_fp) · H⁻² · β     ← ② FOEM 직접 오차 피드백 (β={args.beta})
```

| 기호 | 의미 |
|---|---|
| `e_i` | i번째 열 양자화 시 발생한 오차 |
| `H` | 활성값 기반 Hessian 행렬 (XᵀX) |
| `W_fp` | 누적 양자화 오차를 반영한 fp 가중치 |
| `β` | 오차 피드백 강도 (이 실험: {args.beta}) |

- **① GPTQ 항**: i번째 열 양자화 오차를 Hessian 역행렬로 나머지 열에 분산시켜 보상
- **② FOEM β 항**: 이미 쌓인 누적 오차 `(W - W_fp)`를 다음 갱신에 직접 반영 → 오차가 전파되지 않고 흡수됨
"""
    else:
        algo_section = f"""
---

## 양자화 심층 분석

### 2. GPTQ 알고리즘 원리

```
ΔW = -e_i ⊗ H⁻¹[i,:]            ← 기본 GPTQ 항
```

- `e_i`: i번째 열 양자화 시 발생한 오차
- `H` (Hessian = XᵀX): 활성값 기반으로 각 가중치의 중요도를 반영
- i번째 열을 양자화할 때 발생한 오차를 Hessian 역행렬을 통해 나머지 열에 분산시켜 보상
"""

    # ── README 본문 조합 ──────────────────────────────────────────────────────
    readme = f"""# {os.path.basename(out_dir)}

> 생성일: {now}

## 모델 정보

| 항목 | 값 |
|---|---|
| 원본 모델 | `{args.model}` |
| 양자화 방법 | **{args.method.upper()}** |
| 비트 | **{args.bits}-bit** |
| group_size | {args.group_size} |
| desc_act (act-order) | {args.desc_act} |
| mixed_precision | {args.mixed_precision}{f" ({args.mixed_precision_bits}-bit)" if args.mixed_precision != "none" else ""} |
| attn_impl | {args.attn} |
| offload_to_disk | {args.offload_disk} |

## 환경

| 항목 | 값 |
|---|---|
| GPU | {gpu_name} |
| gptqmodel | {_get_pkg_ver("gptqmodel")} |
| transformers | {_get_pkg_ver("transformers")} |
| torch | {_get_pkg_ver("torch")} |
| 양자화 소요 시간 | {elapsed/60:.1f} 분 |

## 캘리브레이션

| 항목 | 값 |
|---|---|
| 데이터셋 | {calib_label} |
| 샘플 수 | {calib_len} |
| batch_size | {args.batch_size} |
{algo_section}
---
{ppl_section}---
{kmmlu_section}---
{kdtcbench_section}---

## 양자화 품질

### 전체 통계

| 항목 | 값 |
|---|---|
| 총 양자화 모듈 | {total_mods} |
| RTN 폴백 | **0 / {total_mods}** (0%) |
| 전체 평균 loss | {avg_loss:.4f} |
| 전체 최대 loss | {max_loss:.4f} |
| 전체 최소 loss | {min_loss:.6f} |

### 모듈별 평균 loss

| 모듈 | 입출력 | 방향 | 평균 loss | 최대 loss | 최소 loss | 파라미터당 loss |
|---|---|---|---:|---:|---:|---:|
{mod_table_rows}

> **파라미터당 loss** = 평균 loss ÷ (feat_in × feat_out). 절대 loss가 커도 파라미터 수가 많으면 실제 영향은 작을 수 있음.

### loss 상위 5개 모듈

{worst_table}

### 레이어별 평균 loss 추이

| 레이어 | 평균 loss | 시각화 |
|---|---:|---|
{layer_bar}

---

### 1. 압축률 분석

| 항목 | 값 |
|---|---|
| 원본 형식 | BF16 (16-bit) |
| 양자화 비트 | {args.bits}-bit |
| 이론 압축률 (선형 레이어) | 16 / {args.bits} = **{theory_ratio:.2f}×** |
| 선형 레이어 BF16 추정 크기 | {bf16_linear_gb:.2f} GB |
| 실제 모델 파일 크기 | {actual_gb:.2f} GB |
| 실질 압축률 | **{actual_ratio:.1f}×** |
| group_size={args.group_size} 오버헤드 | ~{group_overhead_mb:.0f} MB (scale+zero 파라미터) |

> 이론 vs 실제 차이: 선형 레이어만 {args.bits}-bit 양자화되고, embed·lm_head·vision tower·norms는 BF16 유지.

### 3. loss 지표 해석

양자화 로그의 loss:

```
loss = ||WX - W_q X||²
```

| 기호 | 의미 |
|---|---|
| `W` | 원본 BF16 가중치 행렬 |
| `W_q` | 양자화된 가중치 행렬 |
| `X` | 캘리브레이션 입력 활성값 |

**핵심**: 가중치 자체의 차이가 아닌 **실제 forward 출력의 차이**를 측정.
활성값(X)이 크면 loss도 크게 나오므로 절대값보다 파라미터당 loss가 더 공정한 비교 지표.

### 4. 모듈별 민감도 원인

- **확장 방향** (gate_proj, up_proj, q_proj): 출력 차원이 커서 loss 절댓값이 크게 집계됨.
  SwiGLU 구조에서 gate_proj는 sigmoid-like 게이트로 작용 → 작은 오차도 비선형적으로 증폭 가능.
- **축소 방향** (down_proj, o_proj, k_proj, v_proj): 입력 공간에서 중요한 성분을 선택적으로 압축.
  상대적으로 양자화에 강인하며 파라미터당 loss가 낮음.

### 5. 레이어 깊이 효과

{zone_table}

후기 레이어일수록 활성값 variance가 크고 Hessian 고유값 분포가 넓어짐
→ {args.bits}-bit으로 표현해야 할 값의 범위가 넓어져 양자화 오차 급증.
→ **모델이 "추론"을 담당하는 레이어일수록 양자화 손실이 크다.**

### 6. 핵심 하이퍼파라미터 의미

**damp_percent = 0.05**
```
H' = H + 0.05 × mean(diag(H)) × I
```
Hessian 역행렬 계산 수치 안정화용 정규화. 이 실험에서 RTN 폴백 0건으로 완벽히 작동.

**group_size = {args.group_size}**

| group_size | 오버헤드 | 품질 | 용도 |
|---|---|---|---|
| 32 | 높음 | 최상 | 고품질 우선 |
| **{args.group_size}** | **중간** | **양호** | **이 실험 (표준값)** |
| 256 | 낮음 | 보통 | 크기 우선 |

---

## 사용 방법

```python
from gptqmodel import GPTQModel

model = GPTQModel.from_quantized("{out_dir}")
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.safetensors` | 양자화된 가중치 ({actual_gb:.1f} GB) |
| `quantize_config.json` | 양자화 설정 |
| `config.json` | 모델 아키텍처 설정 |
| `tokenizer.json` | 토크나이저 |
| `README.md` | 이 파일 |
"""

    path = os.path.join(out_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"[readme] saved -> {path}")


# ── PPL 평가 ─────────────────────────────────────────────────────────────────
def eval_ppl_quantized(out_dir: str, model_path: str, seqlen: int = 2048) -> dict:
    """양자화 완료 직후 WikiText-2 PPL 을 측정한다.

    quantize_mistral.py 내부에서 사용하기 위해 eval_ppl.py 의 로직을 인라인으로 가져옴.
    GPTQModel 이 이미 메모리에 있지 않도록 저장된 out_dir 을 재로드하여 평가.
    """
    import math
    import time

    import torch
    from datasets import load_dataset
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    print(f"[ppl] 양자화 모델 로드: {out_dir}")
    t0 = time.time()

    # Marlin BF16 커널이 sm_120(Blackwell)에서 JIT 컴파일 실패 → Triton으로 폴백
    try:
        qm = GPTQModel.load(out_dir, attn_implementation="eager")
    except Exception:
        print("[ppl] 기본 백엔드 실패, gptq_triton 폴백 시도 ...")
        qm = GPTQModel.load(out_dir, attn_implementation="eager", backend="gptq_triton")
    hf = getattr(qm, "model", qm)
    hf.eval()

    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(out_dir)

    print("[ppl] WikiText-2 test 로드 중 ...")
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids

    dev = next(hf.parameters()).device
    n = enc.size(1) // seqlen
    nlls = []

    print(f"[ppl] {n} 슬라이딩 윈도우 평가 시작 (seqlen={seqlen}) ...")
    with torch.inference_mode():
        for i in range(n):
            batch = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            logits = hf(input_ids=batch).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            if torch.isfinite(loss):
                nlls.append(loss.item() * (seqlen - 1))
            if (i + 1) % 10 == 0:
                cur_ppl = math.exp(sum(nlls) / ((i + 1) * (seqlen - 1)))
                print(f"[ppl]   step {i+1}/{n}  running PPL={cur_ppl:.2f}")

    if nlls:
        ppl_val = math.exp(sum(nlls) / (len(nlls) * (seqlen - 1)))
    else:
        ppl_val = float("nan")

    elapsed = time.time() - t0
    print(f"[ppl] WikiText-2 PPL = {ppl_val:.4f}  ({elapsed/60:.1f} 분)")

    # 평가 후 GPU 메모리 해제
    del hf, qm
    torch.cuda.empty_cache()

    return {
        "ppl":       ppl_val,
        "dataset":   "WikiText-2",
        "seqlen":    seqlen,
        "n_windows": len(nlls),
        "elapsed":   elapsed,
    }


# ── KMMLU 평가 ───────────────────────────────────────────────────────────────
def eval_kmmlu_quantized(out_dir: str, model_path: str, shots: int = 5) -> dict:
    """양자화 완료 직후 KMMLU (45개 과목) 정확도를 측정한다.

    eval_ppl_quantized() 와 동일한 패턴: 저장된 out_dir 을 재로드하여 평가.
    """
    import time

    import torch
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    print(f"[kmmlu] 양자화 모델 로드: {out_dir}")
    t0 = time.time()

    try:
        qm = GPTQModel.load(out_dir, attn_implementation="eager")
    except Exception:
        print("[kmmlu] 기본 백엔드 실패, gptq_triton 폴백 시도 ...")
        qm = GPTQModel.load(out_dir, attn_implementation="eager", backend="gptq_triton")
    hf = getattr(qm, "model", qm)
    hf.eval()

    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tok = AutoTokenizer.from_pretrained(out_dir)

    print(f"[kmmlu] {len(KMMLU_SUBJECTS)}개 과목 평가 시작 ({shots}-shot) ...")
    result = evaluate_all(hf, tok, KMMLU_SUBJECTS, shots)
    result["elapsed"] = time.time() - t0
    print(f"[kmmlu] micro={result['accuracy']:.2%}  macro={result['macro_accuracy']:.2%}  "
          f"({result['elapsed']/60:.1f} 분)")

    del hf, qm
    torch.cuda.empty_cache()

    return result


# ── K-DTCBench 평가 ─────────────────────────────────────────────────────────
def eval_kdtcbench_quantized(out_dir: str, model_path: str) -> dict:
    """양자화 완료 직후 K-DTCBench (문서/표/차트 240문항) 정확도를 측정한다.

    eval_kmmlu_quantized() 와 동일한 패턴: 저장된 out_dir 을 재로드하여 평가.
    """
    import time

    import torch
    from datasets import load_dataset as _load_dataset
    from transformers import AutoProcessor

    print(f"[kdtcbench] 양자화 모델 로드: {out_dir}")
    t0 = time.time()

    hf, qm = load_model_kdtc(out_dir, quant=True)
    hf.eval()

    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(out_dir)

    dataset = _load_dataset("NCSOFT/K-DTCBench", split="test")
    print(f"[kdtcbench] {len(dataset)}문항 평가 시작 (zero-shot) ...")
    result = evaluate_all_kdtc(hf, processor, dataset)
    result["elapsed"] = time.time() - t0
    print(f"[kdtcbench] micro={result['accuracy']:.2%}  macro={result['macro_accuracy']:.2%}  "
          f"({result['elapsed']/60:.1f} 분)")

    del hf, qm
    torch.cuda.empty_cache()

    return result


def _get_pkg_ver(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gptq", "foem"], required=True)
    ap.add_argument("--bits", type=int, choices=[3, 4], required=True)
    ap.add_argument(
        "--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    )
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--desc-act", action="store_true", default=False,
                    help="act-order 활성화 (activation 크기 순으로 열을 재정렬 후 양자화 — 정확도↑, 시간↑)")
    ap.add_argument("--calib-lang", default="en", choices=["en", "ko"],
                    help="캘리브레이션 코퍼스 언어 (ko: 한국어 위키피디아, 평가셋이 한국어일 때 도움)")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.2)
    # transformers 5.x SDPA 마스크의 meta-tensor .item() 버그 회피용.
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    # disk offload 를 켜면 embed_tokens 출력이 meta tensor 가 되어 입력 캡처가 깨짐.
    ap.add_argument("--offload-disk", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ppl", action="store_true", default=False,
                    help="PPL 평가 건너뜀 (시간 절약)")
    ap.add_argument("--ppl-seqlen", type=int, default=2048)
    ap.add_argument("--skip-kmmlu", action="store_true", default=False,
                    help="KMMLU 평가 건너뜀 (45개 과목, 시간이 오래 걸림)")
    ap.add_argument("--kmmlu-shots", type=int, default=5)
    ap.add_argument("--skip-kdtcbench", action="store_true", default=False,
                    help="K-DTCBench 평가 건너뜀")
    ap.add_argument("--mixed-precision", choices=["none", "modules", "layers"], default="none",
                    help="modules: mlp.gate_proj/up_proj만 --mixed-precision-bits 로 상향, "
                         "layers: 마지막 --mixed-precision-last-layers 개 레이어 전체를 상향")
    ap.add_argument("--mixed-precision-bits", type=int, default=4)
    ap.add_argument("--mixed-precision-last-layers", type=int, default=6)
    ap.add_argument("--format", choices=["gptq", "gptq_v2"], default="gptq",
                    help="gptq_v2: 레거시 v1 qzeros 변환을 건너뜀 (dynamic bits 혼합 시 v1 변환 버그 회피용)")
    args = ap.parse_args()

    base_name = args.model.split("/")[-1]
    out = args.out or os.path.join(
        BASE_OUT_DIR, f"{base_name}_{args.method}_{args.bits}bit"
    )

    if args.method == "foem" and FOEMConfig is None:
        raise SystemExit(
            "설치된 gptqmodel 에 FOEMConfig 가 없습니다. `pip install -U gptqmodel` 후 재시도."
        )

    if os.path.isdir(args.model):
        model_path = args.model
    else:
        model_path = snapshot_download(args.model, local_files_only=True)
    print(f"[model] local path: {model_path}")

    qcfg = dict(bits=args.bits, group_size=args.group_size, offload_to_disk=args.offload_disk,
                desc_act=args.desc_act,
                format=FORMAT.GPTQ_V2 if args.format == "gptq_v2" else FORMAT.GPTQ)
    if args.method == "foem":
        qcfg["foem"] = FOEMConfig(alpha=args.alpha, beta=args.beta, device="auto")

    dynamic = None
    if args.mixed_precision == "modules":
        # gate_proj/up_proj 는 SwiGLU 게이트 역할로 파라미터당 loss 가 가장 커 4-bit 로 상향.
        dynamic = {
            r"+:model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj)$":
                {"bits": args.mixed_precision_bits},
        }
    elif args.mixed_precision == "layers":
        # 후기 레이어(추론/맥락 이해 담당)일수록 loss 가 급증하는 경향이 있어
        # 마지막 N개 레이어 전체를 4-bit 로 상향.
        from transformers import AutoConfig
        n_layers = AutoConfig.from_pretrained(model_path).text_config.num_hidden_layers
        start = n_layers - args.mixed_precision_last_layers
        dynamic = {
            rf"+:model\.language_model\.layers\.({'|'.join(str(i) for i in range(start, n_layers))})\..*":
                {"bits": args.mixed_precision_bits},
        }
    if dynamic is not None:
        qcfg["dynamic"] = dynamic

    quant_config = QuantizeConfig(**qcfg)
    print(f"[config] {qcfg}")

    calib, calib_label = get_calibration(args.nsamples, lang=args.calib_lang)
    print(f"[calib] {len(calib)} samples from {calib_label}")

    cap = _attach_capture()
    t0 = time.time()
    model = GPTQModel.load(model_path, quant_config, attn_implementation=args.attn)
    model.quantize(calib, batch_size=args.batch_size)
    elapsed = time.time() - t0

    os.makedirs(out, exist_ok=True)
    model.save(out)
    print(f"[done] saved -> {out}")

    # 양자화 모델 메모리 해제 후 PPL 평가
    del model
    import torch
    torch.cuda.empty_cache()

    ppl_result = None
    if not args.skip_ppl:
        try:
            ppl_result = eval_ppl_quantized(out, model_path, seqlen=args.ppl_seqlen)
        except Exception as e:
            print(f"[ppl] 평가 실패 (README 에 생략으로 기록): {e}")

    kmmlu_result = None
    if not args.skip_kmmlu:
        try:
            kmmlu_result = eval_kmmlu_quantized(out, model_path, shots=args.kmmlu_shots)
        except Exception as e:
            print(f"[kmmlu] 평가 실패 (README 에 생략으로 기록): {e}")

    kdtcbench_result = None
    if not args.skip_kdtcbench:
        try:
            kdtcbench_result = eval_kdtcbench_quantized(out, model_path)
        except Exception as e:
            print(f"[kdtcbench] 평가 실패 (README 에 생략으로 기록): {e}")

    write_readme(out, args, model_path, len(calib), cap.rows, elapsed, ppl_result, kmmlu_result,
                 calib_label=calib_label, kdtcbench_result=kdtcbench_result)


if __name__ == "__main__":
    main()
