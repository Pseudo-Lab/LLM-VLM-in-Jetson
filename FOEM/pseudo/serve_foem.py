#!/usr/bin/env -S python -u
"""FOEM 3bit 베이스라인 모델을 FastAPI로 서빙한다.

vLLM은 GPTQ 3-bit를 지원하지 않고(Unsupported quantization config: bits=3) 이
모델(Mistral3 VLM)의 멀티모달 프로세서 경로에서 자체 버그까지 겹쳐 서빙이
불가능했다. 대신 이 세션 내내 검증된 HF transformers + GPTQModel 로딩 방식을
그대로 쓰되, 매 요청마다 재로딩하지 않도록 FastAPI lifespan에서 모델을 한 번만
로드해 상주시킨다.

  uvicorn serve_foem:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_kdtcbench import load_model  # noqa: E402  (양자화 모델 3단계 폴백 로더 재사용)
from transformers import AutoProcessor  # noqa: E402

MODEL_PATH = "/workspace/LLM-VLM-in-Jetson/FOEM/Ministral-3-3B-Instruct-2512-BF16_foem_3bit"
TOKENIZER_PATH = (
    "/root/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16"
    "/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[load] {MODEL_PATH}")
    t0 = time.time()
    hf, _ = load_model(MODEL_PATH, quant=True)
    hf.eval()
    processor = AutoProcessor.from_pretrained(TOKENIZER_PATH)
    state["model"] = hf
    state["processor"] = processor
    print(f"[load] done ({time.time() - t0:.1f}s, device={next(hf.parameters()).device})")
    yield
    state.clear()


app = FastAPI(title="FOEM 3bit Korean chat", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 1024
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    elapsed_sec: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "loaded": "model" in state}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>FOEM 3bit 한국어 챗</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 18px; color: #333; }
  #log { border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 300px;
         margin-bottom: 12px; overflow-y: auto; max-height: 60vh; }
  .msg { margin-bottom: 14px; white-space: pre-wrap; line-height: 1.5; }
  .q { color: #1a56db; font-weight: 600; }
  .a { color: #111; margin-top: 4px; }
  .meta { color: #999; font-size: 12px; margin-top: 2px; }
  form { display: flex; gap: 8px; align-items: center; }
  input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
  button { padding: 10px 18px; border: none; border-radius: 6px; background: #1a56db; color: #fff;
           font-size: 14px; cursor: pointer; }
  button:disabled { background: #9ab; cursor: default; }
  .opts { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; font-size: 13px; color: #555; }
  .opts input[type=number] { width: 70px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; }
</style>
</head>
<body>
<h1>FOEM 3bit 베이스라인 (group_size=128) &mdash; 한국어 질의</h1>
<div class="opts">
  <label>max_tokens <input id="maxTokens" type="number" value="1024" min="16" max="4096" step="16"></label>
  <label>temperature <input id="temperature" type="number" value="0.7" min="0" max="2" step="0.1"></label>
</div>
<div id="log"></div>
<form id="f">
  <input id="msg" type="text" placeholder="질문을 입력하세요" autocomplete="off" autofocus>
  <button id="send" type="submit">보내기</button>
</form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('msg');
const btn = document.getElementById('send');
const maxTokens = document.getElementById('maxTokens');
const temperature = document.getElementById('temperature');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  btn.disabled = true;

  const qDiv = document.createElement('div');
  qDiv.className = 'msg';
  qDiv.innerHTML = '<div class="q">Q. ' + escapeHtml(text) + '</div><div class="a">...생성 중</div>';
  log.appendChild(qDiv);
  log.scrollTop = log.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: text,
        max_tokens: parseInt(maxTokens.value, 10),
        temperature: parseFloat(temperature.value)
      })
    });
    const data = await res.json();
    qDiv.innerHTML = '<div class="q">Q. ' + escapeHtml(text) + '</div>'
      + '<div class="a">' + escapeHtml(data.response) + '</div>'
      + '<div class="meta">' + data.elapsed_sec.toFixed(1) + '초</div>';
  } catch (err) {
    qDiv.innerHTML += '<div class="a" style="color:red">에러: ' + err + '</div>';
  } finally {
    btn.disabled = false;
    log.scrollTop = log.scrollHeight;
    input.focus();
  }
});

function escapeHtml(s) {
  const d = document.createElement('div');
  d.innerText = s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(req: ChatRequest):
    hf = state["model"]
    processor = state["processor"]
    dev = next(hf.parameters()).device

    messages = [{"role": "user", "content": req.message}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(dev)

    t0 = time.time()
    out = hf.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature if req.temperature > 0 else 1.0,
        top_p=0.9,
    )
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    response_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    elapsed = time.time() - t0

    return ChatResponse(response=response_text, elapsed_sec=elapsed)
