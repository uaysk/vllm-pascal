# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Web demo for realtime transcription using file upload.

This app does not call the offline transcription endpoint. Instead it mimics a
microphone by:
1. decoding the uploaded audio,
2. resampling to 16kHz mono PCM16,
3. streaming audio chunks to the realtime WebSocket endpoint at real-time pace.

Start a realtime-capable vLLM server first, for example:

    vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
        --host 0.0.0.0 \
        --port 8000 \
        --enforce-eager \
        --tokenizer-mode mistral \
        --config-format mistral \
        --load-format mistral \
        --gpu-memory-utilization 0.8

Then run this demo:

    python examples/online_serving/openai_realtime_upload_web_demo.py \
        --host 0.0.0.0 \
        --port 3003 \
        --backend-host 127.0.0.1 \
        --backend-port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import librosa
import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Realtime Transcription Demo</title>
  <style>
    :root {
      --bg: #f2efe8;
      --panel: #fffdf7;
      --ink: #1a1a1a;
      --muted: #6d665c;
      --line: #ded6c8;
      --accent: #0b6e4f;
      --accent-2: #d95d39;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iosevka Aile", "IBM Plex Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217,93,57,0.15), transparent 30%),
        radial-gradient(circle at top right, rgba(11,110,79,0.18), transparent 35%),
        linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    p.lead {
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 16px;
      max-width: 720px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(300px, 420px) minmax(320px, 1fr);
      gap: 18px;
    }
    .card {
      background: rgba(255, 253, 247, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 40px rgba(20, 20, 20, 0.06);
      backdrop-filter: blur(10px);
    }
    .stack { display: grid; gap: 14px; }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    input[type="file"], input[type="text"], input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 13px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), #0e8b63);
    }
    button[disabled] {
      cursor: wait;
      filter: grayscale(0.3);
      opacity: 0.7;
    }
    .status {
      padding: 12px 14px;
      border-radius: 12px;
      background: #f7f2e8;
      border: 1px solid var(--line);
      min-height: 48px;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .transcript {
      min-height: 360px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 16px;
      font-size: 16px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .pill {
      border-radius: 999px;
      padding: 8px 12px;
      background: #f7f2e8;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .error { color: var(--accent-2); }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Realtime Upload Demo</h1>
    <p class="lead">
      Upload an audio file, and the demo will stream it to the vLLM realtime
      WebSocket endpoint as if it were microphone input. You should see the same
      incremental transcription behavior as live audio playback.
    </p>
    <div class="grid">
      <form id="form" class="card stack">
        <label>
          Audio File
          <input id="audio" name="audio" type="file" accept="audio/*" required>
        </label>
        <label>
          Model
          <input id="model" name="model" type="text" value="">
        </label>
        <div class="row">
          <label>
            Chunk Size (ms)
            <input id="chunk_ms" name="chunk_ms" type="number" min="20" max="2000" value="1000">
          </label>
          <label>
            Realtime Factor
            <input id="realtime_factor" name="realtime_factor" type="number" min="0.1" max="8" step="0.1" value="1.0">
          </label>
        </div>
        <button id="submit" type="submit">Start Realtime Transcription</button>
        <div id="status" class="status">Idle.</div>
      </form>
      <div class="card">
        <div id="transcript" class="transcript"></div>
        <div class="meta">
          <div id="meta-chunks" class="pill">Chunks: 0/0</div>
          <div id="meta-elapsed" class="pill">Elapsed: 0.0s</div>
          <div id="meta-final" class="pill">State: idle</div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const form = document.getElementById("form");
    const submit = document.getElementById("submit");
    const statusEl = document.getElementById("status");
    const transcriptEl = document.getElementById("transcript");
    const metaChunks = document.getElementById("meta-chunks");
    const metaElapsed = document.getElementById("meta-elapsed");
    const metaFinal = document.getElementById("meta-final");
    const modelInput = document.getElementById("model");
    modelInput.value = "%MODEL%";

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.classList.toggle("error", isError);
    }

    async function* readNdjson(stream) {
      const decoder = new TextDecoder();
      const reader = stream.getReader();
      let buffer = "";
      while (true) {
        const {value, done} = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, {stream: true});
        let idx;
        while ((idx = buffer.indexOf("\\n")) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) {
            continue;
          }
          yield JSON.parse(line);
        }
      }
      buffer = buffer.trim();
      if (buffer) {
        yield JSON.parse(buffer);
      }
    }

    function interleaveToMono(audioBuffer) {
      if (audioBuffer.numberOfChannels === 1) {
        return audioBuffer.getChannelData(0);
      }
      const length = audioBuffer.length;
      const mono = new Float32Array(length);
      for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
        const data = audioBuffer.getChannelData(ch);
        for (let i = 0; i < length; i++) {
          mono[i] += data[i];
        }
      }
      for (let i = 0; i < length; i++) {
        mono[i] /= audioBuffer.numberOfChannels;
      }
      return mono;
    }

    function resampleLinear(input, inputRate, outputRate) {
      if (inputRate === outputRate) {
        return input;
      }
      const outputLength = Math.max(1, Math.round(input.length * outputRate / inputRate));
      const output = new Float32Array(outputLength);
      const ratio = inputRate / outputRate;
      for (let i = 0; i < outputLength; i++) {
        const pos = i * ratio;
        const left = Math.floor(pos);
        const right = Math.min(left + 1, input.length - 1);
        const frac = pos - left;
        output[i] = input[left] * (1 - frac) + input[right] * frac;
      }
      return output;
    }

    function encodeWavPcm16(samples, sampleRate) {
      const bytesPerSample = 2;
      const blockAlign = bytesPerSample;
      const byteRate = sampleRate * blockAlign;
      const dataSize = samples.length * bytesPerSample;
      const buffer = new ArrayBuffer(44 + dataSize);
      const view = new DataView(buffer);
      const writeString = (offset, text) => {
        for (let i = 0; i < text.length; i++) {
          view.setUint8(offset + i, text.charCodeAt(i));
        }
      };

      writeString(0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, byteRate, true);
      view.setUint16(32, blockAlign, true);
      view.setUint16(34, 16, true);
      writeString(36, "data");
      view.setUint32(40, dataSize, true);

      let offset = 44;
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
      }
      return new Blob([buffer], {type: "audio/wav"});
    }

    async function convertFileToWavBlob(file, targetRate = 16000) {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      try {
        const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        const mono = interleaveToMono(decoded);
        const resampled = resampleLinear(mono, decoded.sampleRate, targetRate);
        return encodeWavPcm16(resampled, targetRate);
      } finally {
        await audioContext.close();
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      transcriptEl.textContent = "";
      metaChunks.textContent = "Chunks: 0/0";
      metaElapsed.textContent = "Elapsed: 0.0s";
      metaFinal.textContent = "State: starting";
      setStatus("Opening realtime session...");
      submit.disabled = true;

      try {
        const file = document.getElementById("audio").files[0];
        if (!file) {
          throw new Error("No audio file selected.");
        }
        setStatus("Decoding audio in browser and converting to 16kHz WAV...");
        const wavBlob = await convertFileToWavBlob(file, 16000);
        const body = new FormData(form);
        body.set("audio", wavBlob, `${file.name || "upload"}.wav`);

        const response = await fetch("/transcribe", {
          method: "POST",
          body,
        });
        if (!response.ok || !response.body) {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }

        for await (const event of readNdjson(response.body)) {
          if (event.type === "status") {
            setStatus(event.message || "Working...");
          } else if (event.type === "progress") {
            metaChunks.textContent = `Chunks: ${event.sent_chunks}/${event.total_chunks}`;
            metaElapsed.textContent = `Elapsed: ${event.elapsed_s.toFixed(1)}s`;
          } else if (event.type === "delta") {
            transcriptEl.textContent = event.text;
          } else if (event.type === "done") {
            transcriptEl.textContent = event.text;
            metaFinal.textContent = "State: done";
            metaElapsed.textContent = `Elapsed: ${event.elapsed_s.toFixed(1)}s`;
            setStatus("Realtime transcription completed.");
          } else if (event.type === "error") {
            metaFinal.textContent = "State: error";
            setStatus(event.message || "Transcription failed.", true);
          }
        }
      } catch (err) {
        metaFinal.textContent = "State: error";
        setStatus(err.message || String(err), true);
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class DemoConfig:
    model: str
    backend_host: str
    backend_port: int

    @property
    def websocket_url(self) -> str:
        return f"ws://{self.backend_host}:{self.backend_port}/v1/realtime"


def _create_app(config: DemoConfig) -> FastAPI:
    app = FastAPI(title="Realtime Upload Demo")
    app.state.demo_config = config

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML.replace("%MODEL%", config.model)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(
        audio: UploadFile = File(...),
        model: str | None = Form(None),
        chunk_ms: int = Form(1000),
        realtime_factor: float = Form(1.0),
    ) -> StreamingResponse:
        if chunk_ms < 20 or chunk_ms > 2000:
            raise HTTPException(status_code=400, detail="chunk_ms must be 20..2000")
        if realtime_factor <= 0 or realtime_factor > 8:
            raise HTTPException(
                status_code=400, detail="realtime_factor must be > 0 and <= 8"
            )

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        selected_model = model or config.model

        async def event_stream() -> AsyncIterator[bytes]:
            async for event in _stream_transcription_events(
                config=config,
                audio_bytes=audio_bytes,
                filename=audio.filename or "upload.bin",
                model=selected_model,
                chunk_ms=chunk_ms,
                realtime_factor=realtime_factor,
            ):
                yield (json.dumps(event, ensure_ascii=True) + "\n").encode("utf-8")

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
        )

    return app


def _load_audio_bytes(audio_bytes: bytes, suffix: str) -> np.ndarray:
    fd, tmp_path = tempfile.mkstemp(prefix="vllm-realtime-upload-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


async def _maybe_wait_for_session_updated(ws) -> None:
    try:
        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
        event = json.loads(message)
        if event.get("type") not in {"session.updated", "session.created"}:
            return
    except TimeoutError:
        return


async def _stream_transcription_events(
    *,
    config: DemoConfig,
    audio_bytes: bytes,
    filename: str,
    model: str,
    chunk_ms: int,
    realtime_factor: float,
) -> AsyncIterator[dict[str, object]]:
    suffix = Path(filename).suffix or ".bin"
    try:
        audio = _load_audio_bytes(audio_bytes, suffix=suffix)
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        yield {"type": "error", "message": f"Failed to decode audio: {detail}"}
        return

    samples_per_chunk = max(1, SAMPLE_RATE * chunk_ms // 1000)
    pcm16 = (audio * 32767.0).astype(np.int16)
    total_chunks = (len(pcm16) + samples_per_chunk - 1) // samples_per_chunk

    yield {
        "type": "status",
        "message": (
            f"Decoded {filename} to {len(audio) / SAMPLE_RATE:.2f}s mono PCM16 "
            f"at {SAMPLE_RATE}Hz. Connecting to {config.websocket_url}..."
        ),
    }

    try:
        async with websockets.connect(config.websocket_url, max_size=None) as ws:
            first_message = json.loads(await ws.recv())
            if first_message.get("type") != "session.created":
                yield {
                    "type": "error",
                    "message": f"Unexpected first event: {first_message}",
                }
                return

            yield {"type": "status", "message": "Realtime session created."}
            await ws.send(json.dumps({"type": "session.update", "model": model}))
            await _maybe_wait_for_session_updated(ws)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            transcript = ""
            started_at = asyncio.get_event_loop().time()
            recv_done = asyncio.Event()
            event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

            async def sender() -> None:
                for idx in range(total_chunks):
                    start = idx * samples_per_chunk
                    stop = start + samples_per_chunk
                    chunk = pcm16[start:stop]
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk.tobytes()).decode(
                                    "utf-8"
                                ),
                            }
                        )
                    )
                    await event_queue.put(
                        {
                            "type": "progress",
                            "sent_chunks": idx + 1,
                            "total_chunks": total_chunks,
                            "elapsed_s": asyncio.get_event_loop().time() - started_at,
                        }
                    )
                    await asyncio.sleep((len(chunk) / SAMPLE_RATE) / realtime_factor)

                await ws.send(
                    json.dumps({"type": "input_audio_buffer.commit", "final": True})
                )

            async def receiver() -> None:
                nonlocal transcript
                async for message in ws:
                    event = json.loads(message)
                    event_type = event.get("type")
                    if event_type == "transcription.delta":
                        transcript += str(event.get("delta", ""))
                        await event_queue.put(
                            {
                                "type": "delta",
                                "text": transcript,
                                "delta": event.get("delta", ""),
                            }
                        )
                    elif event_type == "transcription.done":
                        await event_queue.put(
                            {
                                "type": "done",
                                "text": str(event.get("text", transcript)),
                                "usage": event.get("usage"),
                                "elapsed_s": (
                                    asyncio.get_event_loop().time() - started_at
                                ),
                            }
                        )
                        recv_done.set()
                        break
                    elif event_type == "error":
                        await event_queue.put(
                            {
                                "type": "error",
                                "message": str(event.get("error", event)),
                            }
                        )
                        recv_done.set()
                        break

            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            while True:
                event = await event_queue.get()
                yield event
                if event["type"] in {"done", "error"}:
                    break

            await recv_done.wait()
            await sender_task
            await receiver_task
    except Exception as exc:
        yield {"type": "error", "message": f"Realtime demo failed: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime upload web demo")
    parser.add_argument("--host", default="0.0.0.0", help="Demo bind host")
    parser.add_argument("--port", type=int, default=3003, help="Demo bind port")
    parser.add_argument(
        "--backend-host", default="127.0.0.1", help="vLLM realtime server host"
    )
    parser.add_argument(
        "--backend-port", type=int, default=8000, help="vLLM realtime server port"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name")
    args = parser.parse_args()

    app = _create_app(
        DemoConfig(
            model=args.model,
            backend_host=args.backend_host,
            backend_port=args.backend_port,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
