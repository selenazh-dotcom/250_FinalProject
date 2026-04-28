""" 
receiver.py

Function:
    Opens up a user interface that receives camera footage and text translation
    from the RPi over UDP. 
    Can read the receiving text out loud with a toggle function.

Usage:
    In terminal - python3 receiver.py --port 5005 --web-port 8080


Claude was used to create the HTML code for the flask app. 

"""

from __future__ import annotations

import pickle
import socket
import threading
import time
import argparse
import cv2
import numpy as np
import pyttsx3
from flask import Flask, Response, render_template_string, jsonify

# connections 

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 5005
WORDS_FILE  = "words.txt"


_latest_jpeg : bytes = b""
_latest_text : str   = "no data yet"
_frame_lock          = threading.Lock()
_text_lock           = threading.Lock()
_fps_counter         = {"frames": 0, "last": time.time(), "fps": 0.0}
_tts_enabled = False
_tts_lock    = threading.Lock()

# text to speech
def speak_word(word: str) -> None:
    """Speak a single word in a background thread."""
    try:
        engine = pyttsx3.init()
        engine.say(word)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[tts] Error: {e}")
        

# receive from UDP
def udp_listener(port: int) -> None:
    global _latest_jpeg, _latest_text

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, port))
    print(f"[receiver] Listening for UDP on {LISTEN_HOST}:{port}")

    with open(WORDS_FILE, "w") as f:
        f.write("")                         # clear file on startup

    while True:
        data, addr = sock.recvfrom(65535)
        obj = pickle.loads(data)

        if obj[0] == "TEXT":
            text = str(obj[1])
            with _text_lock:
                changed = text != _latest_text
                if changed:
                    _latest_text = text
                    if text and text != "no data yet":
                        with open(WORDS_FILE, "a") as f:
                            f.write(text + " ")

            # Speak immediately if TTS is on and the word just changed
            if changed and text and text != "no data yet":
                with _tts_lock:
                    enabled = _tts_enabled
                if enabled:
                    threading.Thread(target=speak_word, args=(text,), daemon=True).start()

            print(f"[receiver] Text: {text}")

          
        elif obj[0] == "FRAME":
            frame = cv2.imdecode(np.frombuffer(obj[1], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with _text_lock:
                text = _latest_text

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (5, 19), ((len(text) * 25) + 10, 60), 2, -1)
            cv2.putText(frame, text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            _, jpeg_buf = cv2.imencode(".jpg", frame)
            with _frame_lock:
                _latest_jpeg = jpeg_buf.tobytes()

            _fps_counter["frames"] += 1
            now = time.time()
            if now - _fps_counter["last"] >= 1.0:
                _fps_counter["fps"]    = _fps_counter["frames"]
                _fps_counter["frames"] = 0
                _fps_counter["last"]   = now

        


# App HTML code - generated from Claude

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ASL Live Feed</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f0f;
      color: #f0f0f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 24px 16px;
      gap: 20px;
    }
    h1 {
      font-size: 1.4rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      color: #aaa;
      text-transform: uppercase;
    }
    #video-wrap {
      position: relative;
      width: 100%;
      max-width: 720px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #2a2a2a;
      background: #1a1a1a;
    }
    #feed { width: 100%; display: block; }
    #status-bar {
      position: absolute;
      top: 10px; right: 12px;
      background: rgba(0,0,0,0.55);
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      color: #0f0;
    }
    #translation-box {
      width: 100%;
      max-width: 720px;
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 12px;
      padding: 20px 24px;
    }
    #translation-box h2 {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #666;
      margin-bottom: 12px;
    }
    #translation-text {
      font-size: 2rem;
      font-weight: 700;
      color: #fff;
      min-height: 2.5rem;
      word-break: break-word;
      letter-spacing: 0.04em;
    }
    #translation-text.empty {
      color: #333;
      font-style: italic;
      font-weight: 400;
      font-size: 1.2rem;
    }
    #history-box {
      width: 100%;
      max-width: 720px;
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 12px;
      padding: 16px 24px;
    }
    #history-box h2 {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #666;
      margin-bottom: 10px;
    }
    #history-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 160px;
      overflow-y: auto;
    }
    #history-list li {
      font-size: 0.95rem;
      color: #aaa;
      padding: 6px 10px;
      background: #111;
      border-radius: 6px;
    }
    #history-list li span { font-size: 0.7rem; color: #444; float: right; }

    /* ── TTS button ── */
    /* replace the old speak-btn style with this */
    #tts-toggle {
    padding: 14px 32px;
    font-size: 1rem;
    font-weight: 600;
    background: #333;
    color: #aaa;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    display: flex;
    align-items: center;
    gap: 8px;
    }
    #tts-toggle.active {
    background: #1db954;
    color: #000;
    }
    #tts-toggle:active { transform: scale(0.97); }
    #tts-status.speaking { color: #1db954; }

    #no-signal {
      display: none;
      position: absolute;
      inset: 0;
      background: #111;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
      color: #444;
      flex-direction: column;
      gap: 8px;
    }
    #no-signal.visible { display: flex; }
  </style>
</head>
<body>

  <h1>ASL Live Receiver</h1>

  <div id="video-wrap">
    <img id="feed" src="/video_feed" alt="Live feed">
    <div id="status-bar">● LIVE &nbsp;<span id="fps-display">-- fps</span></div>
    <div id="no-signal">
      <span style="font-size:2rem">📡</span>
      <span>Waiting for signal…</span>
    </div>
  </div>

  <div id="translation-box">
    <h2>Current Translation</h2>
    <div id="translation-text" class="empty">waiting…</div>
  </div>

  <div id="tts-box">
    <button id="tts-toggle" onclick="toggleTTS()">
        🔇 TTS Off
    </button>
    <span id="tts-status">TTS is off</span>
  </div>

  <div id="history-box">
    <h2>History</h2>
    <ul id="history-list"></ul>
  </div>

<script>
  const translationEl = document.getElementById("translation-text");
  const historyList   = document.getElementById("history-list");
  const fpsDisplay    = document.getElementById("fps-display");
  const noSignal      = document.getElementById("no-signal");
  const feedImg       = document.getElementById("feed");
  const speakBtn      = document.getElementById("speak-btn");
  const ttsStatus     = document.getElementById("tts-status");
  let lastText = "";

  // ── TTS button ────────────────────────────────────────────────────────────
  // replace triggerSpeak() with this
    let ttsEnabled = false;

    async function toggleTTS() {
        try {
            const r = await fetch("/tts_toggle", { method: "POST" });
            const d = await r.json();
            ttsEnabled = d.enabled;

            const btn = document.getElementById("tts-toggle");
            const status = document.getElementById("tts-status");

            if (ttsEnabled) {
            btn.textContent = "🔊 TTS On";
            btn.classList.add("active");
            status.textContent = "Speaking each new word as it arrives";
            status.classList.add("speaking");
            } else {
            btn.textContent = "🔇 TTS Off";
            btn.classList.remove("active");
            status.textContent = "TTS is off";
            status.classList.remove("speaking");
            }
        } catch (e) {
            document.getElementById("tts-status").textContent = "Error toggling TTS";
        }
    }

  // ── Poll status ───────────────────────────────────────────────────────────
  async function pollStatus() {
    try {
      const r    = await fetch("/status");
      const data = await r.json();
      const text = data.text.trim();

      if (text !== lastText) {
        if (text === "" || text === "no data yet") {
          translationEl.textContent = "waiting…";
          translationEl.classList.add("empty");
        } else {
          translationEl.textContent = text;
          translationEl.classList.remove("empty");
        }
        if (lastText && lastText !== "waiting…" && lastText !== "no data yet") {
          const li = document.createElement("li");
          const ts = new Date().toLocaleTimeString();
          li.innerHTML = `${lastText} <span>${ts}</span>`;
          historyList.prepend(li);
          while (historyList.children.length > 20)
            historyList.removeChild(historyList.lastChild);
        }
        lastText = text;
      }

      fpsDisplay.textContent = `${data.fps} fps`;
      noSignal.classList.toggle("visible", data.fps === 0);

    } catch (e) { /* not ready yet */ }
    setTimeout(pollStatus, 200);
  }

  feedImg.addEventListener("error", () => {
    setTimeout(() => { feedImg.src = "/video_feed?" + Date.now(); }, 1000);
  });

  pollStatus();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/status")
def status():
    with _text_lock:
        text = _latest_text
    return {"text": text, "fps": int(_fps_counter["fps"])}


# @app.route("/speak", methods=["POST"])
# def speak_route():
#     """Triggered by the browser's Speak button. Runs TTS in a background thread."""
#     with _tts_lock:
#         if _tts_speaking:
#             return jsonify({"status": "already_speaking"})

#     try:
#         with open(WORDS_FILE, "r") as f:
#             text = f.read().strip()
#         if not text:
#             return jsonify({"status": "empty"})
#     except FileNotFoundError:
#         return jsonify({"status": "empty"})

#     threading.Thread(target=speak, daemon=True).start()
#     return jsonify({"status": "speaking"})


@app.route("/tts_toggle", methods=["POST"])
def tts_toggle():
    global _tts_enabled
    with _tts_lock:
        _tts_enabled = not _tts_enabled
        state = _tts_enabled
    print(f"[tts] {'enabled' if state else 'disabled'}")
    return jsonify({"enabled": state})


def generate_mjpeg():
    BOUNDARY = b"--frame"
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg:
            yield (
                BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=5005, help="UDP listen port")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    args = parser.parse_args()

    udp_thread = threading.Thread(
        target=udp_listener, args=(args.port,), daemon=True
    )
    udp_thread.start()

    print(f"[receiver] Web interface → http://localhost:{args.web_port}")
    app.run(host="0.0.0.0", port=args.web_port, threaded=True)


if __name__ == "__main__":
    main()
