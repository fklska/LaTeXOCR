from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from infer_qwen3vl_latex import DEFAULT_PROMPT, infer_image_to_latex


app = Flask(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


PAGE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Formula to LaTeX</title>
  <script>
    window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] } };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #667085;
      --line: #d0d5dd;
      --panel: #ffffff;
      --bg: #f6f7f9;
      --accent: #2f6fed;
      --accent-dark: #1f56bf;
      --danger: #b42318;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 28px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      font-weight: 750;
    }

    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.1fr) minmax(320px, 0.9fr);
      gap: 20px;
      padding: 20px 28px 28px;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }

    h2 {
      margin: 0 0 14px;
      font-size: 18px;
      line-height: 1.25;
    }

    .tools {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }

    button, .file-label {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      font-size: 14px;
      padding: 9px 12px;
    }

    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 650;
    }

    button.primary:hover { background: var(--accent-dark); }
    button:disabled { cursor: not-allowed; opacity: 0.6; }

    input[type="range"] { width: 130px; }
    input[type="file"] { display: none; }

    .canvas-wrap {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }

    canvas {
      display: block;
      width: 100%;
      height: 320px;
      touch-action: none;
      cursor: crosshair;
      background: #fff;
    }

    .upload-preview {
      max-width: 100%;
      max-height: 320px;
      display: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      object-fit: contain;
    }

    .result-box {
      min-height: 160px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 14px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      font-family: Consolas, "Courier New", monospace;
      font-size: 15px;
    }

    .preview {
      min-height: 120px;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
      overflow-x: auto;
    }

    .hint, .status {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }

    .status.error { color: var(--danger); }

    @media (max-width: 920px) {
      header { padding: 16px; }
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      canvas { height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Formula to LaTeX</h1>
    <span class="hint">Нарисуйте формулу мышкой или загрузите изображение.</span>
  </header>

  <main>
    <div>
      <section>
        <h2>Рукописный ввод</h2>
        <div class="tools">
          <button id="penSmall" type="button">Тонко</button>
          <button id="penMedium" type="button">Средне</button>
          <button id="penLarge" type="button">Жирно</button>
          <button id="clearCanvas" type="button">Очистить</button>
          <button id="recognizeCanvas" class="primary" type="button">Распознать</button>
        </div>
        <div class="canvas-wrap">
          <canvas id="formulaCanvas"></canvas>
        </div>
      </section>

      <section style="margin-top: 20px;">
        <h2>Загрузка изображения</h2>
        <div class="tools">
          <label class="file-label" for="imageInput">Выбрать файл</label>
          <button id="recognizeUpload" class="primary" type="button">Распознать</button>
        </div>
        <input id="imageInput" type="file" accept="image/png,image/jpeg,image/webp,image/bmp">
        <p id="fileName" class="hint">Файл не выбран.</p>
        <img id="uploadPreview" class="upload-preview" alt="Предпросмотр изображения">
      </section>
    </div>

    <section>
      <h2>LaTeX</h2>
      <p id="status" class="status">Результат появится здесь после распознавания.</p>
      <pre id="latexResult" class="result-box"></pre>
      <div class="tools" style="margin-top: 12px;">
        <button id="copyResult" type="button">Копировать</button>
      </div>
      <h2 style="margin-top: 20px;">Предпросмотр</h2>
      <div id="mathPreview" class="preview"></div>
    </section>
  </main>

  <script>
    const canvas = document.getElementById("formulaCanvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("latexResult");
    const previewEl = document.getElementById("mathPreview");
    const imageInput = document.getElementById("imageInput");
    const uploadPreview = document.getElementById("uploadPreview");
    const fileName = document.getElementById("fileName");
    let drawing = false;
    let penWidth = 5;

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const oldImage = ctx.getImageData(0, 0, canvas.width || 1, canvas.height || 1);
      canvas.width = Math.floor(rect.width * window.devicePixelRatio);
      canvas.height = Math.floor(rect.height * window.devicePixelRatio);
      ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
      clearCanvas();
      if (oldImage.width > 1) {
        const temp = document.createElement("canvas");
        temp.width = oldImage.width;
        temp.height = oldImage.height;
        temp.getContext("2d").putImageData(oldImage, 0, 0);
        ctx.drawImage(temp, 0, 0, rect.width, rect.height);
      }
    }

    function clearCanvas() {
      const rect = canvas.getBoundingClientRect();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, rect.width, rect.height);
    }

    function point(event) {
      const rect = canvas.getBoundingClientRect();
      const source = event.touches ? event.touches[0] : event;
      return {
        x: source.clientX - rect.left,
        y: source.clientY - rect.top
      };
    }

    function startDraw(event) {
      event.preventDefault();
      drawing = true;
      const p = point(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
    }

    function draw(event) {
      if (!drawing) return;
      event.preventDefault();
      const p = point(event);
      ctx.lineWidth = penWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "#111827";
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }

    function stopDraw() {
      drawing = false;
      ctx.closePath();
    }

    function setBusy(message) {
      statusEl.textContent = message;
      statusEl.className = "status";
      document.querySelectorAll("button").forEach(button => button.disabled = true);
    }

    function setReady(message, isError = false) {
      statusEl.textContent = message;
      statusEl.className = isError ? "status error" : "status";
      document.querySelectorAll("button").forEach(button => button.disabled = false);
    }

    function showResult(latex) {
      resultEl.textContent = latex;
      previewEl.textContent = "";
      previewEl.innerHTML = "\\\\[" + latex + "\\\\]";
      if (window.MathJax) {
        MathJax.typesetPromise([previewEl]);
      }
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Ошибка распознавания.");
      return data;
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    window.addEventListener("touchend", stopDraw);

    document.getElementById("penSmall").addEventListener("click", () => penWidth = 3);
    document.getElementById("penMedium").addEventListener("click", () => penWidth = 5);
    document.getElementById("penLarge").addEventListener("click", () => penWidth = 8);
    document.getElementById("clearCanvas").addEventListener("click", () => {
      clearCanvas();
      resultEl.textContent = "";
      previewEl.textContent = "";
      setReady("Поле очищено.");
    });

    document.getElementById("recognizeCanvas").addEventListener("click", async () => {
      try {
        setBusy("Распознаю рукописную формулу...");
        const imageData = canvas.toDataURL("image/png");
        const data = await postJson("/api/recognize-drawing", { image: imageData });
        showResult(data.latex);
        setReady("Готово.");
      } catch (error) {
        setReady(error.message, true);
      }
    });

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (!file) return;
      fileName.textContent = file.name;
      uploadPreview.src = URL.createObjectURL(file);
      uploadPreview.style.display = "block";
    });

    document.getElementById("recognizeUpload").addEventListener("click", async () => {
      const file = imageInput.files[0];
      if (!file) {
        setReady("Сначала выберите изображение.", true);
        return;
      }

      try {
        setBusy("Распознаю изображение...");
        const form = new FormData();
        form.append("image", file);
        const response = await fetch("/api/recognize-upload", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Ошибка распознавания.");
        showResult(data.latex);
        setReady("Готово.");
      } catch (error) {
        setReady(error.message, true);
      }
    });

    document.getElementById("copyResult").addEventListener("click", async () => {
      if (!resultEl.textContent.trim()) return;
      await navigator.clipboard.writeText(resultEl.textContent);
      setReady("LaTeX скопирован.");
    });

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""


def save_data_url_image(image_data: str) -> Path:
    if "," not in image_data:
        raise ValueError("Некорректные данные изображения.")

    header, encoded = image_data.split(",", 1)
    if "image/png" not in header:
        raise ValueError("Рукописный ввод должен быть PNG.")

    image_bytes = base64.b64decode(encoded)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(image_bytes)
    temp_file.close()
    return Path(temp_file.name)


def save_uploaded_image(uploaded_file) -> Path:
    suffix = Path(uploaded_file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Поддерживаются PNG, JPG, JPEG, WEBP и BMP.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded_file.save(temp_file)
    temp_file.close()
    return Path(temp_file.name)


def recognize_image(image_path: Path) -> str:
    try:
        return infer_image_to_latex(image_path, prompt=DEFAULT_PROMPT)
    finally:
        image_path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return "ok"


@app.post("/api/recognize-drawing")
def recognize_drawing():
    try:
        payload = request.get_json(force=True)
        image_path = save_data_url_image(payload.get("image", ""))
        latex = recognize_image(image_path)
        return jsonify({"latex": latex})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/recognize-upload")
def recognize_upload():
    try:
        uploaded_file = request.files.get("image")
        if uploaded_file is None:
            raise ValueError("Файл не выбран.")

        image_path = save_uploaded_image(uploaded_file)
        latex = recognize_image(image_path)
        return jsonify({"latex": latex})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
