<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Eye Gaze Correction Demo</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0a0a0a;
      color: #e0e0e0;
      padding: 20px;
      min-height: 100vh;
    }
    .container { max-width: 1400px; margin: 0 auto; }
    h1 { text-align: center; margin-bottom: 20px; color: #a78bfa; }
    .replit-hint {
      background: #1e1e1e; border: 2px solid #a78bfa; padding: 12px; border-radius: 8px;
      margin-bottom: 20px; text-align: center;
    }
    .controls {
      background: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px;
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;
    }
    .control-group { display:flex; flex-direction:column; gap:8px; }
    .control-group label { font-size:14px; color:#b0b0b0; }
    button {
      padding:12px 24px; border:none; border-radius:8px; font-size:14px; font-weight:600;
      cursor:pointer; transition:all .2s;
    }
    button.primary { background:#a78bfa; color:#0a0a0a; }
    button.secondary { background:#2a2a2a; color:#e0e0e0; }
    input[type="range"] { width:100%; accent-color:#a78bfa; }
    .toggle { display:flex; align-items:center; gap:10px; }
    input[type="checkbox"] { width:20px; height:20px; accent-color:#a78bfa; cursor:pointer; }
    .video-container { position:relative; background:#000; border-radius:12px; overflow:hidden; margin-bottom:20px; }
    canvas { display:block; width:100%; height:auto; background:#000; }
    .debug-overlay {
      position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.8); padding:12px; border-radius:8px;
      font-family:monospace; font-size:13px; line-height:1.6; z-index:10;
    }
    .watermark { position:absolute; bottom:10px; right:10px; background:rgba(0,0,0,0.7); padding:8px 16px; border-radius:6px; font-size:14px; font-weight:600; color:#a78bfa; z-index:10; }
    .error-banner { background:#ff4444; color:white; padding:15px; border-radius:8px; margin-bottom:20px; display:none; }
    .error-banner.visible { display:block; }
    .value-display { font-weight:600; color:#a78bfa; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎯 Eye Gaze Correction Demo</h1>

    <div class="replit-hint">
      💡 Running on Replit? Click the "Open in new tab" button (↗️) to allow camera access.
    </div>

    <div class="error-banner" id="errorBanner"></div>

    <div class="controls">
      <div class="control-group">
        <button id="startBtn" class="primary">Start Camera</button>
      </div>

      <div class="control-group">
        <label> Strength: <span class="value-display" id="strengthValue">55</span>% </label>
        <input type="range" id="strengthSlider" min="0" max="100" value="55" />
      </div>

      <div class="control-group toggle">
        <input type="checkbox" id="naturalFalloff" checked />
        <label for="naturalFalloff">Natural Falloff</label>
      </div>

      <div class="control-group toggle">
        <input type="checkbox" id="syntheticEyes" checked />
        <label for="syntheticEyes">Synthetic Eyes (more realistic)</label>
      </div>

      <div class="control-group toggle">
        <input type="checkbox" id="splitView" />
        <label for="splitView">Split View</label>
      </div>

      <div class="control-group toggle">
        <input type="checkbox" id="watermarkToggle" checked />
        <label for="watermarkToggle">Watermark</label>
      </div>

      <div class="control-group">
        <button id="recordBtn" class="secondary" disabled>Start Recording</button>
      </div>

      <div class="control-group">
        <button id="saveBtn" class="secondary" disabled>Save Video</button>
      </div>
    </div>

    <div class="video-container">
      <canvas id="outputCanvas"></canvas>
      <div class="debug-overlay">
        <div>FPS: <span class="value-display" id="fpsDisplay">0</span></div>
        <div>Face: <span id="faceLock" class="unlocked">—</span></div>
        <div>Angle: <span class="value-display" id="angleDisplay">0.0</span>°</div>
      </div>
      <div class="watermark" id="watermark">EyeCorrect AI</div>
    </div>
  </div>

  <script type="module">
    import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

    // CONFIG
    const MAX_REDIRECT_DEG = 12;
    const EYE_PATCH_RADIUS = 28;
    const FEATHER = 18;
    const BLINK_THRESHOLD = 0.015;
    const BLINK_PAUSE_MS = 150;
    const MAX_IRIS_PIXEL_OFFSET = 12; // how many px we can nudge the synthetic iris
    const IRIS_COLORS = ['#6b4c9b', '#3c6b8f', '#5d3b2b']; // example color palettes

    // MediaPipe indices (FaceMesh style)
    const LEFT_EYE = [33, 133, 159, 158, 157, 173, 145, 144, 153, 154, 155, 7];
    const RIGHT_EYE = [362, 263, 386, 385, 384, 398, 374, 373, 380, 381, 382, 249];
    const LEFT_EYE_CORNERS = [33, 133];
    const RIGHT_EYE_CORNERS = [362, 263];

    // STATE
    let faceLandmarker = null;
    let videoElement = null;
    let canvasElement = null;
    let ctx = null;
    let stream = null;
    let animationId = null;
    let lastFrameTime = 0;
    let fps = 0;
    let faceDetected = false;
    let lastBlinkTime = 0;
    let mediaRecorder = null;
    let recordedChunks = [];
    let isRecording = false;

    // UI
    const startBtn = document.getElementById('startBtn');
    const strengthSlider = document.getElementById('strengthSlider');
    const strengthValue = document.getElementById('strengthValue');
    const naturalFalloff = document.getElementById('naturalFalloff');
    const splitView = document.getElementById('splitView');
    const watermarkToggle = document.getElementById('watermarkToggle');
    const watermark = document.getElementById('watermark');
    const recordBtn = document.getElementById('recordBtn');
    const saveBtn = document.getElementById('saveBtn');
    const errorBanner = document.getElementById('errorBanner');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const faceLock = document.getElementById('faceLock');
    const angleDisplay = document.getElementById('angleDisplay');
    const syntheticEyesToggle = document.getElementById('syntheticEyes');

    function showError(message) {
      errorBanner.textContent = message;
      errorBanner.classList.add('visible');
    }
    function hideError() {
      errorBanner.classList.remove('visible');
    }

    function px(landmarks, idx, width, height) {
      const lm = landmarks[idx];
      return { x: lm.x * width, y: lm.y * height, z: lm.z ?? 0 };
    }

    function eyeCenter(landmarks, eyeIndices, width, height) {
      let sumX = 0, sumY = 0;
      for (const idx of eyeIndices) {
        const p = px(landmarks, idx, width, height);
        sumX += p.x; sumY += p.y;
      }
      return { x: sumX / eyeIndices.length, y: sumY / eyeIndices.length };
    }

    function eyeCorners(landmarks, cornerIndices, width, height) {
      const inner = px(landmarks, cornerIndices[0], width, height);
      const outer = px(landmarks, cornerIndices[1], width, height);
      return { inner, outer };
    }

    function computeTarget(corners) {
      const midX = (corners.inner.x + corners.outer.x) / 2;
      const midY = (corners.inner.y + corners.outer.y) / 2;
      const dy = corners.outer.y - corners.inner.y;
      return { x: midX, y: midY - dy * 0.1 };
    }

    function distance(a, b) {
      const dx = b.x - a.x; const dy = b.y - a.y; return Math.sqrt(dx*dx + dy*dy);
    }

    function angle(a, b) {
      const dx = b.x - a.x; const dy = b.y - a.y;
      return (Math.atan2(Math.hypot(dx, dy), 1) * 180) / Math.PI;
    }

    function isEyeClosed(landmarks, eyeIndices, width, height) {
      const top = px(landmarks, eyeIndices[1], width, height);
      const bottom = px(landmarks, eyeIndices[4], width, height);
      const vertDist = Math.abs(bottom.y - top.y);
      const eyeWidth = distance(px(landmarks, eyeIndices[0], width, height), px(landmarks, eyeIndices[3], width, height));
      return vertDist / eyeWidth < BLINK_THRESHOLD;
    }

    // existing patch-based move (keeps original pixels) - used if syntheticEyes false
    function moveIrisPatch(ctx, src, dst, strength) {
      if (strength <= 0) return;
      const r = EYE_PATCH_RADIUS;
      const f = FEATHER;
      const sx = Math.round(src.x - r);
      const sy = Math.round(src.y - r);
      const dx = Math.round(dst.x - r);
      const dy = Math.round(dst.y - r);
      try {
        const patch = ctx.getImageData(sx, sy, r * 2, r * 2);

        // mask
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = r * 2; maskCanvas.height = r * 2;
        const maskCtx = maskCanvas.getContext('2d');
        const grad = maskCtx.createRadialGradient(r, r, r - f, r, r, r);
        grad.addColorStop(0, 'rgba(255,255,255,1)');
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        maskCtx.fillStyle = grad; maskCtx.fillRect(0,0,r*2,r*2);

        // temp patch
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = r * 2; tempCanvas.height = r * 2;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(patch, 0, 0);
        tempCtx.globalCompositeOperation = 'destination-in';
        tempCtx.drawImage(maskCanvas, 0, 0);

        // blend
        ctx.globalAlpha = strength;
        ctx.drawImage(tempCanvas, dx, dy);
        ctx.globalAlpha = 1.0;
      } catch (e) {
        // ignore if patch is out of bounds
      }
    }

    // synthetic iris renderer: draws a layered, pseudo-realistic iris + pupil + highlight
    function drawSyntheticIris(ctx, center, pixelOffset, radius, strength, eyeOpen=true, colorHex='#3c6b8f') {
      if (strength <= 0 || !eyeOpen) return;
      const r = Math.round(radius);
      const canvas = document.createElement('canvas');
      canvas.width = r * 2; canvas.height = r * 2;
      const c = canvas.getContext('2d');

      // draw iris gradient (multi-stop)
      const irisGrad = c.createRadialGradient(r - 0.6*r/3, r - r/6, r*0.2, r, r, r);
      irisGrad.addColorStop(0, '#ffffff'); // tiny bright center
      irisGrad.addColorStop(0.08, shadeColor(colorHex, -8));
      irisGrad.addColorStop(0.4, colorHex);
      irisGrad.addColorStop(1, shadeColor(colorHex, -35));
      c.fillStyle = irisGrad;
      c.beginPath(); c.arc(r, r, r, 0, Math.PI*2); c.fill();

      // add radial texture lines (subtle)
      c.globalAlpha = 0.06;
      for (let i=0;i<14;i++) {
        c.beginPath();
        c.moveTo(r, r);
        const ang = i * Math.PI * 2 / 14;
        const x = r + Math.cos(ang) * r;
        const y = r + Math.sin(ang) * r;
        c.lineTo(x, y);
        c.strokeStyle = 'rgba(0,0,0,0.08)';
        c.lineWidth = 1;
        c.stroke();
      }
      c.globalAlpha = 1.0;

      // pupil
      const pupilRadius = Math.max(6, r * 0.33);
      c.fillStyle = '#000';
      c.beginPath(); c.arc(r, r, pupilRadius, 0, Math.PI*2); c.fill();

      // highlight
      const specGrad = c.createRadialGradient(r - r*0.3, r - r*0.45, 1, r - r*0.3, r - r*0.45, r*0.8);
      specGrad.addColorStop(0, 'rgba(255,255,255,0.9)');
      specGrad.addColorStop(1, 'rgba(255,255,255,0)');
      c.fillStyle = specGrad;
      c.beginPath(); c.arc(r - r*0.28, r - r*0.45, r*0.35, 0, Math.PI*2); c.fill();

      // apply soft circular mask (feathering)
      const mask = document.createElement('canvas');
      mask.width = r * 2; mask.height = r * 2;
      const mc = mask.getContext('2d');
      const mgrad = mc.createRadialGradient(r, r, r*0.4, r, r, r);
      mgrad.addColorStop(0, 'rgba(0,0,0,1)');
      mgrad.addColorStop(1, 'rgba(0,0,0,0)');
      mc.fillStyle = mgrad;
      mc.fillRect(0,0, r*2, r*2);
      // mask the iris canvas
      c.globalCompositeOperation = 'destination-in';
      c.drawImage(mask, 0,0);

      // draw onto main context with blending
      ctx.globalAlpha = strength;
      ctx.drawImage(canvas, Math.round(center.x + pixelOffset.x - r), Math.round(center.y + pixelOffset.y - r));
      ctx.globalAlpha = 1.0;
    }

    // small utility: shade color (hex) by percent (- to darken)
    function shadeColor(hex, percent) {
      const f = hex.slice(1);
      const R = parseInt(f.substring(0,2),16);
      const G = parseInt(f.substring(2,4),16);
      const B = parseInt(f.substring(4,6),16);
      const t = percent < 0 ? 0 : 255;
      const p = Math.abs(percent)/100;
      const newR = Math.round((t - R) * p) + R;
      const newG = Math.round((t - G) * p) + G;
      const newB = Math.round((t - B) * p) + B;
      return `#${(0x1000000 + (newR<<16) + (newG<<8) + newB).toString(16).slice(1)}`;
    }

    // MEDIA PIPE INIT
    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        console.log('MediaPipe FaceLandmarker ready');
      } catch (err) {
        showError(`Failed to load MediaPipe model: ${err.message}`);
        throw err;
      }
    }

    // CAMERA
    async function startCamera() {
      try {
        hideError();
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false
        });

        videoElement = document.createElement('video');
        videoElement.srcObject = stream;
        videoElement.autoplay = true; videoElement.playsInline = true;

        await new Promise(res => videoElement.onloadedmetadata = () => { videoElement.play(); res(); });

        canvasElement = document.getElementById('outputCanvas');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        ctx = canvasElement.getContext('2d', { willReadFrequently: true });

        startBtn.textContent = 'Stop Camera';
        startBtn.classList.remove('primary'); startBtn.classList.add('secondary');
        recordBtn.disabled = false;

        requestAnimationFrame(processFrame);
      } catch (err) {
        showError(`Camera error: ${err.message}. Use Chrome/Edge and allow camera access (open in a real tab).`);
      }
    }

    function stopCamera() {
      if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      if (videoElement) { videoElement.srcObject = null; videoElement = null; }
      if (isRecording) stopRecording();
      startBtn.textContent = 'Start Camera'; startBtn.classList.remove('secondary'); startBtn.classList.add('primary');
      recordBtn.disabled = true;
    }

    // MAIN LOOP
    function processFrame(timestamp) {
      animationId = requestAnimationFrame(processFrame);
      if (!videoElement || !faceLandmarker) return;

      // fps
      if (lastFrameTime) { const delta = timestamp - lastFrameTime; fps = Math.round(1000 / delta); }
      lastFrameTime = timestamp;

      const width = canvasElement.width, height = canvasElement.height;
      ctx.clearRect(0,0,width,height);
      ctx.drawImage(videoElement, 0,0,width,height);

      // detect landmarks (synchronous API)
      const results = faceLandmarker.detectForVideo(videoElement, timestamp);

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        faceDetected = true;
        const landmarks = results.faceLandmarks[0];

        const leftClosed = isEyeClosed(landmarks, LEFT_EYE, width, height);
        const rightClosed = isEyeClosed(landmarks, RIGHT_EYE, width, height);
        const nowBlinking = leftClosed || rightClosed;
        if (nowBlinking) lastBlinkTime = timestamp;
        const timeSinceBlink = timestamp - lastBlinkTime;
        const isBlinkPaused = timeSinceBlink < BLINK_PAUSE_MS;

        const leftCenter = eyeCenter(landmarks, LEFT_EYE, width, height);
        const rightCenter = eyeCenter(landmarks, RIGHT_EYE, width, height);
        const leftCorners = eyeCorners(landmarks, LEFT_EYE_CORNERS, width, height);
        const rightCorners = eyeCorners(landmarks, RIGHT_EYE_CORNERS, width, height);
        const leftTarget = computeTarget(leftCorners);
        const rightTarget = computeTarget(rightCorners);

        const leftAngle = angle(leftCenter, leftTarget);
        const rightAngle = angle(rightCenter, rightTarget);
        const meanAngle = (leftAngle + rightAngle) / 2;

        let baseStrength = strengthSlider.value / 100;
        if (isBlinkPaused) baseStrength = 0;
        else if (naturalFalloff.checked && meanAngle > MAX_REDIRECT_DEG) {
          const excess = meanAngle - MAX_REDIRECT_DEG;
          const falloff = Math.max(0, 1 - excess / MAX_REDIRECT_DEG);
          baseStrength *= falloff;
        }

        // In split view we render original on left and corrected on right
        if (splitView.checked) {
          // left: original
          ctx.drawImage(videoElement, 0, 0, width/2, height, 0, 0, width/2, height);
          // right: original base then corrections
          ctx.drawImage(videoElement, width/2, 0, width/2, height, width/2, 0, width/2, height);

          // clip to right half for correction draws
          ctx.save();
          ctx.beginPath();
          ctx.rect(width/2, 0, width/2, height);
          ctx.clip();

          applyEyeCorrection(ctx, leftCenter, leftTarget, leftClosed, baseStrength, width, height, /*xOffset=*/0);
          applyEyeCorrection(ctx, rightCenter, rightTarget, rightClosed, baseStrength, width, height, /*xOffset=*/0);

          ctx.restore();

          // divider
          ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(width/2, 0); ctx.lineTo(width/2, height); ctx.stroke();
        } else {
          // full correction applied
          applyEyeCorrection(ctx, leftCenter, leftTarget, leftClosed, baseStrength, width, height, 0);
          applyEyeCorrection(ctx, rightCenter, rightTarget, rightClosed, baseStrength, width, height, 0);
        }

        // debug dots on eye centers
        ctx.fillStyle = '#00ff00';
        ctx.beginPath(); ctx.arc(leftCenter.x, leftCenter.y, 4, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(rightCenter.x, rightCenter.y, 4, 0, Math.PI*2); ctx.fill();

        angleDisplay.textContent = meanAngle.toFixed(1);
      } else {
        faceDetected = false;
        angleDisplay.textContent = '0.0';
      }

      fpsDisplay.textContent = fps;
      faceLock.textContent = faceDetected ? 'locked' : '—';
      faceLock.className = faceDetected ? 'locked' : 'unlocked';
      watermark.style.display = watermarkToggle.checked ? 'block' : 'none';
    }

    // helper to apply either patch-based or synthetic correction for one eye
    function applyEyeCorrection(ctx, center, target, eyeClosed, strength, canvasW, canvasH, xOffset) {
      // compute pixel target that nudges eye toward camera
      const dx = target.x - center.x;
      const dy = target.y - center.y;
      // normalized direction
      const len = Math.hypot(dx, dy) || 1;
      const dirX = dx / len, dirY = dy / len;
      // magnitude proportional to distance but clamped
      const mag = Math.min(MAX_IRIS_PIXEL_OFFSET, len * 0.6);
      const pixelOffset = { x: dirX * mag * strength, y: dirY * mag * strength };

      if (!syntheticEyesToggle.checked) {
        // move original iris patch
        const dst = { x: center.x + pixelOffset.x, y: center.y + pixelOffset.y };
        moveIrisPatch(ctx, center, dst, strength);
      } else {
        // draw synthetic iris on top (blend by strength)
        // choose color based on eye index (simple alternating)
        const color = IRIS_COLORS[Math.floor(Math.random()*IRIS_COLORS.length)];
        drawSyntheticIris(ctx, center, pixelOffset, EYE_PATCH_RADIUS, strength, !eyeClosed, color);
      }
    }

    // RECORDING
    function startRecording() {
      try {
        recordedChunks = [];
        const canvasStream = canvasElement.captureStream(30);
        const mimeTypes = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'];
        let mimeType = 'video/webm';
        for (const t of mimeTypes) if (MediaRecorder.isTypeSupported(t)) { mimeType = t; break; }

        mediaRecorder = new MediaRecorder(canvasStream, { mimeType, videoBitsPerSecond: 8000000 });
        mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) recordedChunks.push(e.data); };
        mediaRecorder.onstop = () => { saveBtn.disabled = false; };
        mediaRecorder.start();
        isRecording = true; recordBtn.textContent = 'Stop Recording';
      } catch (err) {
        showError(`Recording failed: ${err.message}`);
      }
    }

    function stopRecording() {
      if (mediaRecorder && isRecording) { mediaRecorder.stop(); isRecording = false; recordBtn.textContent = 'Start Recording'; }
    }

    function saveVideo() {
      if (!recordedChunks.length) return;
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = `eye-corrected-${Date.now()}.webm`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
      recordedChunks = []; saveBtn.disabled = true;
    }

    // EVENTS
    startBtn.addEventListener('click', async () => {
      if (!stream) {
        if (!faceLandmarker) await initMediaPipe();
        await startCamera();
      } else stopCamera();
    });

    strengthSlider.addEventListener('input', (e) => { strengthValue.textContent = e.target.value; });
    recordBtn.addEventListener('click', () => { if (!isRecording) startRecording(); else stopRecording(); });
    saveBtn.addEventListener('click', saveVideo);

    console.log('Eye Gaze Correction Demo ready');
  </script>
</body>
</html>
