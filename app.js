(() => {
  'use strict';

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const audio = document.getElementById('audio-player');
  const segmentsTrackEl = document.getElementById('segments-track');
  const analyzeBtn = document.getElementById('analyze-btn');
  const phraseInput = document.getElementById('phrase-input');
  const statusEl = document.getElementById('status');
  const resultsSummary = document.getElementById('results-summary');
  const segmentsList = document.getElementById('segments-list');
  const emotionsList = document.getElementById('emotions-list');
  const recordToggle = document.getElementById('record-toggle');
  const recordingIndicator = document.getElementById('recording-indicator');
  const recordingHint = document.getElementById('recording-hint');
  const audioMeta = document.getElementById('audio-meta');
  // phrase input removed; tone-only

  let currentBlob = null;
  let currentFileName = null;
  let audioUrl = null;
  let mediaRecorder = null;
  let recordedChunks = [];

  function setStatus(text, type) {
    statusEl.textContent = text || '';
    statusEl.className = 'status' + (type ? ' ' + type : '');
  }

  function formatTime(totalSeconds) {
    const sec = Math.max(0, Math.floor(totalSeconds || 0));
    const s = sec % 60;
    const m = Math.floor(sec / 60) % 60;
    const h = Math.floor(sec / 3600);
    const pad = (n) => n.toString().padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
  }

  function clearTrackCues() {
    if (!segmentsTrackEl || !segmentsTrackEl.track) return;
    const track = segmentsTrackEl.track;
    if (!track || !track.cues) return;
    Array.from(track.cues).forEach((c) => {
      try { track.removeCue(c); } catch (_) {}
    });
  }

  async function loadAudioFromBlob(blob, name) {
    try { if (audioUrl) URL.revokeObjectURL(audioUrl); } catch (_) {}
    currentBlob = blob;
    currentFileName = name || 'audio.' + ((blob.type.split('/')[1]) || 'webm');
    audioUrl = URL.createObjectURL(blob);
    audio.src = audioUrl;
    analyzeBtn.disabled = false;
    resultsSummary.textContent = 'Ready. Click Analyze to run.';
    segmentsList.innerHTML = '';
    clearTrackCues();
    setStatus('');
  }

  function handleFile(fileOrBlob) {
    const blob = fileOrBlob instanceof Blob ? fileOrBlob : new Blob([fileOrBlob], { type: fileOrBlob.type || 'audio/*' });
    const name = fileOrBlob.name || 'recording.' + (blob.type.split('/')[1] || 'webm');
    loadAudioFromBlob(blob, name);
    audio.focus();
  }

  function initDropzone() {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });
    ['dragenter', 'dragover'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('is-dragover');
      });
    });
    ['dragleave', 'drop'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('is-dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (!dt) return;
      const file = dt.files && dt.files[0];
      if (file && file.type && file.type.startsWith('audio/')) {
        handleFile(file);
      } else {
        setStatus('Please drop an audio file.', 'warn');
      }
    });
    fileInput.addEventListener('change', () => {
      const file = fileInput.files && fileInput.files[0];
      if (file) handleFile(file);
    });
  }

  function initRecording() {
    const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    const mediaRecorderSupported = typeof window.MediaRecorder !== 'undefined';
    if (!supported || !mediaRecorderSupported) {
      recordToggle.disabled = true;
      recordingHint.textContent = 'Recording not supported in this browser.';
      return;
    }

    let stream = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = async () => {
          const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          try { stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
          recordingIndicator.classList.remove('active');
          recordToggle.textContent = 'Start recording';
          recordToggle.dataset.state = 'idle';
          await loadAudioFromBlob(blob, 'recording.webm');
        };
        mediaRecorder.start();
        recordToggle.textContent = 'Stop recording';
        recordToggle.dataset.state = 'recording';
        recordingIndicator.classList.add('active');
      } catch (err) {
        setStatus('Microphone access denied or unavailable.', 'error');
      }
    }

    function stop() {
      try {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
      } catch (_) {}
    }

    recordToggle.addEventListener('click', () => {
      if (recordToggle.dataset.state === 'recording') stop(); else start();
    });
  }

  async function liveAnalyze(blob) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const form = new FormData();
      form.append('audio', blob, currentFileName || 'audio.webm');
      const res = await fetch('/api/analyze', { method: 'POST', body: form, signal: controller.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      return {
        extremist: data.isExtremist ?? data.extremist ?? null,
        riskScore: data.riskScore ?? data.score ?? null,
        segments: Array.isArray(data.segments) ? data.segments : []
      };
    } finally {
      clearTimeout(timeout);
    }
  }

  function renderResults(result) {
    const extremist = result.extremist;
    const score = result.textScore ?? result.riskScore;
    const segments = Array.isArray(result.segments) ? result.segments : [];

    let summary = 'Extremist: ' + (extremist === true ? 'Yes' : extremist === false ? 'No' : 'Uncertain');
    if (typeof score === 'number') summary += ` (score ${Math.round(score * 100)}/100)`;
    resultsSummary.textContent = summary;

    segmentsList.innerHTML = '';
    emotionsList.innerHTML = '';
    clearTrackCues();

    const listToShow = segments;
    if (listToShow.length === 0) {
      const li = document.createElement('li');
      li.textContent = 'No segments detected.';
      segmentsList.appendChild(li);
      return;
    }

    const track = segmentsTrackEl.track;
    listToShow.forEach((s, idx) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.className = 'segment-btn';
      btn.type = 'button';
      const suffix = typeof s.textScore === 'number' ? ` · textScore ${(Math.round(s.textScore * 1000) / 1000).toFixed(3)}` : '';
      btn.textContent = `${formatTime(s.start)} → ${formatTime(s.end)}${suffix}`;
      btn.setAttribute('aria-label', `Jump to segment ${idx + 1}`);
      btn.addEventListener('click', () => {
        audio.currentTime = Math.max(0, (s.start || 0) - 0.05);
        audio.play().catch(() => {});
      });
      li.appendChild(btn);

      if (s.text) {
        const meta = document.createElement('div');
        meta.className = 'segment-meta';
        meta.textContent = s.text;
        li.appendChild(meta);
      }

      segmentsList.appendChild(li);

      try {
        const cue = new VTTCue(s.start || 0, s.end || (s.start || 0) + 1, (s.label || `segment ${idx + 1}`));
        track.addCue(cue);
      } catch (_) {}
    });
  }

  function initAnalyze() {
    analyzeBtn.addEventListener('click', async () => {
      if (!currentBlob) return;
      const phrase = (phraseInput && phraseInput.value || '').trim();
      if (!phrase) {
        setStatus('Enter a phrase to search.', 'warn');
        return;
      }
      analyzeBtn.disabled = true;
      setStatus('Searching phrase...', 'info');
      resultsSummary.textContent = '';
      segmentsList.innerHTML = '';
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 60000);
        const form = new FormData();
        form.append('audio', currentBlob, currentFileName || 'audio.webm');
        form.append('phrase', phrase);
        const res = await fetch('/api/phrase_search', { method: 'POST', body: form, signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const segments = Array.isArray(data.segments) ? data.segments : (Array.isArray(data.matches) ? data.matches : []);
        resultsSummary.textContent = `Matches for "${data.phrase || phrase}": ${segments.length}`;
        renderMatches(segments);
        setStatus('Done.', 'success');
      } catch (err) {
        console.error(err);
        setStatus('Phrase search failed: ' + (err && err.message ? err.message : err), 'error');
      } finally {
        analyzeBtn.disabled = !currentBlob;
      }
    });
  }

  function renderMatches(matches) {
    const list = Array.isArray(matches) ? matches : [];
    segmentsList.innerHTML = '';
    clearTrackCues();
    if (list.length === 0) {
      const li = document.createElement('li');
      li.textContent = 'No matching segments.';
      segmentsList.appendChild(li);
      return;
    }
    const track = segmentsTrackEl.track;
    list.forEach((s, idx) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.className = 'segment-btn';
      btn.type = 'button';
      btn.textContent = `${formatTime(s.start)} → ${formatTime(s.end)}`;
      btn.addEventListener('click', () => {
        audio.currentTime = Math.max(0, (s.start || 0) - 0.05);
        audio.play().catch(() => {});
      });
      li.appendChild(btn);
      if (s.text) {
        const meta = document.createElement('div');
        meta.className = 'segment-meta';
        meta.textContent = s.text;
        li.appendChild(meta);
      }
      segmentsList.appendChild(li);
      try {
        const cue = new VTTCue(s.start || 0, s.end || (s.start || 0) + 1, `match ${idx + 1}`);
        track.addCue(cue);
      } catch (_) {}
    });
  }

  function initAudioMeta() {
    audio.addEventListener('loadedmetadata', () => {
      const dur = audio.duration;
      if (isFinite(dur)) audioMeta.textContent = `Duration: ${formatTime(dur)}`; else audioMeta.textContent = '';
    });
  }

  function init() {
    initDropzone();
    initRecording();
    initAnalyze();
    initAudioMeta();
  }

  window.addEventListener('DOMContentLoaded', init);
})();
