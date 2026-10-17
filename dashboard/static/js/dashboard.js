/**
 * ==============================================================================
 * VitalsHub - Unified Vital Signs Monitoring Controller (rPPG & Chest Motion)
 * ==============================================================================
 */

// Global State
let currentTab = "tab-overview";
let soundEnabled = true;
let isConnected = false;

// Thresholds (Default values, will be synced from server)
let thresholds = {
  hr_low: 50,
  hr_high: 100,
  rr_low: 10,
  rr_high: 24,
};

// Charts instances
let chartBVP = null;
let chartResp = null;
let chartFFT = null;
let chartHistory = null;
let chartChestWave = null;

// Audio Context for Client-Side Clinical Warning Beep
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playWarningBeep(freq = 2400, type = "sine", duration = 0.25) {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (err) {
    console.warn("[Audio] Không thể phát âm thanh:", err);
  }
}

// ==============================================================================
// 1. CLOCK & TAB NAVIGATION CONTROLLER
// ==============================================================================
function initClock() {
  const clockEl = document.getElementById("top-clock");
  function tick() {
    const now = new Date();
    if (clockEl) {
      clockEl.textContent = now.toLocaleTimeString("vi-VN", { hour12: false });
    }
  }
  tick();
  setInterval(tick, 1000);
}

function updateVideoStreams(targetId) {
  const rppgImg = document.getElementById("rppg-video-stream");
  const chestImg = document.getElementById("chest-video-stream");
  const unifiedImg = document.getElementById("unified-video-stream");

  if (targetId === "tab-unified") {
    // Kích hoạt luồng 1-CAM ALL-IN-ONE (Mặt + Ngực đồng thời)
    if (unifiedImg) {
      const currentSrc = unifiedImg.getAttribute("src") || "";
      if (!currentSrc.includes("/api/video_feed/unified")) {
        unifiedImg.src = "/api/video_feed/unified";
      }
    }
    if (rppgImg) rppgImg.src = "";
    if (chestImg) chestImg.src = "";
  } else if (targetId === "tab-rppg") {
    // Chỉ kích hoạt luồng rPPG khi xem tab rPPG
    if (rppgImg) {
      const currentSrc = rppgImg.getAttribute("src") || "";
      if (!currentSrc.includes("/api/video_feed/rppg")) {
        rppgImg.src = "/api/video_feed/rppg";
      }
    }
    if (chestImg) chestImg.src = "";
    if (unifiedImg) unifiedImg.src = "";
  } else if (targetId === "tab-chest") {
    // Chỉ kích hoạt luồng Ngực & YOLO Pose khi xem tab Ngực
    if (chestImg) {
      const currentSrc = chestImg.getAttribute("src") || "";
      if (!currentSrc.includes("/api/video_feed/chest")) {
        chestImg.src = "/api/video_feed/chest";
      }
    }
    if (rppgImg) rppgImg.src = "";
    if (unifiedImg) unifiedImg.src = "";
  } else {
    // Khi xem Tổng quan, Lịch sử, Cài đặt: Ngắt tất cả stream camera
    // CPU được nghỉ ngơi hoàn toàn, đồ thị và số liệu sinh hiệu vẫn cập nhật siêu mượt 60 FPS qua SSE
    if (rppgImg) rppgImg.src = "";
    if (chestImg) chestImg.src = "";
    if (unifiedImg) unifiedImg.src = "";
  }
}

function initTabs() {
  const tabButtons = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      if (!targetId) return;

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
      currentTab = targetId;

      // Quản lý luồng video thông minh theo tab
      updateVideoStreams(targetId);

      // Trigger redraw of charts on tab switch if needed
      if (targetId === "tab-records" && chartHistory) {
        chartHistory.resize();
      }
      if (targetId === "tab-rppg" && chartFFT) {
        chartFFT.resize();
      }
      if (targetId === "tab-chest" && chartChestWave) {
        chartChestWave.resize();
      }
    });
  });

  // Mặc định ban đầu ở Tab Tổng quan: Tắt luồng video trực tiếp để giao diện mượt nhất
  updateVideoStreams("tab-overview");
}

// ==============================================================================
// 2. OSCILLOSCOPE CHARTS INITIALIZATION (Chart.js)
// ==============================================================================
const MAX_WAVE_POINTS = 90;

function initOscilloscopes() {
  const ctxBVP = document.getElementById("chart-bvp-wave");
  const ctxResp = document.getElementById("chart-resp-wave");
  const ctxFFT = document.getElementById("chart-fft-spectrum");

  // Initial wave dummy data
  const initialLabels = Array.from({ length: MAX_WAVE_POINTS }, (_, i) => i);
  const initialBVP = Array.from({ length: MAX_WAVE_POINTS }, () => 0);
  const initialResp = Array.from({ length: MAX_WAVE_POINTS }, () => 0);

  // 1. BVP Cardiac Pulse Wave
  if (ctxBVP) {
    chartBVP = new Chart(ctxBVP, {
      type: "line",
      data: {
        labels: initialLabels,
        datasets: [
          {
            label: "BVP Cardiac Pulse",
            data: initialBVP,
            borderColor: "#FF3366",
            borderWidth: 2.2,
            backgroundColor: "rgba(255, 51, 102, 0.08)",
            fill: true,
            tension: 0.35,
            pointRadius: 0,
            pointHoverRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { display: false },
          y: {
            display: false,
            suggestedMin: -2.5,
            suggestedMax: 2.5,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
    });
  }

  // 2. Respiratory Waveform
  if (ctxResp) {
    chartResp = new Chart(ctxResp, {
      type: "line",
      data: {
        labels: initialLabels,
        datasets: [
          {
            label: "Respiratory Wave",
            data: initialResp,
            borderColor: "#00D2B4",
            borderWidth: 2.2,
            backgroundColor: "rgba(0, 210, 180, 0.08)",
            fill: true,
            tension: 0.45,
            pointRadius: 0,
            pointHoverRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { display: false },
          y: {
            display: false,
            suggestedMin: -2.5,
            suggestedMax: 2.5,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
    });
  }

  // 3. FFT Frequency Spectrum
  if (ctxFFT) {
    const fftFreqs = Array.from({ length: 40 }, (_, i) => (0.1 + i * 0.07).toFixed(2));
    const fftMags = Array.from({ length: 40 }, () => 0.05);

    chartFFT = new Chart(ctxFFT, {
      type: "bar",
      data: {
        labels: fftFreqs,
        datasets: [
          {
            label: "Công suất phổ (Power)",
            data: fftMags,
            backgroundColor: (context) => {
              const val = context.parsed ? context.parsed.x : 0;
              // Highlight cardiac band in pink, respiratory in cyan
              if (val >= 0.75 && val <= 2.5) return "rgba(255, 51, 102, 0.75)";
              return "rgba(0, 210, 180, 0.75)";
            },
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#94A3B8",
              font: { family: "JetBrains Mono", size: 9.5 },
              maxTicksLimit: 10,
            },
            title: {
              display: true,
              text: "Tần số (Hz)",
              color: "#64748B",
              font: { size: 10 },
            },
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { display: false },
            suggestedMax: 1.0,
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
  }

  // 4. Chest Motion Waveform (Tab 3: Ngực & YOLO)
  const ctxChest = document.getElementById("chart-chest-wave");
  if (ctxChest) {
    chartChestWave = new Chart(ctxChest, {
      type: "line",
      data: {
        labels: initialLabels,
        datasets: [
          {
            label: "Chest Motion dY",
            data: initialResp,
            borderColor: "#6366F1",
            borderWidth: 2.2,
            backgroundColor: "rgba(99, 102, 241, 0.1)",
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { display: false },
          y: {
            display: false,
            suggestedMin: -3.0,
            suggestedMax: 3.0,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
    });
  }
}

// Helper to push wave data smoothly
let waveStep = 0;
function updateOscilloscopes(bvpPoints, respPoints, hr, rr, chestBpm, chestMotion) {
  waveStep++;
  const f_pulse = (hr || 74.0) / 60.0;
  const f_resp = (rr || 16.0) / 60.0;
  const f_chest = (chestBpm || 16.0) / 60.0;
  const t = waveStep * 0.05;

  // If backend provided wave buffer points, use them; otherwise generate continuous physiological waveform
  let newBvpVal = Math.sin(2 * Math.PI * f_pulse * t) * 1.5 + Math.sin(4 * Math.PI * f_pulse * t) * 0.4;
  let newRespVal = Math.sin(2 * Math.PI * f_resp * t) * 1.8;
  let newChestVal = (chestMotion !== undefined && chestMotion !== 0.0) 
    ? chestMotion * 3.0 
    : Math.sin(2 * Math.PI * f_chest * t) * 1.6;

  if (bvpPoints && bvpPoints.length > 0) {
    newBvpVal = bvpPoints[bvpPoints.length - 1];
  }
  if (respPoints && respPoints.length > 0) {
    newRespVal = respPoints[respPoints.length - 1];
  }

  if (chartBVP) {
    const data = chartBVP.data.datasets[0].data;
    data.push(newBvpVal);
    if (data.length > MAX_WAVE_POINTS) data.shift();
    chartBVP.update();
  }

  if (chartResp) {
    const data = chartResp.data.datasets[0].data;
    data.push(newRespVal);
    if (data.length > MAX_WAVE_POINTS) data.shift();
    chartResp.update();
  }

  if (chartChestWave) {
    const data = chartChestWave.data.datasets[0].data;
    data.push(newChestVal);
    if (data.length > MAX_WAVE_POINTS) data.shift();
    chartChestWave.update();
  }

  // Update FFT bar peak simulation
  if (chartFFT && waveStep % 3 === 0) {
    const freqs = chartFFT.data.labels;
    const newMags = freqs.map((fStr) => {
      const f = parseFloat(fStr);
      let power = 0.05;
      const distPulse = Math.abs(f - f_pulse);
      const distResp = Math.abs(f - f_resp);
      if (distPulse < 0.15) power += Math.max(0, 0.85 - distPulse * 4);
      if (distResp < 0.1) power += Math.max(0, 0.7 - distResp * 5);
      return power + Math.random() * 0.05;
    });
    chartFFT.data.datasets[0].data = newMags;
    chartFFT.update();
  }
}

// ==============================================================================
// 3. HISTORICAL TREND CHART (Chart.js + Zoom)
// ==============================================================================
function initHistoryChart(records = []) {
  const canvas = document.getElementById("chart-history");
  if (!canvas) return;

  const reversed = [...records].reverse();
  const labels = reversed.map((r) => {
    const dt = new Date(r.recorded_at || r.created_at || Date.now());
    return dt.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  });

  const hrVals = reversed.map((r) => (r.heart_rate ? Number(r.heart_rate) : null));
  const rrVals = reversed.map((r) => {
    const v = r.respiration_rate || r.chest_bpm || r.bpm;
    return v ? Number(v) : null;
  });

  if (chartHistory) {
    chartHistory.data.labels = labels;
    chartHistory.data.datasets[0].data = hrVals;
    chartHistory.data.datasets[1].data = rrVals;
    chartHistory.update();
    return;
  }

  chartHistory = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Nhịp tim (HR BPM)",
          data: hrVals,
          borderColor: "#FF3366",
          backgroundColor: "rgba(255, 51, 102, 0.1)",
          fill: false,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
        },
        {
          label: "Nhịp thở (RR RPM / BPM)",
          data: rrVals,
          borderColor: "#00D2B4",
          backgroundColor: "rgba(0, 210, 180, 0.1)",
          fill: false,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#94A3B8", font: { family: "Inter", size: 11.5 } },
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.95)",
          titleColor: "#F8FAFC",
          bodyColor: "#94A3B8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          bodyFont: { family: "JetBrains Mono" },
        },
        zoom: {
          pan: { enabled: true, mode: "x" },
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x" },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 }, maxTicksLimit: 12 },
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748B", font: { family: "JetBrains Mono", size: 10 } },
          suggestedMin: 10,
          suggestedMax: 140,
        },
      },
    },
  });

  const resetBtn = document.getElementById("btn-reset-history-zoom");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (chartHistory) chartHistory.resetZoom();
    });
  }
}

// ==============================================================================
// 4. HERO VITALS & UI STATE UPDATE
// ==============================================================================
function updateHeroVitals(data) {
  const hr = parseFloat(data.heart_rate || 0);
  const rr = parseFloat(data.respiration_rate || 0);
  const chest = parseFloat(data.chest_bpm || data.bpm || rr);
  const fps = parseFloat(data.fps || 30);

  // 1. Elements
  const elHR = document.getElementById("val-heart-rate");
  const elRR = document.getElementById("val-resp-rate");
  const elChest = document.getElementById("val-chest-bpm");
  const elFps = document.getElementById("banner-fps");
  const elHudFps = document.getElementById("hud-fps");
  const heartSvg = document.getElementById("heart-svg");
  const elSyncTime = document.getElementById("last-updated-text");

  if (elHR) elHR.textContent = hr > 0 ? hr.toFixed(1) : "--";
  if (elRR) elRR.textContent = rr > 0 ? rr.toFixed(1) : "--";
  if (elChest) elChest.textContent = chest > 0 ? chest.toFixed(1) : "--";
  if (elFps) elFps.textContent = `${fps.toFixed(1)} FPS`;
  if (elHudFps) elHudFps.textContent = fps.toFixed(1);

  const elHudChestFps = document.getElementById("hud-chest-fps");
  const elHudChestBpm = document.getElementById("hud-chest-bpm");
  if (elHudChestFps) elHudChestFps.textContent = fps.toFixed(1);
  if (elHudChestBpm) elHudChestBpm.textContent = chest > 0 ? chest.toFixed(1) : "--";

  // Unified 1-Cam elements
  const elUnifiedHR = document.getElementById("unified-val-hr");
  const elUnifiedRR = document.getElementById("unified-val-rr");
  const elUnifiedChest = document.getElementById("unified-val-chest");
  const elHudUnifiedFps = document.getElementById("hud-unified-fps");
  if (elUnifiedHR) elUnifiedHR.textContent = hr > 0 ? hr.toFixed(1) : "--";
  if (elUnifiedRR) elUnifiedRR.textContent = rr > 0 ? rr.toFixed(1) : "--";
  if (elUnifiedChest) elUnifiedChest.textContent = chest > 0 ? chest.toFixed(1) : "--";
  if (elHudUnifiedFps) elHudUnifiedFps.textContent = fps.toFixed(1);

  if (elSyncTime) {
    elSyncTime.textContent = new Date().toLocaleTimeString("vi-VN", { hour12: false });
  }

  // 2. Adjust Heart Pulse Animation Speed according to actual BPM
  if (heartSvg && hr > 30) {
    const periodSec = 60.0 / hr;
    heartSvg.style.animationDuration = `${periodSec.toFixed(2)}s`;
  }

  // 3. Progress Bars & Ranges
  const hrProgress = document.getElementById("hr-progress-bar");
  if (hrProgress && hr > 0) {
    const hrPct = Math.min(100, Math.max(5, ((hr - 40) / 120) * 100));
    hrProgress.style.width = `${hrPct}%`;
  }

  const rrProgress = document.getElementById("rr-progress-bar");
  if (rrProgress && rr > 0) {
    const rrPct = Math.min(100, Math.max(5, ((rr - 6) / 26) * 100));
    rrProgress.style.width = `${rrPct}%`;
  }

  const chestProgress = document.getElementById("chest-progress-bar");
  if (chestProgress && chest > 0) {
    const chestPct = Math.min(100, Math.max(5, ((chest - 8) / 24) * 100));
    chestProgress.style.width = `${chestPct}%`;
  }

  // 4. Status Pills
  updateStatusPills(hr, rr, chest);

  // 5. Difference between rPPG and Chest
  const diffEl = document.getElementById("chest-diff");
  if (diffEl && hr > 0 && chest > 0) {
    const diff = Math.abs(rr - chest);
    diffEl.innerHTML = `&Delta; ${diff.toFixed(1)}`;
  }
}

function updateStatusPills(hr, rr, chest) {
  const hrPill = document.getElementById("hr-status-pill");
  const rrPill = document.getElementById("rr-status-pill");
  const chestPill = document.getElementById("chest-status-pill");
  const clinPill = document.getElementById("clinical-status-pill");
  const summaryEl = document.getElementById("health-summary");
  const scoreEl = document.getElementById("health-score");

  let isAbnormal = false;
  let clinMsg = "Chỉ số sinh lý ổn định, trong giới hạn bình thường.";
  let score = 96;

  // Evaluate HR
  if (hrPill && hr > 0) {
    if (hr < thresholds.hr_low) {
      hrPill.className = "status-pill pill-warning";
      hrPill.textContent = "Nhịp chậm";
      isAbnormal = true;
      score -= 15;
    } else if (hr > thresholds.hr_high) {
      hrPill.className = "status-pill pill-critical";
      hrPill.textContent = "Nhịp nhanh";
      isAbnormal = true;
      score -= 25;
    } else {
      hrPill.className = "status-pill pill-normal";
      hrPill.textContent = "Bình thường";
    }
  }

  // Evaluate RR
  if (rrPill && rr > 0) {
    if (rr < thresholds.rr_low) {
      rrPill.className = "status-pill pill-warning";
      rrPill.textContent = "Thở chậm";
      isAbnormal = true;
      score -= 15;
    } else if (rr > thresholds.rr_high) {
      rrPill.className = "status-pill pill-critical";
      rrPill.textContent = "Thở nhanh";
      isAbnormal = true;
      score -= 20;
    } else {
      rrPill.className = "status-pill pill-normal";
      rrPill.textContent = "Bình thường";
    }
  }

  // Evaluate Chest
  if (chestPill && chest > 0) {
    if (chest < thresholds.rr_low || chest > thresholds.rr_high) {
      chestPill.className = "status-pill pill-warning";
      chestPill.textContent = "Bất thường";
    } else {
      chestPill.className = "status-pill pill-normal";
      chestPill.textContent = "Đồng bộ";
    }
  }

  // Clinical badge & summary
  if (clinPill) {
    if (isAbnormal) {
      clinPill.className = "status-pill pill-critical";
      clinPill.textContent = "Cần theo dõi";
      clinMsg = "Phát hiện chỉ số ngoài ngưỡng chuẩn. Đang kích hoạt theo dõi sát và gửi cảnh báo.";
      playWarningBeep();
    } else {
      clinPill.className = "status-pill pill-normal";
      clinPill.textContent = "Bình thường";
    }
  }

  if (summaryEl) summaryEl.textContent = clinMsg;
  if (scoreEl) scoreEl.textContent = Math.max(40, score);
}

// ==============================================================================
// 5. ALERT EVENT LOG
// ==============================================================================
let alertEvents = [];

function addAlertEvent(record) {
  const container = document.getElementById("alert-events-list");
  const countBadge = document.getElementById("alert-counter-badge");
  if (!container) return;

  const hr = record.heart_rate;
  const rr = record.respiration_rate || record.chest_bpm || record.bpm;
  const severity = record.status || "normal";
  const timeStr = new Date(record.recorded_at || record.created_at || Date.now()).toLocaleTimeString("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  let message = "Ghi nhận chỉ số sinh hiệu định kỳ.";
  if (severity === "critical") {
    message = `CẢNH BÁO NGUY HIỂM: HR=${hr || "--"} BPM, RR=${rr || "--"} RPM vượt ngưỡng an toàn!`;
  } else if (severity === "warning") {
    message = `Lưu ý: Chỉ số sinh hiệu biến thiên nhẹ (HR=${hr || "--"}, RR=${rr || "--"}).`;
  }

  const alertItem = { timeStr, severity, message };
  alertEvents.unshift(alertItem);
  if (alertEvents.length > 25) alertEvents.pop();

  if (countBadge) countBadge.textContent = `${alertEvents.length} sự kiện`;

  // Render list
  container.innerHTML = alertEvents
    .map(
      (a) => `
    <div class="alert-item ${a.severity}">
      <div class="alert-msg-wrap">
        <span class="status-dot ${a.severity === "critical" ? "disconnected" : ""}"></span>
        <span>${a.message}</span>
      </div>
      <span class="alert-time font-mono">${a.timeStr}</span>
    </div>
  `
    )
    .join("");
}

// ==============================================================================
// 6. DATA RECORDS TABLE & FILTERS
// ==============================================================================
async function loadHistory(metricType = "all") {
  try {
    const res = await fetch(`/api/history?limit=50&type=${metricType}`);
    if (!res.ok) return;
    const records = await res.json();

    renderTable(records);
    initHistoryChart(records);
  } catch (err) {
    console.error("[Dashboard] Lỗi tải lịch sử:", err);
  }
}

function renderTable(records) {
  const tbody = document.getElementById("table-vitals-body");
  if (!tbody) return;

  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-4" style="color: var(--text-muted);">Chưa có dữ liệu lịch sử nào.</td></tr>`;
    return;
  }

  tbody.innerHTML = records
    .map((r) => {
      const dt = new Date(r.recorded_at || r.created_at || Date.now()).toLocaleString("vi-VN");
      const hr = r.heart_rate ? `${parseFloat(r.heart_rate).toFixed(1)}` : "--";
      const rr = r.respiration_rate ? `${parseFloat(r.respiration_rate).toFixed(1)}` : "--";
      const chest = r.chest_bpm || r.bpm ? `${parseFloat(r.chest_bpm || r.bpm).toFixed(1)}` : "--";
      const status = r.status || "normal";
      const pillClass = status === "critical" ? "pill-critical" : status === "warning" ? "pill-warning" : "pill-normal";

      return `
      <tr>
        <td class="font-mono">#${r.id}</td>
        <td class="font-mono">${dt}</td>
        <td><strong class="text-rose font-mono">${hr}</strong> BPM</td>
        <td><strong class="text-emerald font-mono">${rr}</strong> RPM</td>
        <td><strong class="font-mono">${chest}</strong> BPM</td>
        <td><span class="badge-tag">${r.source || "camera_ai"}</span></td>
        <td><span class="status-pill ${pillClass}">${status.toUpperCase()}</span></td>
        <td style="color: var(--text-muted); font-size: 11.5px;">${r.note || ""}</td>
      </tr>
    `;
    })
    .join("");
}

// ==============================================================================
// 7. REALTIME SERVER-SENT EVENTS (SSE) STREAM
// ==============================================================================
function connectSSE() {
  const statusPill = document.getElementById("connection-status-pill");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  const es = new EventSource("/api/stream");

  es.addEventListener("connected", () => {
    isConnected = true;
    if (statusDot) statusDot.className = "status-dot pulse";
    if (statusText) statusText.textContent = "Realtime SSE Online";
    console.log("[SSE] Đã kết nối thành công với Backend Flask.");
    loadHistory();
    loadSettings();
  });

  es.addEventListener("live_vitals", (e) => {
    try {
      const data = JSON.parse(e.data);
      updateHeroVitals(data);
      updateOscilloscopes(
        data.bvp_wave, 
        data.resp_wave, 
        data.heart_rate, 
        data.respiration_rate,
        data.chest_bpm,
        data.chest_motion
      );
    } catch (err) {
      console.warn("[SSE] Lỗi phân tích live_vitals:", err);
    }
  });

  es.addEventListener("update", (e) => {
    try {
      const record = JSON.parse(e.data);
      addAlertEvent(record);
      // Reload history table to show new record
      const typeSelect = document.getElementById("filter-metric-type");
      loadHistory(typeSelect ? typeSelect.value : "all");
    } catch (err) {
      console.warn("[SSE] Lỗi phân tích sự kiện update:", err);
    }
  });

  es.addEventListener("history_cleared", () => {
    alertEvents = [];
    loadHistory();
  });

  es.onerror = () => {
    isConnected = false;
    if (statusDot) statusDot.className = "status-dot disconnected";
    if (statusText) statusText.textContent = "Mất kết nối. Đang thử lại...";
  };
}

// ==============================================================================
// 8. INTERACTIVE CONTROLS & EVENT LISTENERS
// ==============================================================================
function initControls() {
  // 1. Audio Mute / Unmute Toggle
  const btnAudio = document.getElementById("btn-audio-toggle");
  const iconSoundOn = document.getElementById("icon-sound-on");
  const iconSoundOff = document.getElementById("icon-sound-off");

  if (btnAudio) {
    btnAudio.addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      if (iconSoundOn && iconSoundOff) {
        iconSoundOn.classList.toggle("hidden", !soundEnabled);
        iconSoundOff.classList.toggle("hidden", soundEnabled);
      }
      if (soundEnabled) playWarningBeep(1800, "sine", 0.15);
    });
  }

  // 2. Video Mode Selector (Simulation / Demo Video / Webcam)
  const selectMode = document.getElementById("select-rppg-mode");
  if (selectMode) {
    selectMode.addEventListener("change", async (e) => {
      const mode = e.target.value;
      const bannerMode = document.getElementById("banner-rppg-mode");
      if (bannerMode) {
        bannerMode.textContent = mode === "simulation" ? "Mô phỏng động" : mode === "demo_video" ? "Video mẫu" : "Webcam Live";
      }

      await fetch("/api/control/rppg/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
    });
  }

  // 3. Interactive Simulation Sliders
  const sliderHR = document.getElementById("slider-sim-hr");
  const sliderRR = document.getElementById("slider-sim-rr");
  const valSimHR = document.getElementById("val-sim-hr");
  const valSimRR = document.getElementById("val-sim-rr");

  async function syncSimParams() {
    const hr = parseFloat(sliderHR.value);
    const rr = parseFloat(sliderRR.value);
    if (valSimHR) valSimHR.textContent = `${hr} BPM`;
    if (valSimRR) valSimRR.textContent = `${rr} RPM`;

    await fetch("/api/control/rppg/params", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heart_rate: hr, respiration_rate: rr }),
    });
  }

  if (sliderHR) sliderHR.addEventListener("input", syncSimParams);
  if (sliderRR) sliderRR.addEventListener("input", syncSimParams);

  // Preset Buttons
  const presetBtns = document.querySelectorAll(".btn-preset");
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const hr = btn.getAttribute("data-hr");
      const rr = btn.getAttribute("data-rr");
      if (sliderHR) sliderHR.value = hr;
      if (sliderRR) sliderRR.value = rr;
      syncSimParams();
    });
  });

  // 4. Toggle FaceMesh
  const btnMesh = document.getElementById("btn-toggle-facemesh");
  let meshState = false;
  if (btnMesh) {
    btnMesh.addEventListener("click", async () => {
      meshState = !meshState;
      await fetch("/api/control/rppg/mesh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_mesh: meshState }),
      });
      btnMesh.style.borderColor = meshState ? "var(--resp)" : "var(--border)";
    });
  }

  // 5. Toggle ROI Mode
  const btnROI = document.getElementById("btn-roi-mode");
  const lblROI = document.getElementById("btn-roi-label");
  const hudROI = document.getElementById("hud-roi");
  const roiCycle = ["all", "forehead", "cheeks"];
  let roiIdx = 0;

  if (btnROI) {
    btnROI.addEventListener("click", async () => {
      roiIdx = (roiIdx + 1) % roiCycle.length;
      const mode = roiCycle[roiIdx];
      const textMap = {
        all: "Vùng: Cả trán & má",
        forehead: "Vùng: Chỉ trán",
        cheeks: "Vùng: Chỉ 2 má",
      };

      if (lblROI) lblROI.textContent = textMap[mode];
      if (hudROI) hudROI.textContent = textMap[mode].replace("Vùng: ", "");

      await fetch("/api/control/rppg/roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ roi_mode: mode }),
      });
    });
  }

  // 6. Clear History Button
  const btnClear = document.getElementById("btn-clear-history");
  if (btnClear) {
    btnClear.addEventListener("click", async () => {
      if (confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử đo lường không?")) {
        await fetch("/api/history/clear", { method: "POST" });
        loadHistory();
      }
    });
  }

  // 7. Filter Type Select in Records Tab
  const filterType = document.getElementById("filter-metric-type");
  if (filterType) {
    filterType.addEventListener("change", (e) => {
      loadHistory(e.target.value);
    });
  }

  // 8. Snapshot Button rPPG
  const btnSnapshot = document.getElementById("btn-capture-snapshot");
  if (btnSnapshot) {
    btnSnapshot.addEventListener("click", () => {
      window.open("/api/video_feed/rppg", "_blank");
    });
  }

  // 9. Chest Video Controls (Tab 3: Ngực & YOLO)
  const selectChestSource = document.getElementById("select-chest-source");
  if (selectChestSource) {
    selectChestSource.addEventListener("change", async (e) => {
      const source = e.target.value;
      const espUrlInput = document.getElementById("input-esp32-url");
      const esp32_url = espUrlInput ? espUrlInput.value.trim() : null;
      await fetch("/api/control/chest/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, esp32_url }),
      });
    });
  }

  const btnChestSkeleton = document.getElementById("btn-toggle-chest-skeleton");
  let chestSkeletonState = true;
  if (btnChestSkeleton) {
    btnChestSkeleton.addEventListener("click", async () => {
      chestSkeletonState = !chestSkeletonState;
      await fetch("/api/control/chest/skeleton", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_skeleton: chestSkeletonState }),
      });
      btnChestSkeleton.style.borderColor = chestSkeletonState ? "var(--motion)" : "var(--border)";
    });
  }

  const btnChestROI = document.getElementById("btn-toggle-chest-roi");
  let chestROIState = true;
  if (btnChestROI) {
    btnChestROI.addEventListener("click", async () => {
      chestROIState = !chestROIState;
      await fetch("/api/control/chest/roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_roi: chestROIState }),
      });
      btnChestROI.style.borderColor = chestROIState ? "var(--motion)" : "var(--border)";
    });
  }

  const btnCaptureChest = document.getElementById("btn-capture-chest-snapshot");
  if (btnCaptureChest) {
    btnCaptureChest.addEventListener("click", () => {
      window.open("/api/video_feed/chest", "_blank");
    });
  }

  // 10. ESP32-CAM Test Button
  const btnTestESP32 = document.getElementById("btn-test-esp32");
  const inputESP32 = document.getElementById("input-esp32-url");
  if (btnTestESP32 && inputESP32) {
    btnTestESP32.addEventListener("click", async () => {
      const url = inputESP32.value.trim();
      if (!url) return alert("Vui lòng nhập URL hợp lệ của ESP32-CAM");
      // Switch chest source to esp32 and test
      await fetch("/api/control/chest/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "esp32", esp32_url: url }),
      });
      if (selectChestSource) selectChestSource.value = "esp32";
    });
  }

  // 10. Test Audio Beep Button in Settings
  const btnTestSound = document.getElementById("btn-test-sound");
  if (btnTestSound) {
    btnTestSound.addEventListener("click", () => {
      playWarningBeep(2600, "sine", 0.35);
    });
  }

  // 11. Settings Form Submission
  const formSettings = document.getElementById("form-thresholds");
  const saveToast = document.getElementById("save-toast");

  if (formSettings) {
    formSettings.addEventListener("submit", async (e) => {
      e.preventDefault();
      const hr_low = parseFloat(document.getElementById("set-hr-low").value);
      const hr_high = parseFloat(document.getElementById("set-hr-high").value);
      const rr_low = parseFloat(document.getElementById("set-rr-low").value);
      const rr_high = parseFloat(document.getElementById("set-rr-high").value);
      const sound_alert = document.getElementById("set-sound-alert").checked;
      const email_alert = document.getElementById("set-email-alert").checked;

      thresholds = { hr_low, hr_high, rr_low, rr_high };

      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hr_low, hr_high, rr_low, rr_high, sound_alert, email_alert }),
      });

      if (res.ok && saveToast) {
        saveToast.classList.remove("hidden");
        setTimeout(() => saveToast.classList.add("hidden"), 3000);
      }
    });
  }
}

// ==============================================================================
// 9. LOAD SETTINGS FROM SERVER
// ==============================================================================
async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) return;
    const data = await res.json();

    thresholds = {
      hr_low: data.hr_low || 50,
      hr_high: data.hr_high || 100,
      rr_low: data.rr_low || 10,
      rr_high: data.rr_high || 24,
    };

    if (document.getElementById("set-hr-low")) document.getElementById("set-hr-low").value = thresholds.hr_low;
    if (document.getElementById("set-hr-high")) document.getElementById("set-hr-high").value = thresholds.hr_high;
    if (document.getElementById("set-rr-low")) document.getElementById("set-rr-low").value = thresholds.rr_low;
    if (document.getElementById("set-rr-high")) document.getElementById("set-rr-high").value = thresholds.rr_high;
    if (document.getElementById("set-sound-alert")) document.getElementById("set-sound-alert").checked = !!data.sound_alert;
    if (document.getElementById("set-email-alert")) document.getElementById("set-email-alert").checked = !!data.email_alert;
  } catch (err) {
    console.warn("[Settings] Không thể tải cài đặt từ server:", err);
  }
}

// ==============================================================================
// INITIALIZATION ON DOM READY
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initTabs();
  initOscilloscopes();
  initControls();
  connectSSE();
});
