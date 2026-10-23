/**
 * Diffusion Model for Image Denoising
 * Frontend Controller Script
 * Handles uploads, sample selection, forward noise injection,
 * and PyTorch reverse diffusion with trajectory visualization.
 */

// Global Application State
const appState = {
  currentCleanFilename: null,
  currentNoisyFilename: null,
  currentTimestep: 150,
  totalTimesteps: 300,
  betaStart: 0.0001,
  betaEnd: 0.02
};

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadStatusText = document.getElementById("uploadStatusText");
const currentFilenameBadge = document.getElementById("currentFilenameBadge");
const sampleDigitsContainer = document.getElementById("sampleDigitsContainer");

const timestepSlider = document.getElementById("timestepSlider");
const timestepDisplay = document.getElementById("timestepDisplay");
const signalWeightVal = document.getElementById("signalWeightVal");
const noiseWeightVal = document.getElementById("noiseWeightVal");
const alphaBarVal = document.getElementById("alphaBarVal");

const btnForwardDiffuse = document.getElementById("btnForwardDiffuse");
const btnReverseDiffuse = document.getElementById("btnReverseDiffuse");
const reverseProgressContainer = document.getElementById("reverseProgressContainer");
const reverseProgressStatus = document.getElementById("reverseProgressStatus");

const cleanImg = document.getElementById("cleanImg");
const cleanPlaceholder = document.getElementById("cleanPlaceholder");
const noisyImg = document.getElementById("noisyImg");
const noisyPlaceholder = document.getElementById("noisyPlaceholder");
const denoisedImg = document.getElementById("denoisedImg");
const denoisedPlaceholder = document.getElementById("denoisedPlaceholder");
const btnDownload = document.getElementById("btnDownload");

const filmstripContainer = document.getElementById("filmstripContainer");
const metricTime = document.getElementById("metricTime");
const metricMSE = document.getElementById("metricMSE");
const metricPSNR = document.getElementById("metricPSNR");
const metricCorr = document.getElementById("metricCorr");

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupFileUpload();
  loadSampleDigits();
  setupSlider();
  setupActionButtons();
  updateMathDisplay(appState.currentTimestep);
});

// ============================================================================
// 1. Pre-loaded Sample Digit Selector
// ============================================================================
async function loadSampleDigits() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();

    if (!samples || samples.length === 0) {
      sampleDigitsContainer.innerHTML = `<span class="text-muted small">No samples found.</span>`;
      return;
    }

    sampleDigitsContainer.innerHTML = "";
    samples.forEach((sample, idx) => {
      const tile = document.createElement("div");
      tile.className = "sample-tile";
      tile.title = `Select ${sample.label}`;
      tile.dataset.filename = sample.filename;

      const img = document.createElement("img");
      img.src = sample.url;
      img.alt = sample.label;

      tile.appendChild(img);
      tile.addEventListener("click", () => selectSampleDigit(tile, sample.filename));

      sampleDigitsContainer.appendChild(tile);

      // Auto-select digit 3 on first load for immediate out-of-the-box readiness!
      if (idx === 3) {
        setTimeout(() => selectSampleDigit(tile, sample.filename), 200);
      }
    });
  } catch (err) {
    console.error("Error loading sample digits:", err);
    sampleDigitsContainer.innerHTML = `<span class="text-danger small">Failed to load digit gallery.</span>`;
  }
}

async function selectSampleDigit(tileElement, filename) {
  // Update tile active state
  document.querySelectorAll(".sample-tile").forEach(t => t.classList.remove("selected"));
  tileElement.classList.add("selected");

  uploadStatusText.innerHTML = `<i class="fa-solid fa-spinner fa-spin me-1 text-primary"></i> Loading sample digit...`;

  const formData = new FormData();
  formData.append("sample_file", filename);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.success) {
      handleUploadSuccess(data, filename);
    } else {
      showError(data.error || "Failed to load sample digit.");
    }
  } catch (err) {
    showError("Network error while loading sample digit.");
  }
}

// ============================================================================
// 2. Drag & Drop and File Upload Handling
// ============================================================================
function setupFileUpload() {
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  });
}

async function uploadFile(file) {
  // Validate file type
  const validTypes = ["image/png", "image/jpeg", "image/jpg"];
  if (!validTypes.includes(file.type)) {
    showError("Invalid file type! Please upload a PNG, JPG, or JPEG image.");
    return;
  }

  // Validate size (max 5MB)
  if (file.size > 5 * 1024 * 1024) {
    showError("File size exceeds 5MB limit. Please upload a smaller image.");
    return;
  }

  // Clear sample tile selection
  document.querySelectorAll(".sample-tile").forEach(t => t.classList.remove("selected"));

  uploadStatusText.innerHTML = `<i class="fa-solid fa-spinner fa-spin me-1 text-primary"></i> Uploading & resizing to 28x28...`;

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.success) {
      handleUploadSuccess(data, file.name);
    } else {
      showError(data.error || "Upload failed.");
    }
  } catch (err) {
    showError("Upload request failed. Check server connection.");
  }
}

function handleUploadSuccess(data, label) {
  appState.currentCleanFilename = data.clean_filename;
  appState.currentNoisyFilename = null;

  // Display original image
  cleanImg.src = data.preview_url;
  cleanImg.classList.remove("d-none");
  cleanPlaceholder.classList.add("d-none");

  // Reset noisy & denoised boxes
  noisyImg.classList.add("d-none");
  noisyPlaceholder.classList.remove("d-none");
  denoisedImg.classList.add("d-none");
  denoisedPlaceholder.classList.remove("d-none");
  btnDownload.classList.add("d-none");

  // Clear metrics & filmstrip
  resetMetrics();
  filmstripContainer.innerHTML = `<div class="text-muted small py-3 px-2">Run reverse diffusion to view intermediate trajectory frames.</div>`;

  // Enable Forward Diffusion button
  btnForwardDiffuse.disabled = false;
  btnReverseDiffuse.disabled = true;

  // Feedback label
  uploadStatusText.innerHTML = `<span class="text-success"><i class="fa-solid fa-circle-check me-1"></i> Ready (${data.resolution})</span>`;
  currentFilenameBadge.textContent = label;
  currentFilenameBadge.classList.remove("d-none");
}

// ============================================================================
// 3. Timestep Slider & Mathematical Display
// ============================================================================
function setupSlider() {
  timestepSlider.addEventListener("input", (e) => {
    const t = parseInt(e.target.value, 10);
    appState.currentTimestep = t;
    timestepDisplay.textContent = `t = ${t}`;
    updateMathDisplay(t);
  });
}

function setTimestep(val) {
  timestepSlider.value = val;
  appState.currentTimestep = val;
  timestepDisplay.textContent = `t = ${val}`;
  updateMathDisplay(val);

  // If clean image already loaded, trigger forward diffuse preview
  if (appState.currentCleanFilename) {
    runForwardDiffusion();
  }
}

function updateMathDisplay(t) {
  const T = appState.totalTimesteps;
  // Closed-form approximate linear beta schedule
  const beta_1 = appState.betaStart;
  const beta_T = appState.betaEnd;
  const beta_t = beta_1 + (t / (T - 1)) * (beta_T - beta_1);

  // Approximate alpha_bar for visual feedback (exact value returned by server)
  const alpha_bar_approx = Math.exp(-0.5 * (beta_1 * t + 0.5 * (beta_T - beta_1) * (t * t) / T));
  const clamped_alpha_bar = Math.max(0.0001, Math.min(0.9999, alpha_bar_approx));
  const signal_weight = Math.sqrt(clamped_alpha_bar);
  const noise_weight = Math.sqrt(1.0 - clamped_alpha_bar);

  signalWeightVal.textContent = signal_weight.toFixed(4);
  noiseWeightVal.textContent = noise_weight.toFixed(4);
  alphaBarVal.textContent = clamped_alpha_bar.toFixed(4);
}

// ============================================================================
// 4. Forward Diffusion (Add Gaussian Noise)
// ============================================================================
function setupActionButtons() {
  btnForwardDiffuse.addEventListener("click", runForwardDiffusion);
  btnReverseDiffuse.addEventListener("click", runReverseDiffusion);
}

async function runForwardDiffusion() {
  if (!appState.currentCleanFilename) {
    showError("Please select or upload an image first.");
    return;
  }

  btnForwardDiffuse.disabled = true;
  btnForwardDiffuse.innerHTML = `<i class="fa-solid fa-spinner fa-spin me-2"></i>Injecting Noise...`;

  try {
    const res = await fetch("/api/forward-diffuse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clean_filename: appState.currentCleanFilename,
        timestep: appState.currentTimestep
      })
    });
    const data = await res.json();

    if (data.success) {
      appState.currentNoisyFilename = data.noisy_filename;

      // Update noisy image box
      noisyImg.src = data.noisy_preview_url;
      noisyImg.classList.remove("d-none");
      noisyPlaceholder.classList.add("d-none");

      // Update exact math parameters from backend
      if (data.math_info) {
        signalWeightVal.textContent = data.math_info.signal_weight;
        noiseWeightVal.textContent = data.math_info.noise_weight;
        alphaBarVal.textContent = data.math_info.alpha_bar;
      }

      // Enable reverse diffusion
      btnReverseDiffuse.disabled = false;
      // Reset denoised view until user clicks reverse
      denoisedImg.classList.add("d-none");
      denoisedPlaceholder.classList.remove("d-none");
      btnDownload.classList.add("d-none");
      resetMetrics();
    } else {
      showError(data.error || "Forward diffusion failed.");
    }
  } catch (err) {
    showError("Forward diffusion request failed.");
  } finally {
    btnForwardDiffuse.disabled = false;
    btnForwardDiffuse.innerHTML = `<i class="fa-solid fa-plus-minus me-2"></i>Inject Gaussian Noise (x_t)`;
  }
}

// ============================================================================
// 5. Reverse Diffusion (AI Iterative Denoising)
// ============================================================================
async function runReverseDiffusion() {
  if (!appState.currentCleanFilename || !appState.currentNoisyFilename) {
    showError("Please complete forward diffusion first.");
    return;
  }

  btnReverseDiffuse.disabled = true;
  btnForwardDiffuse.disabled = true;
  reverseProgressContainer.classList.remove("d-none");

  // Animate dynamic status messages
  const t_start = appState.currentTimestep;
  let simulatedStep = t_start;
  const statusTimer = setInterval(() => {
    if (simulatedStep > 10) {
      simulatedStep = Math.max(0, simulatedStep - Math.ceil(t_start / 8));
      reverseProgressStatus.textContent = `Reverse sampling: Timestep t = ${simulatedStep} → 0`;
    }
  }, 180);

  try {
    const res = await fetch("/api/reverse-diffuse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clean_filename: appState.currentCleanFilename,
        noisy_filename: appState.currentNoisyFilename,
        timestep: appState.currentTimestep
      })
    });
    const data = await res.json();
    clearInterval(statusTimer);

    if (data.success) {
      // Display clean recovered image
      denoisedImg.src = data.denoised_preview_url;
      denoisedImg.classList.remove("d-none");
      denoisedPlaceholder.classList.add("d-none");

      // Set download link
      btnDownload.href = `/download/outputs/${data.denoised_filename}`;
      btnDownload.classList.remove("d-none");

      // Update metrics
      if (data.metrics) {
        metricTime.textContent = `${data.metrics.inference_time_ms} ms`;
        metricMSE.textContent = data.metrics.mse;
        metricPSNR.textContent = `${data.metrics.psnr_db} dB`;
        metricCorr.textContent = data.metrics.correlation;
      }

      // Populate snapshot trajectory filmstrip
      if (data.snapshots && data.snapshots.length > 0) {
        renderTrajectory(data.snapshots);
      }

    } else {
      showError(data.error || "Reverse diffusion failed.");
    }
  } catch (err) {
    clearInterval(statusTimer);
    showError("Reverse diffusion failed to complete.");
  } finally {
    reverseProgressContainer.classList.add("d-none");
    btnReverseDiffuse.disabled = false;
    btnForwardDiffuse.disabled = false;
  }
}

function renderTrajectory(snapshots) {
  filmstripContainer.innerHTML = "";
  snapshots.forEach((snap) => {
    const item = document.createElement("div");
    item.className = "filmstrip-item";

    const img = document.createElement("img");
    img.src = snap.preview_url;
    img.alt = `Step ${snap.step}`;

    const label = document.createElement("div");
    label.className = "filmstrip-label";
    label.textContent = snap.step === 0 ? "Final (t=0)" : `Step t=${snap.step}`;

    item.appendChild(img);
    item.appendChild(label);
    filmstripContainer.appendChild(item);
  });
}

function resetMetrics() {
  metricTime.textContent = "--";
  metricMSE.textContent = "--";
  metricPSNR.textContent = "--";
  metricCorr.textContent = "--";
}

function showError(msg) {
  alert(msg);
}
