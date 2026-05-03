/**
 * SkinAI — Client-Side Logic
 * Cloud-Enabled AI System for Skin Cancer Detection
 * Drag-and-drop upload, image preview, API calls, animated results rendering.
 */

(function () {
  'use strict';

  // DOM Elements
  const uploadZone = document.getElementById('uploadZone');
  const uploadBtn = document.getElementById('uploadBtn');
  const fileInput = document.getElementById('fileInput');
  const previewArea = document.getElementById('previewArea');
  const previewImg = document.getElementById('previewImg');
  const previewName = document.getElementById('previewName');
  const previewSize = document.getElementById('previewSize');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const resultsSection = document.getElementById('resultsSection');
  const uploadSection = document.getElementById('uploadSection');
  const newScanBtn = document.getElementById('newScanBtn');
  const toast = document.getElementById('toast');

  // Result elements
  const ringFill = document.getElementById('ringFill');
  const confidencePct = document.getElementById('confidencePct');
  const resultDiagnosis = document.getElementById('resultDiagnosis');
  const resultFullname = document.getElementById('resultFullname');
  const riskBadge = document.getElementById('riskBadge');
  const infoDescription = document.getElementById('infoDescription');
  const infoRecommendation = document.getElementById('infoRecommendation');
  const probChart = document.getElementById('probChart');

  let selectedFile = null;

  // ============================================================
  // File Handling
  // ============================================================

  function handleFile(file) {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      showToast('Invalid file type. Please upload JPG, PNG, or WebP.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      showToast('File too large. Maximum size is 10 MB.');
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatFileSize(file.size);
      previewArea.classList.add('active');
      uploadZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewArea.classList.remove('active');
    previewImg.src = '';
    uploadZone.style.display = '';
  }

  // ============================================================
  // Drag and Drop
  // ============================================================

  uploadZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', function (e) {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // Click to browse
  uploadZone.addEventListener('click', function () {
    fileInput.click();
  });

  uploadBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // Keyboard accessibility
  uploadZone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  // ============================================================
  // Analyze Button
  // ============================================================

  analyzeBtn.addEventListener('click', async function () {
    if (!selectedFile) {
      showToast('No image selected.');
      return;
    }

    // Show loading
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⏳ Processing...';
    loadingOverlay.classList.add('active');
    previewArea.classList.remove('active');

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      renderResults(data);

    } catch (err) {
      showToast(err.message || 'Something went wrong. Please try again.');
      clearSelection();
    } finally {
      loadingOverlay.classList.remove('active');
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = '🔍 Analyze';
    }
  });

  // ============================================================
  // Clear & New Scan
  // ============================================================

  clearBtn.addEventListener('click', clearSelection);

  newScanBtn.addEventListener('click', function () {
    resultsSection.classList.remove('active');
    uploadSection.style.display = '';
    clearSelection();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // ============================================================
  // Render Results
  // ============================================================

  function renderResults(data) {
    const pred = data.prediction;
    const probs = data.all_probabilities;

    // Hide upload, show results
    uploadSection.style.display = 'none';
    resultsSection.classList.add('active');

    // Confidence ring
    const circumference = 2 * Math.PI * 60; // r=60
    const offset = circumference - (pred.confidence / 100) * circumference;

    ringFill.classList.remove('high-risk', 'medium-risk', 'low-risk');
    if (pred.risk === 'high') ringFill.classList.add('high-risk');
    else if (pred.risk === 'medium') ringFill.classList.add('medium-risk');
    else ringFill.classList.add('low-risk');

    // Animate ring after a tiny delay
    requestAnimationFrame(function () {
      ringFill.style.strokeDashoffset = circumference;
      requestAnimationFrame(function () {
        ringFill.style.strokeDashoffset = offset;
      });
    });

    // Animated counter
    animateCounter(confidencePct, 0, pred.confidence, 1200);

    // Text
    resultDiagnosis.textContent = pred.name;
    resultFullname.textContent = pred.full_name;

    // Risk badge
    riskBadge.className = 'risk-badge ' + pred.risk;
    const riskIcons = { high: '🔴', medium: '🟡', low: '🟢' };
    riskBadge.textContent = (riskIcons[pred.risk] || '') + ' ' + pred.risk.toUpperCase() + ' RISK';

    // Grad-CAM images
    if (data.images) {
      document.getElementById('imgOriginal').src = data.images.original;
      document.getElementById('imgGradcam').src = data.images.gradcam_overlay;
      document.getElementById('imgHeatmap').src = data.images.heatmap;
      document.getElementById('gradcamSection').style.display = 'block';
    } else {
      document.getElementById('gradcamSection').style.display = 'none';
    }

    // Info cards
    infoDescription.textContent = pred.description;
    infoRecommendation.textContent = pred.recommendation;

    // Probability bars
    probChart.innerHTML = '';
    probs.forEach(function (item, index) {
      const row = document.createElement('div');
      row.className = 'prob-bar-item';
      row.style.animationDelay = (index * 0.08) + 's';

      const isTop = index === 0;

      row.innerHTML =
        '<span class="prob-label">' + escapeHtml(item.name) + '</span>' +
        '<div class="prob-track"><div class="prob-fill' + (isTop ? ' is-top' : '') + '" id="probFill' + index + '"></div></div>' +
        '<span class="prob-value">' + item.probability.toFixed(1) + '%</span>';

      probChart.appendChild(row);

      // Animate bar width after DOM insertion
      setTimeout(function () {
        var fill = document.getElementById('probFill' + index);
        if (fill) fill.style.width = Math.max(item.probability, 0.5) + '%';
      }, 100 + index * 100);
    });

    // Scroll to results
    setTimeout(function () {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
  }

  // ============================================================
  // Utilities
  // ============================================================

  function animateCounter(element, start, end, duration) {
    var startTime = null;
    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      var progress = Math.min((timestamp - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      var current = start + (end - start) * eased;
      element.textContent = current.toFixed(1) + '%';
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(function () {
      toast.classList.remove('show');
    }, 4000);
  }

})();
