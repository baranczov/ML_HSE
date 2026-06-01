// Tab navigation
document.querySelectorAll("nav button[data-tab]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button[data-tab]").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ── Utility ──────────────────────────────────────────────────────────────────

function objectUrl(file) {
  return URL.createObjectURL(file);
}

function showError(el, msg) {
  el.textContent = msg;
  el.classList.add("show");
}
function clearError(el) {
  el.textContent = "";
  el.classList.remove("show");
}

// ── Single predict ────────────────────────────────────────────────────────────

(function initSingle() {
  const dropZone   = document.getElementById("single-drop");
  const fileInput  = document.getElementById("single-file");
  const preview    = document.getElementById("single-preview");
  const previewImg = document.getElementById("single-preview-img");
  const clearBtn   = document.getElementById("single-clear");
  const submitBtn  = document.getElementById("single-submit");
  const resultBox  = document.getElementById("single-result");
  const errorBox   = document.getElementById("single-error");
  const ageNum     = document.getElementById("res-age");
  const ageBin     = document.getElementById("res-bin");
  const ageMeta    = document.getElementById("res-meta");
  const cacheBadge = document.getElementById("res-cache");

  let currentFile = null;

  function setFile(file) {
    currentFile = file;
    previewImg.src = objectUrl(file);
    preview.style.display = "flex";
    dropZone.style.display = "none";
    submitBtn.disabled = false;
    resultBox.classList.remove("show");
    clearError(errorBox);
  }

  function resetSingle() {
    currentFile = null;
    preview.style.display = "none";
    dropZone.style.display = "block";
    submitBtn.disabled = true;
    resultBox.classList.remove("show");
    clearError(errorBox);
    fileInput.value = "";
  }

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("over");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  clearBtn.addEventListener("click", resetSingle);

  submitBtn.addEventListener("click", async () => {
    if (!currentFile) return;
    clearError(errorBox);
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Predicting…';

    const form = new FormData();
    form.append("file", currentFile);

    try {
      const res = await fetch("/v1/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || res.statusText);

      ageNum.textContent = Math.round(data.age);
      ageBin.textContent = data.age_bin;
      ageMeta.textContent = `v${data.model_version}  ·  age ${data.age.toFixed(1)}`;
      cacheBadge.textContent = data.cached ? "Cache HIT" : "Cache MISS";
      cacheBadge.className = "cache-badge " + (data.cached ? "hit" : "miss");
      resultBox.classList.add("show");
    } catch (err) {
      showError(errorBox, "Error: " + err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = "Predict age";
    }
  });
})();

// ── Batch predict ─────────────────────────────────────────────────────────────

(function initBatch() {
  const dropZone  = document.getElementById("batch-drop");
  const fileInput = document.getElementById("batch-file");
  const wrap      = document.getElementById("batch-preview-wrap");
  const clearBtn  = document.getElementById("batch-clear");
  const submitBtn = document.getElementById("batch-submit");
  const gridEl    = document.getElementById("batch-grid");
  const errorBox  = document.getElementById("batch-error");

  let files = [];

  function addFiles(newFiles) {
    const MAX = 8;
    for (const f of newFiles) {
      if (files.length >= MAX) break;
      files.push(f);
      const item = document.createElement("div");
      item.className = "preview-item";
      item.innerHTML = `<img src="${objectUrl(f)}" alt=""><button class="remove-btn" title="Remove">×</button>`;
      item.querySelector(".remove-btn").addEventListener("click", () => {
        const idx = Array.from(wrap.children).indexOf(item);
        files.splice(idx, 1);
        item.remove();
        if (!files.length) submitBtn.disabled = true;
      });
      wrap.appendChild(item);
    }
    submitBtn.disabled = files.length === 0;
    clearError(errorBox);
  }

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { if (fileInput.files.length) addFiles(fileInput.files); });
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("over"));
  dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("over"); addFiles(e.dataTransfer.files); });

  clearBtn.addEventListener("click", () => {
    files = [];
    wrap.innerHTML = "";
    submitBtn.disabled = true;
    gridEl.innerHTML = "";
    clearError(errorBox);
  });

  submitBtn.addEventListener("click", async () => {
    if (!files.length) return;
    clearError(errorBox);
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Predicting…';
    gridEl.innerHTML = "";

    const form = new FormData();
    files.forEach(f => form.append("files", f));

    try {
      const res = await fetch("/v1/predict_batch", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || res.statusText);

      data.results.forEach((r, i) => {
        const card = document.createElement("div");
        card.className = "batch-card";
        card.innerHTML = `
          <img src="${objectUrl(files[i])}" alt="">
          <div class="b-age">${Math.round(r.age)}</div>
          <div class="b-bin">${r.age_bin}</div>
          <div class="age-meta" style="margin-top:4px">
            <span class="cache-badge ${r.cached ? 'hit' : 'miss'}">${r.cached ? 'HIT' : 'MISS'}</span>
          </div>`;
        gridEl.appendChild(card);
      });
    } catch (err) {
      showError(errorBox, "Error: " + err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = "Predict all";
    }
  });
})();

// ── System status ─────────────────────────────────────────────────────────────

(function initStatus() {
  const panel = document.getElementById("tab-status");
  const tbody = document.getElementById("meta-body");

  async function loadMeta() {
    tbody.innerHTML = `<tr><td colspan="2" style="text-align:center"><span class="spinner"></span></td></tr>`;
    try {
      const [metaRes, readyRes] = await Promise.all([
        fetch("/v1/meta"),
        fetch("/ready"),
      ]);
      const meta  = await metaRes.json();
      const ready = await readyRes.json();

      const rows = [
        ["Status", ready.ready ? '<span class="status-chip ok">● Ready</span>' : '<span class="status-chip err">● Not ready</span>'],
        ["Backbone", meta.backbone],
        ["Model version", meta.model_version],
        ["Age range", meta.age_range.join(" – ")],
        ["Image size", `${meta.image_size} × ${meta.image_size}`],
        ["Max upload", `${(meta.max_upload_bytes / 1024 / 1024).toFixed(0)} MB`],
        ["Max batch", meta.max_batch_size],
        ["Cache size", meta.cache_size.toLocaleString()],
        ["— prediction hits", meta.cache_stats.prediction_hits],
        ["— prediction misses", meta.cache_stats.prediction_misses],
        ["— prediction hit rate", `${(meta.cache_stats.prediction_hit_rate * 100).toFixed(1)}%`],
        ["— embedding hits", meta.cache_stats.embedding_hits],
        ["— embedding hit rate", `${(meta.cache_stats.embedding_hit_rate * 100).toFixed(1)}%`],
      ];
      tbody.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="2" style="color:#c00">Failed to load: ${e.message}</td></tr>`;
    }
  }

  // Load when tab becomes visible
  document.querySelectorAll("nav button[data-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.dataset.tab === "tab-status") loadMeta();
    });
  });
  document.getElementById("refresh-btn").addEventListener("click", loadMeta);
})();
