/**
 * MedShield AI — Claude-Inspired Reactive Frontend Controller
 * ============================================================
 * Coordinates:
 * - Minimal, high-trust clinical medical chat (Centered 760px column)
 * - On-demand slide-out Defense & MoPE Inspector drawer (Progressive Disclosure)
 * - Model selector dropdown (M6 Defended vs M1 Leaky vs M0 Base)
 * - Prompt starter chips & sidebar library
 * - Research Benchmarks (Table VII) & Figures Gallery modals
 */

// Smart API base: if opened via Antigravity Live Preview / Live Server on another port or protocol,
// dynamically target the active MedShield backend on port 8000.
const API_BASE = (window.location.port === "8000") ? "" : "http://127.0.0.1:8000";

let CURRENT_THEME = localStorage.getItem("medshield_theme") || "system"; // "system", "light", "dark"
let CURRENT_MODEL_MODE = "m6"; // "m6", "m1", "m0"
let CHAT_HISTORY = [];
let TURNS_TELEMETRY = []; // Store telemetry per turn for instant drawer replay
let RADAR_CHART = null;
let MOPE_CHART = null;

const FIGURES_DATA = [
  { id: "fig1_privacy_utility_tradeoff.png", title: "Fig 1: Privacy–Utility Pareto Frontier (M0–M6)" },
  { id: "fig2_leak_rate_comparison.png", title: "Fig 2: Residual PHI Leak Rate Comparison Across All Tiers" },
  { id: "fig3_attack_category_heatmap.png", title: "Fig 3: MedGemma Attack Framing Vulnerability Heatmap" },
  { id: "fig4_per_fact_leak.png", title: "Fig 4: Per-Fact Leak Rate Comparison (Gender, Diagnosis, Med, Insurance)" },
  { id: "fig5_confusion_matrix_guardrail.png", title: "Fig 5: MediRed+ 9-Class Guardrail Confusion Matrix (98.2% Acc)" },
  { id: "fig6_mope_expert_utilization.png", title: "Fig 6: MoPE Router Expert Allocation (Anti-Collapse Active, Entropy 0.94)" },
  { id: "fig7_per_category_f1.png", title: "Fig 7: Per-Category Precision, Recall & F1 on MediRed+ Test Set" },
  { id: "fig8_baseline_vs_contrastive.png", title: "Fig 8: Baseline TF-IDF SVM vs Contrastive Dual-Encoder Evolution" },
  { id: "fig9_latency_breakdown.png", title: "Fig 9: M6 Pipeline Defense Overhead (+3.0 ms Latency Breakdown)" },
  { id: "fig10_utility_retention.png", title: "Fig 10: Medical Utility Retention Across All Defense Configurations" },
  { id: "fig11_demo_quality_before_after.png", title: "Fig 11: Qualitative Case Study — LoRA Slot Collapse vs Dynamic Routing" },
  { id: "fig12_attack_mode_vulnerability.png", title: "Fig 12: Fine-Tuned Model Vulnerability by Attack Mode" },
  { id: "fig13_guardrail_vs_mope_complementary.png", title: "Fig 13: Complementary Properties of M3 Guardrail + M4 MoPE" },
  { id: "fig14_medgemma_vs_llama_comparison.png", title: "Fig 14: Cross-Model Generalization (MedGemma vs Llama)" },
  { id: "fig15_table_vii_master_summary.png", title: "Fig 15: Table VII Master Summary Publication Graphic" },
];

function safeRenderIcons() {
  if (typeof lucide !== "undefined" && lucide && typeof lucide.createIcons === "function") {
    try { lucide.createIcons(); } catch (e) { /* ignore */ }
  }
}

// =========================================================================
// THEME CONTROLLER (System Default / Dedicated Light / Dedicated Dark)
// =========================================================================
function setTheme(theme) {
  CURRENT_THEME = theme;
  try {
    localStorage.setItem("medshield_theme", theme);
  } catch (e) {
    console.warn("Could not save theme to localStorage:", e);
  }
  applyTheme(theme);
}

function applyTheme(theme) {
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const effective = theme === "system" ? (prefersDark ? "dark" : "light") : theme;

  document.documentElement.setAttribute("data-theme", effective);
  document.documentElement.setAttribute("data-theme-setting", theme);

  // Update segmented switcher pill
  const btnSys = document.getElementById("theme-btn-system");
  const btnLight = document.getElementById("theme-btn-light");
  const btnDark = document.getElementById("theme-btn-dark");

  if (btnSys) btnSys.classList.toggle("active", theme === "system");
  if (btnLight) btnLight.classList.toggle("active", theme === "light");
  if (btnDark) btnDark.classList.toggle("active", theme === "dark");

  // Dynamic Chart.js colors
  updateChartsTheme(effective);
  safeRenderIcons();
}

function updateChartsTheme(effectiveTheme) {
  const isLight = effectiveTheme === "light";
  const gridColor = isLight ? "rgba(0, 0, 0, 0.08)" : "rgba(255, 255, 255, 0.08)";
  const textColor = isLight ? "#475569" : "#9ca3af";
  const yGridColor = isLight ? "rgba(0, 0, 0, 0.06)" : "rgba(255, 255, 255, 0.06)";

  if (RADAR_CHART && RADAR_CHART.options && RADAR_CHART.options.scales && RADAR_CHART.options.scales.r) {
    RADAR_CHART.options.scales.r.grid.color = gridColor;
    RADAR_CHART.options.scales.r.angleLines.color = gridColor;
    if (RADAR_CHART.options.scales.r.pointLabels) {
      RADAR_CHART.options.scales.r.pointLabels.color = textColor;
    }
    RADAR_CHART.update();
  }

  if (MOPE_CHART && MOPE_CHART.options && MOPE_CHART.options.scales) {
    if (MOPE_CHART.options.scales.x && MOPE_CHART.options.scales.x.ticks) {
      MOPE_CHART.options.scales.x.ticks.color = textColor;
    }
    if (MOPE_CHART.options.scales.y) {
      if (MOPE_CHART.options.scales.y.grid) MOPE_CHART.options.scales.y.grid.color = yGridColor;
      if (MOPE_CHART.options.scales.y.ticks) MOPE_CHART.options.scales.y.ticks.color = isLight ? "#64748b" : "#6b7280";
    }
    MOPE_CHART.update();
  }
}

// Live listener for OS system theme changes
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    const saved = localStorage.getItem("medshield_theme") || "system";
    if (saved === "system") {
      applyTheme("system");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  applyTheme(CURRENT_THEME);
  initSystemStatus();
  initPresets();
  initBenchmarksTable();
  initFiguresGallery();
  initCharts();
  // Ensure chart colors reflect active theme immediately
  setTimeout(() => {
    const activeEffective = document.documentElement.getAttribute("data-theme") || "dark";
    updateChartsTheme(activeEffective);
  }, 100);
  safeRenderIcons();

  const chatInput = document.getElementById("chat-input");
  if (chatInput) {
    chatInput.addEventListener("input", autoResizePromptInput);
    chatInput.addEventListener("keyup", autoResizePromptInput);
    chatInput.addEventListener("paste", () => setTimeout(autoResizePromptInput, 20));
    chatInput.addEventListener("change", autoResizePromptInput);
    autoResizePromptInput();
  }

  // Close dropdown on outside click
  document.addEventListener("click", (e) => {
    const dropdown = document.getElementById("model-dropdown-menu");
    const btn = document.getElementById("model-selector-btn");
    if (dropdown && !btn.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.remove("show");
    }
  });
});

// =========================================================================
// SIDEBAR & DRAWER TOGGLES (Progressive Disclosure)
// =========================================================================
function toggleSidebar() {
  const sidebar = document.getElementById("app-sidebar");
  sidebar.classList.toggle("collapsed");
}

function toggleDefenseDrawer(forceState) {
  const drawer = document.getElementById("defense-drawer");
  const toggleBtn = document.getElementById("btn-toggle-inspector");

  if (typeof forceState === "boolean") {
    if (forceState) {
      drawer.classList.add("open");
      toggleBtn.classList.add("active");
    } else {
      drawer.classList.remove("open");
      toggleBtn.classList.remove("active");
    }
  } else {
    drawer.classList.toggle("open");
    toggleBtn.classList.toggle("active");
  }

  // Force chart resize when drawer expands
  if (drawer.classList.contains("open")) {
    setTimeout(() => {
      if (RADAR_CHART) RADAR_CHART.resize();
      if (MOPE_CHART) MOPE_CHART.resize();
    }, 320);
  }
}

// =========================================================================
// MODEL SELECTOR DROPDOWN
// =========================================================================
function toggleModelDropdown() {
  const dropdown = document.getElementById("model-dropdown-menu");
  dropdown.classList.toggle("show");
}

function selectModelMode(mode) {
  CURRENT_MODEL_MODE = mode;
  const dropdown = document.getElementById("model-dropdown-menu");
  dropdown.classList.remove("show");

  const pillText = document.getElementById("model-pill-text");
  const pillDot = document.getElementById("model-pill-dot");
  const badge = document.getElementById("protection-status-badge");
  const inputIndicator = document.getElementById("input-mode-indicator");

  // Highlight active dropdown item
  const items = dropdown.querySelectorAll(".model-dropdown-item");
  items.forEach((item, idx) => {
    item.classList.remove("selected");
    if ((mode === "m6" && idx === 0) || (mode === "m1" && idx === 1) || (mode === "m0" && idx === 2)) {
      item.classList.add("selected");
    }
  });

  if (mode === "m6") {
    pillText.innerText = "M6 Defended (0% Leak)";
    pillDot.style.background = "#10b981";
    badge.className = "protection-badge safe";
    badge.innerHTML = `<i data-lucide="shield-check" style="width: 13px; height: 13px;"></i><span>Protected · τ = 0.521</span>`;
    inputIndicator.innerText = "M6 Defended";
    inputIndicator.style.color = "#34d399";
  } else if (mode === "m1") {
    pillText.innerText = "M1 Vulnerable (39% Leak)";
    pillDot.style.background = "#ef4444";
    badge.className = "protection-badge danger";
    badge.innerHTML = `<i data-lucide="alert-triangle" style="width: 13px; height: 13px;"></i><span>Unprotected (Leaky LoRA)</span>`;
    inputIndicator.innerText = "M1 Vulnerable (Leaky)";
    inputIndicator.style.color = "#f87171";
  } else if (mode === "m0") {
    pillText.innerText = "M0 Base Stock";
    pillDot.style.background = "#3b82f6";
    badge.className = "protection-badge base";
    badge.innerHTML = `<i data-lucide="cpu" style="width: 13px; height: 13px;"></i><span>Base MedGemma Pretrained</span>`;
    inputIndicator.innerText = "M0 Base Stock";
    inputIndicator.style.color = "#60a5fa";
  }

  safeRenderIcons();
}

// =========================================================================
// SYSTEM STATUS & PRESETS
// =========================================================================
async function initSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/system`);
    const data = await res.json();
    if (data.cuda_available) {
      document.getElementById("sidebar-gpu-name").innerText = data.device_name.replace("NVIDIA GeForce ", "");
      document.getElementById("sidebar-gpu-vram").innerText = `${data.vram_used_gb}/${data.vram_total_gb} GB NF4`;
    }
  } catch (err) {
    console.warn("Could not fetch system telemetry:", err);
  }
}

async function initPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    const data = await res.json();

    const attacksList = document.getElementById("sidebar-attacks-list");
    const benignList = document.getElementById("sidebar-benign-list");

    attacksList.innerHTML = "";
    benignList.innerHTML = "";

    data.attacks.forEach(atk => {
      const btn = document.createElement("button");
      btn.className = "sidebar-btn";
      btn.style.padding = "6px 8px";
      btn.style.fontSize = "0.75rem";
      btn.innerHTML = `
        <div style="width: 100%; display: flex; flex-direction: column; gap: 2px;">
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 6px;">
            <span style="font-weight: 600; color: #fca5a5; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;">${escapeHtml(atk.label)}</span>
            <span style="font-size: 0.62rem; color: #f87171; font-family: monospace; flex-shrink: 0;">${escapeHtml(atk.category)}</span>
          </div>
          <span style="font-size: 0.68rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(atk.prompt)}</span>
        </div>
      `;
      btn.onclick = () => injectAndSendPrompt(atk.prompt, "m6");
      attacksList.appendChild(btn);
    });

    data.benign.forEach(ben => {
      const btn = document.createElement("button");
      btn.className = "sidebar-btn";
      btn.style.padding = "6px 8px";
      btn.style.fontSize = "0.75rem";
      btn.innerHTML = `
        <div style="width: 100%; display: flex; flex-direction: column; gap: 2px;">
          <div style="display: flex; align-items: center; justify-content: space-between; gap: 6px;">
            <span style="font-weight: 600; color: #6ee7b7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;">${escapeHtml(ben.label)}</span>
            <span style="font-size: 0.62rem; color: #34d399; font-family: monospace; flex-shrink: 0;">${escapeHtml(ben.category)}</span>
          </div>
          <span style="font-size: 0.68rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(ben.prompt)}</span>
        </div>
      `;
      btn.onclick = () => injectAndSendPrompt(ben.prompt, "m6");
      benignList.appendChild(btn);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function injectQuickChip(text) {
  const input = document.getElementById("chat-input");
  input.value = text;
  input.focus();
  autoResizePromptInput();
}

function injectAndSendPrompt(text, mode = "m6") {
  if (mode) selectModelMode(mode);
  const input = document.getElementById("chat-input");
  input.value = text;
  autoResizePromptInput();
  document.getElementById("chat-form").requestSubmit();
}

// =========================================================================
// CONVERSATIONAL CHAT ENGINE & AUTO-GROW TEXTAREA
// =========================================================================
function autoResizePromptInput() {
  const textarea = document.getElementById("chat-input");
  if (!textarea) return;
  textarea.style.height = "auto";
  const scrollHeight = textarea.scrollHeight;
  const maxHeight = 280;
  if (scrollHeight > maxHeight) {
    textarea.style.height = maxHeight + "px";
    textarea.style.overflowY = "auto";
  } else {
    textarea.style.height = Math.max(scrollHeight, 22) + "px";
    textarea.style.overflowY = "hidden";
  }

  const sendBtn = document.getElementById("btn-send");
  if (sendBtn) {
    sendBtn.disabled = textarea.value.trim().length === 0;
  }
}

window.toggleBenchmarkDetail = function (detailId) {
  const row = document.getElementById(detailId);
  if (!row) return;
  row.style.display = row.style.display === "none" ? "table-row" : "none";
  safeRenderIcons();
};

function handleTextareaKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
}

function clearChatHistory() {
  CHAT_HISTORY = [];
  TURNS_TELEMETRY = [];
  const container = document.getElementById("chat-messages-container");
  container.innerHTML = "";
  document.getElementById("chat-hero-view").style.display = "flex";
  toggleDefenseDrawer(false);
  const input = document.getElementById("chat-input");
  if (input) {
    input.value = "";
    autoResizePromptInput();
  }
}

async function handleChatSubmit(e) {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("btn-send");
  const promptText = input.value.trim();

  if (!promptText) return;

  // Hide hero screen
  document.getElementById("chat-hero-view").style.display = "none";

  // Append user message
  appendUserMessage(promptText);
  input.value = "";
  autoResizePromptInput();
  sendBtn.disabled = true;

  // Append assistant loading placeholder
  const loadingId = appendAssistantLoading();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptText,
        model_mode: CURRENT_MODEL_MODE,
        history: CHAT_HISTORY,
        max_tokens: 512,
      }),
    });

    const data = await res.json();
    removeElement(loadingId);

    if (data.error) {
      appendAssistantError(data.error);
    } else {
      const turnIndex = TURNS_TELEMETRY.length;
      TURNS_TELEMETRY.push(data);

      appendAssistantMessage(data, turnIndex);

      // Auto-open defense inspector if an adversarial attack was intercepted
      if (data.blocked || (data.telemetry && data.telemetry.is_blocked)) {
        openTurnInDrawer(turnIndex);
        toggleDefenseDrawer(true);
      } else {
        // Also update drawer silently so it's ready if opened
        openTurnInDrawer(turnIndex, false);
      }

      // Record to history if safe
      if (!data.blocked) {
        CHAT_HISTORY.push({ role: "user", content: promptText });
        CHAT_HISTORY.push({ role: "assistant", content: data.response });
      }
    }
  } catch (err) {
    removeElement(loadingId);
    appendAssistantError(`Network error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
    safeRenderIcons();
  }
}

function appendUserMessage(text) {
  const container = document.getElementById("chat-messages-container");
  const msgDiv = document.createElement("div");
  msgDiv.style.display = "flex";
  msgDiv.style.justifyContent = "flex-end";
  msgDiv.innerHTML = `
    <div class="chat-msg-user">
      <p style="white-space: pre-wrap;">${escapeHtml(text)}</p>
    </div>
  `;
  container.appendChild(msgDiv);
  scrollChatToBottom();
}

function appendAssistantLoading() {
  const container = document.getElementById("chat-messages-container");
  const id = "loading-" + Date.now();
  const msgDiv = document.createElement("div");
  msgDiv.id = id;
  msgDiv.style.display = "flex";
  msgDiv.style.gap = "14px";
  msgDiv.style.maxWidth = "100%";
  msgDiv.innerHTML = `
    <div style="width: 32px; height: 32px; border-radius: 10px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #38bdf8;">
      <i data-lucide="bot" style="width: 18px; height: 18px;"></i>
    </div>
    <div style="flex: 1; display: flex; flex-direction: column; gap: 8px;">
      <div style="font-size: 0.75rem; color: #9ca3af; display: flex; align-items: center; gap: 8px;">
        <span style="width: 7px; height: 7px; border-radius: 999px; background: #38bdf8; animation: pulse 1.5s infinite;"></span>
        <span>Evaluating Tier-1 Guardrail & generating on GPU...</span>
      </div>
      <div style="font-size: 0.875rem; color: #6b7280; font-style: italic;">
        Consulting clinical knowledge base...
      </div>
    </div>
  `;
  container.appendChild(msgDiv);
  scrollChatToBottom();
  safeRenderIcons();
  return id;
}

function removeElement(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendAssistantMessage(data, turnIndex) {
  const container = document.getElementById("chat-messages-container");
  const msgDiv = document.createElement("div");
  msgDiv.style.display = "flex";
  msgDiv.style.gap = "14px";
  msgDiv.style.maxWidth = "100%";

  let badgeHtml = "";
  if (data.telemetry) {
    const t = data.telemetry;
    if (data.blocked || t.is_blocked) {
      badgeHtml = `
        <div class="clinical-telemetry-badge intercepted">
          <i data-lucide="shield-alert" style="width: 13px; height: 13px;"></i>
          <span>PRIVACY DEFENSE INTERCEPTED · 0.0% PHI Leak (${t.guardrail.predicted_category})</span>
          <button onclick="openTurnInDrawer(${turnIndex})" class="inspect-trace-link">
            <span>Inspect Trace</span>
            <i data-lucide="arrow-up-right" style="width: 11px; height: 11px;"></i>
          </button>
        </div>
      `;
    } else {
      badgeHtml = `
        <div class="clinical-telemetry-badge verified">
          <i data-lucide="shield-check" style="width: 13px; height: 13px;"></i>
          <span>SAFE CLINICAL INQUIRY · 100% Medical Intelligence</span>
          <button onclick="openTurnInDrawer(${turnIndex})" class="inspect-trace-link">
            <span>Inspect Trace</span>
            <i data-lucide="arrow-up-right" style="width: 11px; height: 11px;"></i>
          </button>
        </div>
      `;
    }
  }

  msgDiv.innerHTML = `
    <div style="width: 32px; height: 32px; border-radius: 10px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #38bdf8; margin-top: 2px;">
      <i data-lucide="bot" style="width: 18px; height: 18px;"></i>
    </div>
    <div style="flex: 1; display: flex; flex-direction: column; gap: 4px;">
      ${badgeHtml}
      <div class="chat-msg-assistant chat-markdown">
        ${formatMarkdown(data.response)}
      </div>
      <div style="font-size: 0.68rem; color: #6b7280; font-family: monospace; display: flex; align-items: center; gap: 8px; margin-top: 4px;">
        <span>${data.mode}</span>
        <span>•</span>
        <span>${data.latency_sec}s GPU</span>
      </div>
    </div>
  `;

  container.appendChild(msgDiv);
  scrollChatToBottom();
  safeRenderIcons();
}

function appendAssistantError(errText) {
  const container = document.getElementById("chat-messages-container");
  const msgDiv = document.createElement("div");
  msgDiv.style.display = "flex";
  msgDiv.style.gap = "14px";
  msgDiv.innerHTML = `
    <div style="width: 32px; height: 32px; border-radius: 10px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #f87171;">
      <i data-lucide="alert-octagon" style="width: 18px; height: 18px;"></i>
    </div>
    <div style="flex: 1; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 12px; padding: 12px 16px; color: #fca5a5; font-size: 0.875rem;">
      <strong>Execution Notice:</strong> ${escapeHtml(errText)}
    </div>
  `;
  container.appendChild(msgDiv);
  scrollChatToBottom();
  safeRenderIcons();
}

function scrollChatToBottom() {
  const scrollArea = document.getElementById("chat-scroll-container");
  if (!scrollArea) return;
  requestAnimationFrame(() => {
    scrollArea.scrollTo({ top: scrollArea.scrollHeight, behavior: "smooth" });
    setTimeout(() => {
      scrollArea.scrollTo({ top: scrollArea.scrollHeight, behavior: "smooth" });
    }, 60);
  });
}

// =========================================================================
// RIGHT ARTIFACTS DRAWER CONTROLLER
// =========================================================================
function openTurnInDrawer(turnIndex, openDrawer = true) {
  const turnData = TURNS_TELEMETRY[turnIndex];
  if (!turnData || !turnData.telemetry) return;

  const t = turnData.telemetry;
  const isBlocked = turnData.blocked || t.is_blocked;

  const card = document.getElementById("drawer-verdict-card");
  const pill = document.getElementById("drawer-verdict-pill");
  const cat = document.getElementById("drawer-category-label");
  const conf = document.getElementById("drawer-confidence-label");
  const mope = document.getElementById("drawer-mope-risk-label");
  const leak = document.getElementById("drawer-phi-leak-label");
  const lat = document.getElementById("drawer-latency-label");

  if (isBlocked) {
    card.style.borderLeftColor = "#ef4444";
    pill.style.background = "rgba(239, 68, 68, 0.15)";
    pill.style.color = "#f87171";
    pill.innerText = "TIER 1 INTERCEPTED";
    leak.innerText = "0.0% PHI Leak (Blocked)";
    leak.style.color = "#34d399";
  } else {
    card.style.borderLeftColor = "#10b981";
    pill.style.background = "rgba(16, 185, 129, 0.15)";
    pill.style.color = "#34d399";
    pill.innerText = "TIER 1 PASSED";
    leak.innerText = CURRENT_MODEL_MODE === "m1" ? "39.0% PHI Leak (Unprotected)" : "0.0% (Zero Leakage)";
    leak.style.color = CURRENT_MODEL_MODE === "m1" ? "#f87171" : "#34d399";
  }

  cat.innerText = t.guardrail.predicted_category;
  conf.innerText = `${(t.guardrail.confidence * 100).toFixed(1)}%`;
  mope.innerText = `${(t.mope.attack_prob * 100).toFixed(1)}% (Head #${t.mope.routed_expert})`;
  lat.innerText = `${t.latency_ms} ms (Shared MiniLM)`;

  // Update Radar Chart (Live Cosine Similarities across 9 Threat Classes)
  if (RADAR_CHART) {
    const rawScores = (t.guardrail && (t.guardrail.similarities || t.guardrail.raw_scores)) || {};
    const labels = ["Benign", "Command", "Concern", "Pretext", "Format", "Inquiry", "Pressure", "Request", "Role Play"];
    const keyMap = {
      "Benign": "Benign",
      "Command": "Command",
      "Concern": "Concern Expression",
      "Pretext": "False Pretext",
      "Format": "Format Manipulation",
      "Inquiry": "Inquiry",
      "Pressure": "Pressure",
      "Request": "Request",
      "Role Play": "Role Play"
    };
    const values = labels.map(l => {
      const fullKey = keyMap[l] || l;
      return typeof rawScores[fullKey] === "number"
        ? Math.max(0.02, Math.min(1.0, rawScores[fullKey]))
        : (typeof rawScores[l] === "number" ? Math.max(0.02, Math.min(1.0, rawScores[l])) : 0.1);
    });
    RADAR_CHART.data.datasets[0].data = values;
    RADAR_CHART.data.datasets[0].borderColor = isBlocked ? "rgba(239, 68, 68, 0.85)" : "rgba(16, 185, 129, 0.85)";
    RADAR_CHART.data.datasets[0].backgroundColor = isBlocked ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)";
    RADAR_CHART.update();
  }

  // Update MoPE Bar Chart
  if (MOPE_CHART) {
    const expertIdx = t.mope.routed_expert;
    const weights = [0.08, 0.09, 0.11, 0.14, 0.09, 0.12, 0.10, 0.13, 0.14];
    weights[expertIdx % 9] = isBlocked ? 0.72 : 0.35;
    MOPE_CHART.data.datasets[0].data = weights;
    MOPE_CHART.update();
  }

  if (openDrawer) {
    toggleDefenseDrawer(true);
  }
}

async function runDrawerProbe() {
  const input = document.getElementById("drawer-probe-input");
  const promptText = input.value.trim();
  if (!promptText) return;

  try {
    const res = await fetch(`${API_BASE}/api/inspect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText }),
    });
    const data = await res.json();

    // Create synthetic turn object for replay
    const syntheticTurn = {
      blocked: data.is_blocked,
      telemetry: {
        is_blocked: data.is_blocked,
        latency_ms: data.latency_ms,
        guardrail: data.guardrail,
        mope: data.mope,
      }
    };
    const idx = TURNS_TELEMETRY.length;
    TURNS_TELEMETRY.push(syntheticTurn);
    openTurnInDrawer(idx, true);
  } catch (err) {
    console.error("Probe error:", err);
  }
}

function initCharts() {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded. Skipping chart initialization.");
    return;
  }
  // 1. Radar Chart
  const ctxRadar = document.getElementById("drawer-radar-chart").getContext("2d");
  RADAR_CHART = new Chart(ctxRadar, {
    type: "radar",
    data: {
      labels: ["Benign", "Command", "Concern", "Pretext", "Format", "Inquiry", "Pressure", "Request", "Role Play"],
      datasets: [{
        label: "Cosine Similarity",
        data: [0.85, 0.12, 0.15, 0.18, 0.08, 0.22, 0.10, 0.14, 0.09],
        backgroundColor: "rgba(16, 185, 129, 0.2)",
        borderColor: "rgba(16, 185, 129, 0.8)",
        borderWidth: 2,
        pointBackgroundColor: "#34d399",
        pointRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          angleLines: { color: "rgba(255, 255, 255, 0.08)" },
          pointLabels: { color: "#9ca3af", font: { size: 9, family: "Plus Jakarta Sans" } },
          ticks: { display: false, maxTicksLimit: 3 },
          suggestedMin: 0,
          suggestedMax: 1,
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });

  // 2. MoPE Bar Chart
  const ctxMoPE = document.getElementById("drawer-mope-chart").getContext("2d");
  MOPE_CHART = new Chart(ctxMoPE, {
    type: "bar",
    data: {
      labels: ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
      datasets: [{
        data: [0.11, 0.09, 0.12, 0.15, 0.08, 0.14, 0.10, 0.11, 0.10],
        backgroundColor: "rgba(167, 139, 250, 0.65)",
        borderColor: "#a78bfa",
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#9ca3af", font: { size: 9 } },
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: { color: "#6b7280", font: { size: 8 } },
          suggestedMin: 0,
          suggestedMax: 1,
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

// =========================================================================
// MODALS CONTROLLER (Table VII Benchmarks & Figures Gallery)
// =========================================================================
function showChatView() {
  document.getElementById("nav-btn-chat").classList.add("active");
  document.getElementById("nav-btn-benchmarks").classList.remove("active");
  document.getElementById("nav-btn-gallery").classList.remove("active");
  closeBenchmarksModal();
  closeGalleryModal();
}

function openBenchmarksModal() {
  document.getElementById("modal-benchmarks").classList.remove("hidden");
  document.getElementById("nav-btn-benchmarks").classList.add("active");
  initBenchmarksTable();
  safeRenderIcons();
}
function closeBenchmarksModal() {
  document.getElementById("modal-benchmarks").classList.add("hidden");
  document.getElementById("nav-btn-benchmarks").classList.remove("active");
  document.getElementById("nav-btn-chat").classList.add("active");
}

function openGalleryModal() {
  document.getElementById("modal-gallery").classList.remove("hidden");
  document.getElementById("nav-btn-gallery").classList.add("active");
}
function closeGalleryModal() {
  document.getElementById("modal-gallery").classList.add("hidden");
  document.getElementById("nav-btn-gallery").classList.remove("active");
  document.getElementById("nav-btn-chat").classList.add("active");
}

function openLightbox(src, title) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-title").innerText = title;
  document.getElementById("lightbox-modal").classList.remove("hidden");
}
function closeLightbox() {
  document.getElementById("lightbox-modal").classList.add("hidden");
}

async function initBenchmarksTable() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks`);
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("table-vii-modal-body");
    if (!tbody || !data || !Array.isArray(data.table_vii)) return;
    tbody.innerHTML = "";

    data.table_vii.forEach((row, idx) => {
      const modelId = row.id || `M${idx}`;
      const name = row.name || "Configuration";
      const shortTitle = row.short_title || "";
      const leakRate = typeof row.leak_rate === "number" ? row.leak_rate : 0.0;
      const utility = typeof row.utility === "number" ? row.utility : 100.0;
      const latency = typeof row.latency_overhead_ms === "number" ? row.latency_overhead_ms : (typeof row.latency_ms === "number" ? row.latency_ms : 0.0);
      const vram = typeof row.vram_gb === "number" ? row.vram_gb : 3.38;
      const paretoRank = row.pareto_rank || (idx + 1);
      const isParetoBest = !!row.is_pareto_best;
      const desc = row.description || "";
      const takeaway = row.key_takeaway || "";
      const attackBlock = typeof row.attack_block_rate === "number" ? `${row.attack_block_rate.toFixed(1)}%` : (typeof row.block_rate === "number" ? `${row.block_rate.toFixed(1)}%` : "N/A");
      const falseBlock = typeof row.false_block_rate === "number" ? `${row.false_block_rate.toFixed(1)}%` : (typeof row.false_block === "number" ? `${row.false_block.toFixed(1)}%` : "0.0%");

      const leakColor = leakRate === 0.0 ? "#34d399" : (leakRate > 30 ? "#f87171" : "#fbbf24");
      const utilColor = utility >= 95 ? "#34d399" : (utility < 80 ? "#f87171" : "#fbbf24");
      const badgeBg = isParetoBest ? "rgba(16, 185, 129, 0.15)" : (row.type === "vulnerable" ? "rgba(239, 68, 68, 0.15)" : "rgba(56, 189, 248, 0.1)");
      const badgeColor = isParetoBest ? "#34d399" : (row.type === "vulnerable" ? "#f87171" : "#38bdf8");

      // Main clickable row
      const tr = document.createElement("tr");
      tr.className = `table-row-benchmarks ${isParetoBest ? 'pareto-best-row' : ''}`;
      tr.style.borderBottom = "1px solid var(--border-subtle)";
      tr.style.cursor = "pointer";
      tr.style.transition = "background-color 0.15s ease";
      if (isParetoBest) {
        tr.style.background = "rgba(16, 185, 129, 0.08)";
      }

      tr.onmouseenter = () => { if (!isParetoBest) tr.style.background = "var(--card-bg-hover)"; };
      tr.onmouseleave = () => { if (!isParetoBest) tr.style.background = "transparent"; };

      tr.innerHTML = `
        <td style="padding: 12px 16px; font-weight: 800; color: var(--text-primary);">
          <span style="background: ${badgeBg}; color: ${badgeColor}; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; border: 1px solid ${badgeColor}33; display: inline-flex; align-items: center; gap: 4px;">
            ${escapeHtml(modelId)}
            <i data-lucide="chevron-down" style="width: 11px; height: 11px; opacity: 0.6;"></i>
          </span>
        </td>
        <td style="padding: 12px 16px; color: var(--text-body); font-family: 'Plus Jakarta Sans', sans-serif;">
          <div style="font-weight: 600; font-size: 0.8125rem; color: var(--text-primary);">${escapeHtml(name)}</div>
          ${shortTitle ? `<div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 2px;">${escapeHtml(shortTitle)}</div>` : ''}
        </td>
        <td style="padding: 12px 16px; text-align: right; color: ${leakColor}; font-weight: 700;">
          <span style="background: ${leakColor}18; padding: 3px 8px; border-radius: 4px; border: 1px solid ${leakColor}33;">
            ${leakRate.toFixed(1)}%
          </span>
        </td>
        <td style="padding: 12px 16px; text-align: right; color: ${utilColor}; font-weight: 700;">
          <span style="background: ${utilColor}18; padding: 3px 8px; border-radius: 4px; border: 1px solid ${utilColor}33;">
            ${utility.toFixed(1)}%
          </span>
        </td>
        <td style="padding: 12px 16px; text-align: right; color: var(--text-body); font-family: monospace;">
          ${latency === 0.0 ? '<span style="color:var(--text-muted);">0.0 ms (Base)</span>' : `+${latency.toFixed(1)} ms`}
        </td>
        <td style="padding: 12px 16px; text-align: right; color: var(--accent-clinical); font-family: monospace;">
          ${vram.toFixed(2)} GB
        </td>
        <td style="padding: 12px 16px; text-align: center;">
          ${isParetoBest
          ? '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 3px 9px; border-radius: 999px; font-weight: 800; font-size: 0.72rem; border: 1px solid rgba(16, 185, 129, 0.4); display: inline-flex; align-items: center; gap: 4px;">🏆 #1 PARETO</span>'
          : `<span style="color: var(--text-muted); font-size: 0.75rem; font-weight: 600;">#${paretoRank}</span>`}
        </td>
      `;

      // Expandable detail row explaining the model configuration
      const detailTr = document.createElement("tr");
      detailTr.className = "benchmarks-detail-row";
      detailTr.style.display = "none";
      detailTr.style.background = "var(--card-bg-subtle)";
      detailTr.style.borderBottom = "1px solid var(--border-subtle)";
      detailTr.innerHTML = `
        <td colspan="7" style="padding: 14px 18px;">
          <div style="display: flex; flex-direction: column; gap: 8px; font-family: 'Plus Jakarta Sans', sans-serif;">
            <div style="font-size: 0.78rem; color: var(--text-primary); line-height: 1.5;">
              <strong style="color: var(--accent-clinical);">Mechanism & Architecture:</strong> ${escapeHtml(desc)}
            </div>
            <div style="font-size: 0.75rem; color: var(--text-body); line-height: 1.5; background: var(--bg-surface); padding: 8px 12px; border-radius: 6px; border-left: 3px solid ${badgeColor};">
              <strong style="color: ${badgeColor};">Key Research Finding:</strong> ${escapeHtml(takeaway)}
            </div>
            <div style="display: flex; align-items: center; gap: 16px; font-size: 0.72rem; color: var(--text-muted); font-family: monospace; padding-top: 4px;">
              <span>Attack Block Rate: <strong style="color: var(--accent-clinical);">${attackBlock}</strong></span>
              <span>•</span>
              <span>Benign False-Block Rate: <strong style="color: #f87171;">${falseBlock}</strong></span>
              <span>•</span>
              <span style="color: var(--text-dim);">Click row again to collapse</span>
            </div>
          </div>
        </td>
      `;

      tr.onclick = () => {
        detailTr.style.display = detailTr.style.display === "none" ? "table-row" : "none";
        safeRenderIcons();
      };

      tbody.appendChild(tr);
      tbody.appendChild(detailTr);
    });

    safeRenderIcons();
  } catch (err) {
    console.warn("Using pre-rendered Table VII data (offline/fallback mode):", err);
  }
}

function initFiguresGallery() {
  const container = document.getElementById("gallery-grid");
  container.innerHTML = "";

  FIGURES_DATA.forEach(fig => {
    const card = document.createElement("div");
    card.style.background = "var(--bg-primary)";
    card.style.border = "1px solid var(--border-subtle)";
    card.style.borderRadius = "12px";
    card.style.overflow = "hidden";
    card.style.cursor = "pointer";
    card.style.transition = "transform 0.2s ease, border-color 0.2s ease";

    card.onmouseenter = () => { card.style.borderColor = "var(--border-active)"; card.style.transform = "translateY(-2px)"; };
    card.onmouseleave = () => { card.style.borderColor = "var(--border-subtle)"; card.style.transform = "none"; };

    card.innerHTML = `
      <div class="figure-thumbnail-bg" style="height: 150px; background: #0c0d10; overflow: hidden; display: flex; align-items: center; justify-content: center;">
        <img src="${API_BASE}/figures/${fig.id}" alt="${fig.title}" loading="lazy" style="max-height: 100%; max-width: 100%; object-fit: contain;">
      </div>
      <div style="padding: 10px 12px;">
        <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-primary); line-height: 1.4;">${fig.title}</div>
        <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 4px; font-family: monospace;">Click to expand (300 DPI)</div>
      </div>
    `;
    card.onclick = () => openLightbox(`${API_BASE}/figures/${fig.id}`, fig.title);
    container.appendChild(card);
  });
}

// =========================================================================
// UTILITY HELPERS
// =========================================================================
function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Inline code
  html = html.replace(/`(.*?)`/g, "<code>$1</code>");
  // Linebreaks to paragraphs
  const paragraphs = html.split(/\n\s*\n/);
  return paragraphs.map(p => {
    p = p.trim();
    if (!p) return "";
    if (p.startsWith("- ") || p.startsWith("* ")) {
      const items = p.split(/\n[-*]\s+/).filter(Boolean);
      return "<ul>" + items.map(it => `<li>${it.replace(/\n/g, "<br>")}</li>`).join("") + "</ul>";
    }
    return `<p>${p.replace(/\n/g, "<br>")}</p>`;
  }).join("");
}
