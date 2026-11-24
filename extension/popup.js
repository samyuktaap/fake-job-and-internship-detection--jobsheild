/**
 * extension/popup.js
 * Controls extension popup interactions and talks to local backend.
 */

const API_BASE = "http://127.0.0.1:8000";

async function checkApiHealth() {
  const dot = document.getElementById("status-dot");
  const label = document.getElementById("status-label");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      dot.style.background = "#22c55e";
      label.textContent = "API Online (8000)";
      return true;
    }
  } catch (e) {
    dot.style.background = "#ef4444";
    label.textContent = "API Offline";
    return false;
  }
}

function getJobDataFromTab(tabId) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, { action: "GET_JOB_DATA" }, (res) => {
      if (!chrome.runtime.lastError && res && res.description && res.description.length > 20) {
        resolve(res);
      } else {
        // Tab was opened before extension was loaded or content script listener not yet ready.
        // Inject content.js and content.css dynamically using scripting API!
        chrome.scripting.insertCSS({ target: { tabId: tabId }, files: ["content.css"] }, () => {
          chrome.scripting.executeScript({ target: { tabId: tabId }, files: ["content.js"] }, () => {
            if (chrome.runtime.lastError) {
              resolve(null);
              return;
            }
            setTimeout(() => {
              chrome.tabs.sendMessage(tabId, { action: "GET_JOB_DATA" }, (res2) => {
                resolve(chrome.runtime.lastError ? null : res2);
              });
            }, 250);
          });
        });
      }
    });
  });
}

document.getElementById("btn-scan").addEventListener("click", async () => {
  const isHealthy = await checkApiHealth();
  if (!isHealthy) {
    alert("Please ensure the JobShield server is running: python -m uvicorn backend.main:app");
    return;
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  const resultBox = document.getElementById("result-box");
  const verdictCard = document.getElementById("verdict-card");
  const verdictTitle = document.getElementById("verdict-title");
  const riskScore = document.getElementById("risk-score");
  const flagsList = document.getElementById("flags-list");

  resultBox.style.display = "block";
  verdictTitle.textContent = "Reading listing from page...";
  riskScore.textContent = "";
  flagsList.innerHTML = "";

  const response = await getJobDataFromTab(tab.id);

  if (!response || !response.description || response.description.length < 20) {
    resultBox.style.display = "none";
    alert(
      "No job description detected on this page.\n\n" +
      "Quick fixes:\n" +
      "1. Press F5 (refresh) on the LinkedIn tab once so the newly loaded extension attaches.\n" +
      "2. Click directly on a job title in the left list so the right-side job description opens.\n" +
      "3. You can also highlight (select) any text in the job posting and click Scan!"
    );
    return;
  }

  verdictTitle.textContent = "Analyzing with AI...";

  try {
    const apiRes = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: response.title || "Job Posting",
        description: response.description,
        company_profile: response.company_profile || "",
        has_company_logo: response.has_company_logo || 0,
        has_questions: response.has_questions || 0,
        telecommuting: response.telecommuting || 0
      })
    });

    if (!apiRes.ok) {
      throw new Error(`API error: ${apiRes.statusText}`);
    }

    const data = await apiRes.json();
    const isScam = data.badge_color === "red" || data.badge_color === "orange";

    verdictCard.className = `verdict-card ${isScam ? "verdict-danger" : "verdict-safe"}`;
    verdictTitle.textContent = `${isScam ? "🚨" : "🛡️"} ${data.prediction_label}`;
    riskScore.textContent = `Estimated Scam Risk: ${data.risk_score_pct}% (${data.risk_level})`;

    flagsList.innerHTML = "";

    if (!isScam) {
      // LIKELY GENUINE CASE: Prominently show why the job is genuine!
      const greenHeader = document.createElement("div");
      greenHeader.className = "signals-header-safe";
      greenHeader.textContent = "✅ Verified Legitimacy Markers:";
      flagsList.appendChild(greenHeader);

      if (data.green_flags && data.green_flags.length > 0) {
        data.green_flags.forEach(gf => {
          const li = document.createElement("li");
          li.className = "flag-item-green";
          li.innerHTML = `<strong>• ${gf.category}:</strong> ${gf.detail}`;
          flagsList.appendChild(li);
        });
      }

      // Show top ML positive language signals if available
      if (data.key_genuine_signals && data.key_genuine_signals.length > 0) {
        const li = document.createElement("li");
        li.className = "flag-item-green";
        const wordsHtml = data.key_genuine_signals.map(w => `<span class="ml-keywords">${w}</span>`).join(" ");
        li.innerHTML = `<strong>• Authentic Career Language:</strong> ${wordsHtml}`;
        flagsList.appendChild(li);
      }

      // Subtle note for any minor observations (e.g. missing optional fields)
      if (data.red_flags && data.red_flags.length > 0) {
        const noteHeader = document.createElement("div");
        noteHeader.className = "signals-notes";
        noteHeader.textContent = "ℹ️ Informational Notes:";
        flagsList.appendChild(noteHeader);
        data.red_flags.forEach(rf => {
          const li = document.createElement("li");
          li.className = "flag-item-note";
          li.textContent = `• ${rf.category}: ${rf.detail}`;
          flagsList.appendChild(li);
        });
      }
    } else {
      // SCAM / HIGH RISK CASE: Show red warning flags!
      const redHeader = document.createElement("div");
      redHeader.className = "signals-header-danger";
      redHeader.textContent = "🚨 Risk Warning Indicators:";
      flagsList.appendChild(redHeader);

      if (data.red_flags && data.red_flags.length > 0) {
        data.red_flags.forEach(rf => {
          const li = document.createElement("li");
          li.className = "flag-item-red";
          li.innerHTML = `<strong>• [${rf.category}]:</strong> ${rf.detail}`;
          flagsList.appendChild(li);
        });
      }
    }
  } catch (err) {
    verdictTitle.textContent = "Error scanning listing";
    riskScore.textContent = err.message;
  }
});

checkApiHealth();
