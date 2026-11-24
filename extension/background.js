/**
 * extension/background.js
 * Background service worker for JobShield extension.
 * Bypasses page-level HTTPS mixed-content restrictions by performing API calls
 * in the privileged extension environment.
 */

const API_BASE = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "PREDICT_JOB") {
    fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request.payload)
    })
      .then(async (response) => {
        if (!response.ok) {
          const err = await response.json();
          sendResponse({ success: false, error: err.detail || "Prediction failed" });
        } else {
          const data = await response.json();
          sendResponse({ success: true, data: data });
        }
      })
      .catch((err) => {
        sendResponse({ success: false, error: err.message || "Cannot connect to JobShield backend at 127.0.0.1:8000" });
      });

    return true; // Keep message channel open for async response
  }

  if (request.type === "CHECK_HEALTH") {
    fetch(`${API_BASE}/api/health`)
      .then((res) => res.json())
      .then((data) => sendResponse({ success: true, data: data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }
});
