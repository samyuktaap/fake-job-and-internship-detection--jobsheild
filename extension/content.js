/**
 * extension/content.js
 * Scrapes job advertisement fields from LinkedIn, Indeed, etc.,
 * queries the local JobShield ML API, and injects a real-time risk badge.
 */

const API_BASE = "http://127.0.0.1:8000";

function extractJobDetails() {
  const url = window.location.href;
  let title = "";
  let company = "";
  let description = "";
  let hasLogo = 0;
  let hasQuestions = 0;
  let telecommuting = 0;

  // 1. LinkedIn Selectors
  if (url.includes("linkedin.com")) {
    // Title
    const titleCandidates = [
      "h1.t-24",
      ".job-details-jobs-unified-top-card__job-title",
      ".jobs-unified-top-card__job-title",
      ".jobs-search__job-details h1",
      ".jobs-search__job-details h2",
      ".jobs-details h1",
      "h1",
      "h2.t-24"
    ];
    for (const sel of titleCandidates) {
      const el = document.querySelector(sel);
      if (el && el.innerText.trim().length > 1) {
        title = el.innerText.trim();
        break;
      }
    }

    // Company
    const compCandidates = [
      ".job-details-jobs-unified-top-card__company-name",
      ".jobs-unified-top-card__company-name",
      ".job-details-jobs-unified-top-card__primary-description a",
      ".jobs-details__top-card a",
      "a[href*='/company/']",
      ".jobs-unified-top-card__subtitle-primary-grouping a"
    ];
    for (const sel of compCandidates) {
      const el = document.querySelector(sel);
      if (el && el.innerText.trim().length > 1) {
        company = el.innerText.trim();
        break;
      }
    }

    // Description - prioritize containers that actually have the full text
    const descCandidates = [
      ".jobs-description-content__text",
      ".jobs-description__content",
      "article.jobs-description__container",
      ".jobs-box__html-content",
      ".jobs-description",
      "[data-job-id] .jobs-description",
      ".job-details-module"
    ];

    for (const sel of descCandidates) {
      const el = document.querySelector(sel);
      if (el && el.innerText.trim().length > 40) {
        description = el.innerText.trim();
        break;
      }
    }

    // If still not found, check #job-details and its parent or container
    if (!description || description.length < 40) {
      const jobDetailsEl = document.querySelector("#job-details");
      if (jobDetailsEl) {
        if (jobDetailsEl.innerText.trim().length > 40) {
          description = jobDetailsEl.innerText.trim();
        } else {
          // #job-details is often just <span>About the job</span>. Look at its container!
          const container = jobDetailsEl.closest("article, .jobs-description__content, section, .jobs-box, div[class*='description']");
          if (container && container.innerText.trim().length > 40) {
            description = container.innerText.trim();
          }
        }
      }
    }

    // Fallback: search right-side pane on LinkedIn search (/jobs/search)
    if (!description || description.length < 40) {
      const rightPane = document.querySelector(".jobs-search__job-details--container, .jobs-search__job-details, .jobs-details__main-content");
      if (rightPane && rightPane.innerText.trim().length > 80) {
        description = rightPane.innerText.trim();
      }
    }

    // Logo detection
    const logoEl = document.querySelector(
      ".job-details-jobs-unified-top-card img, .jobs-unified-top-card img, .artdeco-entity-lockup__image img, .job-details-jobs-unified-top-card__company-logo img, .jobs-unified-top-card__company-logo img, .job-details-jobs-unified-top-card__logo-container img, .jobs-unified-top-card__logo-container img, .jobs-details__top-card img, img[alt*='logo' i], img[class*='company-logo']"
    );
    if ((logoEl && !logoEl.src.includes("ghost") && logoEl.src.length > 5) || company.length > 0) {
      hasLogo = 1;
    }

    const fullText = (title + " " + description).toLowerCase();
    if (fullText.includes("remote") || fullText.includes("work from home") || fullText.includes("hybrid")) {
      telecommuting = 1;
    }
    if (
      document.querySelector(".jobs-apply-button--top-card") ||
      document.querySelector("[data-job-id]") ||
      fullText.includes("screening") ||
      fullText.includes("apply")
    ) {
      hasQuestions = 1;
    }
  }
  // 2. Indeed Selectors
  else if (url.includes("indeed.com")) {
    const titleEl = document.querySelector("h1.jobsearch-JobInfoHeader-title, h1");
    if (titleEl) title = titleEl.innerText.trim();

    const compEl = document.querySelector('[data-testid="inlineHeader-companyName"]');
    if (compEl) company = compEl.innerText.trim();

    const descEl = document.querySelector("#jobDescriptionText, .jobsearch-jobDescriptionText");
    if (descEl) description = descEl.innerText.trim();

    const fullText = (title + " " + description).toLowerCase();
    if (fullText.includes("remote")) telecommuting = 1;
    hasLogo = 1;
  }
  // 3. Fallback generic extractor
  else {
    const h1 = document.querySelector("h1");
    if (h1) title = h1.innerText.trim();
    const article = document.querySelector("article, main, .job-description, [class*='description']");
    if (article) description = article.innerText.trim();
  }

  // If description is still empty, grab selected text if user highlighted any
  if (!description) {
    const selected = window.getSelection().toString().trim();
    if (selected && selected.length > 30) {
      description = selected;
    }
  }

  return {
    title: title || document.title,
    description: description || "",
    company_profile: company ? `Verified organization on LinkedIn: ${company}` : "",
    has_company_logo: hasLogo,
    has_questions: hasQuestions,
    telecommuting: telecommuting,
    url: url
  };
}

let lastInspectedTitle = "";

async function analyzeActiveJob() {
  const details = extractJobDetails();
  if (!details.description || details.description.length < 30) return;
  if (details.title === lastInspectedTitle) return;
  lastInspectedTitle = details.title;

  removeExistingBadge();
  insertLoadingBadge();

  const payload = {
    title: details.title,
    description: details.description,
    company_profile: details.company_profile,
    has_company_logo: details.has_company_logo,
    has_questions: details.has_questions,
    telecommuting: details.telecommuting
  };

  chrome.runtime.sendMessage({ type: "PREDICT_JOB", payload: payload }, (response) => {
    if (chrome.runtime.lastError || !response || !response.success) {
      console.warn("[JobShield] Could not query ML server:", chrome.runtime.lastError?.message || response?.error);
      renderOfflineBadge();
      return;
    }
    renderJobShieldBadge(response.data);
  });
}

function removeExistingBadge() {
  const existing = document.getElementById("jobshield-floating-pill");
  if (existing) existing.remove();
}

function insertLoadingBadge() {
  const badge = document.createElement("div");
  badge.id = "jobshield-floating-pill";
  badge.className = "jobshield-pill jobshield-loading";
  badge.innerHTML = `
    <span class="jobshield-icon">🛡️</span>
    <span>JobShield: Scanning listing...</span>
  `;
  document.body.appendChild(badge);
}

function renderJobShieldBadge(data) {
  removeExistingBadge();
  const badge = document.createElement("div");
  badge.id = "jobshield-floating-pill";

  const isScam = data.badge_color === "red" || data.badge_color === "orange";
  badge.className = `jobshield-pill ${isScam ? "jobshield-danger" : "jobshield-safe"}`;

  const redCount = data.red_flags ? data.red_flags.length : 0;
  const flagText = redCount > 0 ? ` (${redCount} Flags)` : "";

  badge.innerHTML = `
    <div class="jobshield-header">
      <span class="jobshield-icon">${isScam ? "🚨" : "🛡️"}</span>
      <strong>JobShield: ${data.prediction_label}</strong>
      <span class="jobshield-risk-tag">${data.risk_score_pct}% Risk</span>
    </div>
    ${
      isScam && data.red_flags && data.red_flags.length > 0
        ? `<div class="jobshield-details">
            <span class="jobshield-flag-title">Warning Signals Detected:</span>
            <ul>
              ${data.red_flags.slice(0, 2).map(rf => `<li>• ${rf.detail}</li>`).join("")}
            </ul>
           </div>`
        : `<div class="jobshield-details-safe">
            <span class="jobshield-flag-title" style="font-weight:700; color:#14532d">Verified Legitimacy Markers:</span>
            <ul style="margin:2px 0 0 0; padding-left:0; list-style:none">
              ${(data.green_flags || []).slice(0, 2).map(gf => `<li>• <strong>${gf.category}:</strong> ${gf.detail}</li>`).join("")}
            </ul>
           </div>`
    }
  `;

  document.body.appendChild(badge);
}

function renderOfflineBadge() {
  removeExistingBadge();
  const badge = document.createElement("div");
  badge.id = "jobshield-floating-pill";
  badge.className = "jobshield-pill jobshield-offline";
  badge.innerHTML = `
    <span class="jobshield-icon">⚠️</span>
    <span>JobShield Local API Offline (Check port 8000)</span>
  `;
  document.body.appendChild(badge);
}

// Listen for messages from extension popup
chrome.runtime.onMessage?.addListener((req, sender, sendResponse) => {
  if (req.action === "GET_JOB_DATA") {
    sendResponse(extractJobDetails());
  }
});

// Periodic observer to check for job page changes (SPA navigation on LinkedIn)
setInterval(analyzeActiveJob, 2500);
