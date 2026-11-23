const PRESETS = {
  prime: {
    person_age: 35,
    person_emp_length: 9,
    person_income: 1200000,
    person_home_ownership: "MORTGAGE",
    loan_intent: "HOMEIMPROVEMENT",
    loan_grade: "A",
    loan_amnt: 180000,
    loan_int_rate: 7.9,
    loan_percent_income: 0.15,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 10,
  },
  borderline: {
    person_age: 26,
    person_emp_length: 2,
    person_income: 420000,
    person_home_ownership: "RENT",
    loan_intent: "MEDICAL",
    loan_grade: "D",
    loan_amnt: 150000,
    loan_int_rate: 15.2,
    loan_percent_income: 0.36,
    cb_person_default_on_file: "N",
    cb_person_cred_hist_length: 3,
  },
  highrisk: {
    person_age: 22,
    person_emp_length: 0,
    person_income: 240000,
    person_home_ownership: "RENT",
    loan_intent: "DEBTCONSOLIDATION",
    loan_grade: "F",
    loan_amnt: 140000,
    loan_int_rate: 19.5,
    loan_percent_income: 0.58,
    cb_person_default_on_file: "Y",
    cb_person_cred_hist_length: 2,
  },
};

const form = document.getElementById("riskForm");
const incomeInput = document.getElementById("person_income");
const loanAmtInput = document.getElementById("loan_amnt");
const ratioInput = document.getElementById("loan_percent_income");
const submitBtn = document.getElementById("submitBtn");
const errorBanner = document.getElementById("errorBanner");

function syncLoanToIncomeRatio() {
  const income = parseFloat(incomeInput.value);
  const amount = parseFloat(loanAmtInput.value);
  if (income > 0 && amount >= 0) {
    ratioInput.value = Math.min(amount / income, 1).toFixed(2);
  }
}

incomeInput.addEventListener("input", syncLoanToIncomeRatio);
loanAmtInput.addEventListener("input", syncLoanToIncomeRatio);

async function checkApiHealth() {
  const pill = document.getElementById("apiStatus");
  const label = document.getElementById("apiStatusText");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    pill.className = "api-pill online";
    label.textContent = "API Online";
    if (typeof data.decision_threshold === "number") {
      const pct = (data.decision_threshold * 100).toFixed(1);
      document.getElementById("thresholdLabel").textContent = `DECISION CUTOFF: ${pct}%`;
      document.getElementById("thresholdMarker").style.left = `${pct}%`;
      document.getElementById("tThreshold").textContent = `${pct}%`;
    }
  } catch {
    pill.className = "api-pill offline";
    label.textContent = "API Offline";
  }
}

function renderShapDrivers(impacts) {
  const container = document.getElementById("shapContainer");
  if (!impacts || impacts.length === 0) {
    container.innerHTML = `<div class="shap-empty">SHAP feature attributions unavailable for this estimator.</div>`;
    return;
  }

  const maxAbs = Math.max(...impacts.map((d) => Math.abs(d.impact)), 0.01);

  container.innerHTML = impacts
    .map((item) => {
      const isPos = item.impact >= 0;
      const widthPct = Math.min((Math.abs(item.impact) / maxAbs) * 48, 48).toFixed(1);
      const signedVal = `${isPos ? "+" : ""}${item.impact.toFixed(3)}`;
      return `
        <div class="shap-row">
          <span class="shap-name" title="${item.feature}">${item.feature}</span>
          <div class="shap-bar-zone">
            <div class="shap-axis"></div>
            <div class="shap-bar ${isPos ? "pos" : "neg"}" style="width: ${widthPct}%"></div>
          </div>
          <span class="shap-val ${isPos ? "pos" : "neg"}">${signedVal}</span>
        </div>
      `;
    })
    .join("");
}

function updateDashboard(data) {
  const probPct = (data.default_probability * 100).toFixed(1);
  const threshPct = (data.threshold * 100).toFixed(1);
  const isHigh = data.default_prediction === 1;
  const isModerate = !isHigh && data.default_probability >= data.threshold * 0.65;

  document.getElementById("probValue").textContent = `${probPct}%`;
  document.getElementById("tProb").textContent = `${probPct}%`;
  document.getElementById("tThreshold").textContent = `${threshPct}%`;
  document.getElementById("tTier").textContent = data.risk_tier || data.Result;
  document.getElementById("tDecision").textContent = isHigh ? "DECLINE / MANUAL REVIEW" : "APPROVE";

  document.getElementById("thresholdLabel").textContent = `DECISION CUTOFF: ${threshPct}%`;
  document.getElementById("thresholdMarker").style.left = `${Math.min(Math.max(parseFloat(threshPct), 2), 98)}%`;

  const meterFill = document.getElementById("meterFill");
  meterFill.style.width = `${Math.min(parseFloat(probPct), 100)}%`;

  const badge = document.getElementById("riskBadge");
  if (isHigh) {
    badge.className = "risk-badge high";
    badge.textContent = "HIGH DEFAULT RISK";
    meterFill.style.backgroundColor = "#ef4444";
  } else if (isModerate) {
    badge.className = "risk-badge moderate";
    badge.textContent = "MODERATE RISK";
    meterFill.style.backgroundColor = "#f59e0b";
  } else {
    badge.className = "risk-badge low";
    badge.textContent = "LOW DEFAULT RISK";
    meterFill.style.backgroundColor = "#22c55e";
  }

  renderShapDrivers(data.shap_impacts);
}

async function runAssessment(e) {
  if (e) e.preventDefault();
  errorBanner.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Running Underwriting Model...";

  const payload = {
    person_age: parseInt(document.getElementById("person_age").value, 10),
    person_income: parseFloat(document.getElementById("person_income").value),
    person_home_ownership: document.getElementById("person_home_ownership").value,
    person_emp_length: parseFloat(document.getElementById("person_emp_length").value),
    loan_intent: document.getElementById("loan_intent").value,
    loan_grade: document.getElementById("loan_grade").value,
    loan_amnt: parseFloat(document.getElementById("loan_amnt").value),
    loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
    loan_percent_income: parseFloat(document.getElementById("loan_percent_income").value),
    cb_person_default_on_file: document.getElementById("cb_person_default_on_file").value,
    cb_person_cred_hist_length: parseInt(document.getElementById("cb_person_cred_hist_length").value, 10),
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    updateDashboard(result);
  } catch (err) {
    errorBanner.textContent = `Underwriting API Error: ${err.message}`;
    errorBanner.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Credit Risk";
  }
}

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = PRESETS[btn.dataset.preset];
    if (!preset) return;
    Object.entries(preset).forEach(([key, val]) => {
      const el = document.getElementById(key);
      if (el) el.value = val;
    });
    runAssessment();
  });
});

form.addEventListener("submit", runAssessment);

window.addEventListener("DOMContentLoaded", async () => {
  await checkApiHealth();
  await runAssessment();
});