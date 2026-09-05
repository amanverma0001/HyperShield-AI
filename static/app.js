let shapChartInstance = null;
let currentPredictionData = null;

const samplesNLP = {
  1: "48 year old male, BMI 29.5, heart rate 82 bpm, high salt diet with extra sodium, heavy work stress, smokes cigarettes daily, father had high BP, reports severe morning headaches and dizziness.",
  2: "38 year old female, BMI 25.0, heart rate 72 bpm, moderate salt diet, moderate stress, walks daily, smokes cigarettes, light alcohol, no family history of hypertension.",
  3: "24 year old male, BMI 21.5, heart rate 68 bpm, low sodium diet, relaxed low stress, exercise 5 hours weekly, non-smoker, no family history, feels great."
};

function loadSampleNLP(type) {
  const nlpText = document.getElementById('nlpText');
  if (samplesNLP[type]) {
    nlpText.value = samplesNLP[type];
    runNLPParser();
  }
}

async function runNLPParser() {
  const nlpText = document.getElementById('nlpText').value.trim();
  if (!nlpText || nlpText.length < 5) {
    alert('Please enter a clinical note or patient complaint paragraph to extract NLP features.');
    return;
  }

  const nlpBtn = document.getElementById('nlpBtn');
  nlpBtn.disabled = true;
  nlpBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Parsing Text Entities & Auto-Filling Form...';

  try {
    const res = await fetch('/api/nlp-parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: nlpText })
    });

    const data = await res.json();
    if (data.status === 'success') {
      const ext = data.extracted_parameters;
      
      if (ext.age !== null) document.getElementById('age').value = ext.age;
      if (ext.bmi !== null) document.getElementById('bmi').value = ext.bmi;
      if (ext.resting_hr !== null) document.getElementById('restingHr').value = ext.resting_hr;
      if (ext.salt_intake !== null) document.getElementById('saltIntake').value = ext.salt_intake;
      if (ext.physical_activity !== null) document.getElementById('physicalActivity').value = ext.physical_activity;
      if (ext.stress_score !== null) document.getElementById('stressScore').value = ext.stress_score;
      if (ext.smoking !== null) document.getElementById('smoking').value = ext.smoking;
      if (ext.alcohol !== null) document.getElementById('alcohol').value = ext.alcohol;
      if (ext.family_history !== null) document.getElementById('familyHistory').value = ext.family_history;

      displayNLPTags(ext.entity_tags);
      document.getElementById('assessmentForm').dispatchEvent(new Event('submit'));
    }
  } catch (err) {
    console.error('NLP Parse Error:', err);
    alert('Failed to parse text via NLP server.');
  } finally {
    nlpBtn.disabled = false;
    nlpBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> ⚡ Extract Vitals & Auto-Fill Form (NLP)';
  }
}

function displayNLPTags(tags) {
  const nlpTagsList = document.getElementById('nlpTagsList');
  const nlpTagCount = document.getElementById('nlpTagCount');

  if (!tags || tags.length === 0) {
    nlpTagsList.innerHTML = '<span style="font-size:12px; color:#94A3B8; font-style:italic;">No text entities extracted yet.</span>';
    nlpTagCount.innerText = '0 Entities';
    return;
  }

  nlpTagCount.innerText = `${tags.length} Entities Extracted`;
  let tagsHtml = '';
  tags.forEach(tag => {
    tagsHtml += `
      <span class="nlp-tag-item" style="background-color: ${tag.color};">
        <i class="fa-solid fa-check"></i> [${tag.category}] ${tag.text}
      </span>
    `;
  });

  nlpTagsList.innerHTML = tagsHtml;
}

async function handleAssessmentSubmit(event) {
  event.preventDefault();
  
  const predictBtn = document.getElementById('predictBtn');
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Patient Vitals...';

  const patientData = {
    name: document.getElementById('patientName').value || 'Patient',
    age: parseInt(document.getElementById('age').value) || 30,
    bmi: parseFloat(document.getElementById('bmi').value) || 22.0,
    resting_hr: parseInt(document.getElementById('restingHr').value) || 72,
    salt_intake: parseInt(document.getElementById('saltIntake').value) || 2,
    physical_activity: parseFloat(document.getElementById('physicalActivity').value) || 3.0,
    stress_score: parseInt(document.getElementById('stressScore').value) || 4,
    smoking: parseInt(document.getElementById('smoking').value) || 0,
    alcohol: parseInt(document.getElementById('alcohol').value) || 0,
    family_history: parseInt(document.getElementById('familyHistory').value) || 0
  };

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patientData)
    });

    const result = await response.json();
    if (result.status === 'success') {
      currentPredictionData = result;
      displayPredictionResult(result);
      renderShapChart(result.risk_breakdown);
      setupSimulatorDefaults(result.inputs);
      await fetchDashDiet(result.predicted_stage);
    }
  } catch (error) {
    alert('Error connecting to prediction server. Ensure app.py is running.');
    console.error(error);
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Predict Risk Stage & Analyze';
  }
}

function displayPredictionResult(data) {
  const resultBody = document.getElementById('resultBody');
  const riskBadge = document.getElementById('riskBadge');
  const info = data.stage_info;

  riskBadge.className = 'badge';
  riskBadge.style.backgroundColor = info.color;
  riskBadge.style.color = '#FFFFFF';
  riskBadge.innerText = info.badge;

  const probs = data.probabilities;
  const stageNames = ['Normal', 'Pre-Hypertension', 'Stage 1', 'Stage 2'];
  const stageColors = ['#10B981', '#F59E0B', '#F97316', '#EF4444'];

  let probBarsHtml = '';
  stageNames.forEach((name, idx) => {
    probBarsHtml += `
      <div class="prob-bar-group">
        <div class="prob-label-row">
          <span>${name}</span>
          <span>${probs[idx]}%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" style="width: ${probs[idx]}%; background-color: ${stageColors[idx]};"></div>
        </div>
      </div>
    `;
  });

  resultBody.innerHTML = `
    <div class="result-main" style="border-left-color: ${info.color}">
      <div>
        <div style="font-size:12px; color:#64748B; font-weight:600;">ASSESSED RISK STAGE</div>
        <div class="result-stage-name" style="color: ${info.color}">${info.name}</div>
        <div style="font-size:12px; color:#475569; margin-top:4px;">${info.description}</div>
      </div>
      <div class="result-bp-box">
        <div class="bp-value">${data.estimated_bp}</div>
        <div class="bp-label">Est. Blood Pressure</div>
      </div>
    </div>

    <div class="prob-section">
      <div class="prob-title">CLINICAL STAGE PROBABILITY BREAKDOWN</div>
      ${probBarsHtml}
    </div>

    <div class="actions-row">
      <button onclick="downloadClinicalPDF()" class="btn btn-primary btn-block">
        <i class="fa-solid fa-file-pdf"></i> Download 1-Click PDF Report
      </button>
    </div>
  `;
}

function renderShapChart(breakdown) {
  const ctx = document.getElementById('shapChart').getContext('2d');
  
  const labels = Object.keys(breakdown);
  const values = Object.values(breakdown);

  if (shapChartInstance) {
    shapChartInstance.destroy();
  }

  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Risk Contribution %',
        data: values,
        backgroundColor: [
          '#4F46E5', '#0EA5E9', '#F59E0B', '#EF4444', '#10B981', '#8B5CF6', '#EC4899', '#6366F1', '#14B8A6'
        ],
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` Contribution: ${context.raw}%`
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (val) => val + '%' }
        }
      }
    }
  });
}

function setupSimulatorDefaults(inputs) {
  const saltSlider = document.getElementById('simSaltSlider');
  const actSlider = document.getElementById('simActSlider');

  saltSlider.value = Math.max(1, inputs.salt_intake - 1);
  actSlider.value = Math.min(14, inputs.physical_activity + 3.0);

  updateSimulation();
}

async function updateSimulation() {
  if (!currentPredictionData) return;

  const saltVal = parseInt(document.getElementById('simSaltSlider').value);
  const actVal = parseFloat(document.getElementById('simActSlider').value);

  const saltLabels = ['Low (< 3g/day)', 'Normal (3-6g/day)', 'High (6-9g/day)', 'Severe (> 9g/day)'];
  document.getElementById('targetSaltVal').innerText = saltLabels[saltVal - 1];
  document.getElementById('targetActVal').innerText = `${actVal} Hrs/Wk`;

  const simPayload = {
    patient: {
      name: currentPredictionData.patient_name,
      ...currentPredictionData.inputs
    },
    target_salt: saltVal,
    target_activity: actVal
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(simPayload)
    });

    const data = await res.json();
    if (data.status === 'success') {
      const simResults = document.getElementById('simResults');
      const info = data.simulated_stage_info;

      simResults.innerHTML = `
        <div class="sim-results-box">
          <div>
            <div style="font-size:11px; color:#047857; font-weight:700;">SIMULATED OUTCOME</div>
            <div style="font-size:18px; font-weight:800; color: ${info.color};">${info.name} (${data.simulated_bp})</div>
            <div style="font-size:12px; color:#475569;">Target Sodium & Activity Adjustments Applied</div>
          </div>
          <div style="text-align:right;">
            <div class="sim-reduction-val">-${data.risk_reduction_pct}%</div>
            <div style="font-size:10px; color:#047857; font-weight:700;">RISK REDUCTION</div>
          </div>
        </div>
      `;
    }
  } catch (err) {
    console.error('Simulation error:', err);
  }
}

async function fetchDashDiet(stage) {
  try {
    const res = await fetch('/api/diet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage: stage })
    });

    const diet = await res.json();
    document.getElementById('sodiumLimitBadge').innerText = `Sodium Limit: ${diet.daily_sodium_limit}`;

    const dietContent = document.getElementById('dietContent');
    dietContent.innerHTML = `
      <div style="font-size:13px; font-weight:700; color:#1E293B; margin-bottom:6px;">
        Focus: ${diet.primary_focus}
      </div>

      <div class="diet-grid">
        <div class="diet-item">
          <div class="diet-item-title"><i class="fa-solid fa-mug-hot"></i> Breakfast</div>
          <div class="diet-item-desc">${diet.meals.Breakfast}</div>
        </div>

        <div class="diet-item">
          <div class="diet-item-title"><i class="fa-solid fa-bowl-rice"></i> Lunch</div>
          <div class="diet-item-desc">${diet.meals.Lunch}</div>
        </div>

        <div class="diet-item">
          <div class="diet-item-title"><i class="fa-solid fa-apple-whole"></i> Healthy Snack</div>
          <div class="diet-item-desc">${diet.meals.Snack}</div>
        </div>

        <div class="diet-item">
          <div class="diet-item-title"><i class="fa-solid fa-fish"></i> Dinner</div>
          <div class="diet-item-desc">${diet.meals.Dinner}</div>
        </div>
      </div>
    `;
  } catch (err) {
    console.error('Diet fetch error:', err);
  }
}

function downloadClinicalPDF() {
  if (!currentPredictionData) return;
  window.print();
}

// Initial render of empty SHAP chart on page load
window.addEventListener('DOMContentLoaded', () => {
  const initialBreakdown = {
    'Daily Salt / Sodium Intake': 0,
    'Daily Stress Level': 0,
    'Physical Activity Deficit': 0,
    'Smoking Habit': 0,
    'Genetics / Family History': 0,
    'Body Mass Index (BMI)': 0,
    'Alcohol Consumption': 0,
    'Resting Heart Rate': 0,
    'Age Factor': 0
  };
  renderShapChart(initialBreakdown);
});
