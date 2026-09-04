let shapChartInstance = null;
let currentPredictionData = null;

async function handleAssessmentSubmit(event) {
  event.preventDefault();
  
  const predictBtn = document.getElementById('predictBtn');
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Patient Vitals...';

  const patientData = {
    name: document.getElementById('patientName').value || 'Patient',
    age: parseInt(document.getElementById('age').value),
    bmi: parseFloat(document.getElementById('bmi').value),
    resting_hr: parseInt(document.getElementById('restingHr').value),
    salt_intake: parseInt(document.getElementById('saltIntake').value),
    physical_activity: parseFloat(document.getElementById('physicalActivity').value),
    stress_score: parseInt(document.getElementById('stressScore').value),
    smoking: parseInt(document.getElementById('smoking').value),
    alcohol: parseInt(document.getElementById('alcohol').value),
    family_history: parseInt(document.getElementById('familyHistory').value)
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
      
      // Reveal Cards
      document.getElementById('shapCard').classList.remove('hidden');
      document.getElementById('simCard').classList.remove('hidden');
      document.getElementById('dietCard').classList.remove('hidden');
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

// Auto-trigger initial prediction on page load
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('assessmentForm').dispatchEvent(new Event('submit'));
});
