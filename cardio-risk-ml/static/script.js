let chol = null;
let act  = null;


const SLIDERS = [
  { id: 'age', out: 'age-val', fmt: v => v + ' yrs' },
  { id: 'bmi', out: 'bmi-val', fmt: v => parseFloat(v).toFixed(1) },
];

SLIDERS.forEach(({ id, out, fmt }) => {
  const el  = document.getElementById(id);
  const lbl = document.getElementById(out);
  if (!el || !lbl) return;
  const refresh = () => { lbl.textContent = fmt(el.value); paintSlider(el); };
  el.addEventListener('input', refresh);
  refresh();
});

function paintSlider(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.background =
    `linear-gradient(to right, var(--primary) 0%, var(--primary) ${pct}%, var(--border) ${pct}%, var(--border) 100%)`;
}


function clearGroup(id) {
  document.querySelectorAll(`#${id} .pill`)
    .forEach(b => b.classList.remove('active', 'sedentary'));
}

function setChol(btn, val) {
  clearGroup('chol-group');
  btn.classList.add('active');
  chol = val;
}

function setAct(btn, val) {
  clearGroup('act-group');
  btn.classList.add('active');
  if (val === 0) btn.classList.add('sedentary');
  act = val;
}

function animateGauge(pct) {
  const circle = document.getElementById('gauge-fill');
  const R = 76;
  const C = 2 * Math.PI * R;

  // Colour by risk tier
  const colour =
    pct < 30 ? '#00b896' :
    pct < 50 ? '#3b82f6' :
    pct < 70 ? '#f59e0b' :
               '#ef4444';

  circle.setAttribute('stroke', colour);
  circle.style.strokeDasharray  = C;
  circle.style.strokeDashoffset = C * (1 - pct / 100);

  // Animate number counter
  const el = document.getElementById('gauge-pct');
  let cur  = 0;
  const target = Math.round(pct);
  const step   = target / 45;
  const timer  = setInterval(() => {
    cur = Math.min(cur + step, target);
    el.textContent = Math.round(cur) + '%';
    if (cur >= target) clearInterval(timer);
  }, 18);
}


// ERROR DISPLAY

function showErr(msg) {
  const el = document.getElementById('err-msg');
  el.textContent = '⚠  ' + msg;
  el.classList.add('show');
}

function clearErr() {
  const el = document.getElementById('err-msg');
  el.textContent = '';
  el.classList.remove('show');
}


function riskClass(tier) {
  const map = { Low: 'risk-low', Moderate: 'risk-moderate', High: 'risk-high', Critical: 'risk-critical' };
  return map[tier] || 'risk-moderate';
}

function lvlClass(level) {
  const map = { Low: 'lvl-low', Moderate: 'lvl-moderate', High: 'lvl-high', Critical: 'lvl-critical' };
  return map[level] || 'lvl-moderate';
}

function barColour(level) {
  const map = { Low: '#00b896', Moderate: '#3b82f6', High: '#f59e0b', Critical: '#ef4444' };
  return map[level] || '#3b82f6';
}

async function predict() {
  clearErr();

  const age       = document.getElementById('age').value;
  const bmi       = document.getElementById('bmi').value;
  const systolic  = document.getElementById('systolic').value;
  const diastolic = document.getElementById('diastolic').value;

  // Client-side validation
  if (!systolic || !diastolic) {
    showErr('Please enter both systolic and diastolic blood pressure values.');
    return;
  }
  if (parseInt(systolic) <= parseInt(diastolic)) {
    showErr('Systolic BP must be greater than diastolic BP.');
    return;
  }
  if (chol === null) {
    showErr('Please select a cholesterol level.');
    return;
  }
  if (act === null) {
    showErr('Please select your physical activity level.');
    return;
  }

  // Loading state
  const btn = document.getElementById('predict-btn');
  btn.textContent = 'Analyzing…';
  btn.disabled    = true;

  try {
    const res = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ age, bmi, systolic, diastolic, cholesterol: chol, active: act })
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      showErr(data.error || 'Server error. Please try again.');
      return;
    }

    renderResults(data);

  } catch (err) {
    showErr('Connection error. Make sure the Flask server is running.');
  } finally {
    btn.textContent = 'Run Analysis';
    btn.disabled    = false;
  }
}


function renderResults(data) {
  const pct  = Math.round(data.probability * 100);
  const tier = data.risk_tier;

  // Hide empty state, show result panel
  document.getElementById('result-empty').style.display = 'none';
  const rc = document.getElementById('result-content');
  rc.classList.add('show');

  // Reset and animate gauge
  const circle = document.getElementById('gauge-fill');
  const R = 76;
  const C = 2 * Math.PI * R;
  circle.style.transition       = 'none';
  circle.style.strokeDasharray  = C;
  circle.style.strokeDashoffset = C;
  requestAnimationFrame(() => {
    circle.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1), stroke 0.4s';
    setTimeout(() => animateGauge(pct), 30);
  });

  // Risk badge
  const badge = document.getElementById('risk-badge');
  badge.className = 'risk-badge ' + riskClass(tier);
  document.getElementById('risk-badge-text').textContent = tier + ' Risk';

  // Quick stats
  setVal('qs-pulse',   (data.pulse_pressure          != null ? data.pulse_pressure          + ' mmHg' : '—'));
  setVal('qs-map',     (data.mean_arterial_pressure   != null ? data.mean_arterial_pressure  + ' mmHg' : '—'));
  setVal('qs-bmi-cat', data.bmi_category || '—');
  setVal('qs-bp-cat',  data.bp_category  || '—');

  // Dynamic sections
  renderFactors(data.factors || []);
  renderRecommendations(data.recommendations || []);
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}


function renderFactors(factors) {
  const grid = document.getElementById('factors-grid');
  const section = document.getElementById('factors-section');

  // Clear previous content
  grid.innerHTML = '';

  if (!factors.length) {
    section.style.display = 'none';
    return;
  }

  factors.forEach(function(f, i) {
    const card = document.createElement('div');
    card.className = 'fc';
    card.style.animationDelay = (i * 0.06) + 's';

    card.innerHTML =
      '<div class="fc-head">' +
        '<span class="fc-name">' + escHtml(f.name) + '</span>' +
        '<span class="level-badge ' + lvlClass(f.level) + '">' + escHtml(f.level) + '</span>' +
      '</div>' +
      '<div class="fc-bar-bg">' +
        '<div class="fc-bar" id="fcbar-' + i + '" style="background:' + barColour(f.level) + '"></div>' +
      '</div>' +
      '<div class="fc-advice">' + escHtml(f.advice) + '</div>';

    grid.appendChild(card);

    // Animate bar after short delay
    setTimeout(function() {
      var bar = document.getElementById('fcbar-' + i);
      if (bar) bar.style.width = Math.min(100, Math.max(0, f.score)) + '%';
    }, 200 + i * 70);
  });

  section.style.display = 'block';
}

function renderRecommendations(recos) {
  const grid    = document.getElementById('reco-grid');
  const section = document.getElementById('reco-section');

  // Clear previous content
  grid.innerHTML = '';

  if (!recos.length) {
    section.style.display = 'none';
    return;
  }

  recos.forEach(function(r, i) {
    const card = document.createElement('div');
    card.className = 'rc';
    card.style.animationDelay = (i * 0.08) + 's';


    var iconHtml = r.icon
      ? '<div class="rc-icon">' + escHtml(r.icon) + '</div>'
      : '';

    card.innerHTML =
      iconHtml +
      '<div class="rc-title">' + escHtml(r.title) + '</div>' +
      '<div class="rc-text">'  + escHtml(r.text)  + '</div>';

    grid.appendChild(card);
  });

  section.style.display = 'block';
}


function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}