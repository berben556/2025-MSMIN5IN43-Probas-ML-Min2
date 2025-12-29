/*********************************************************
 * ESPÈCES
 *********************************************************/
const Species = Object.freeze({
  HAWK: "HAWK",
  DOVE: "DOVE",
  GRUDGE: "GRUDGE",
  DETECTIVE: "DETECTIVE"
});

/*********************************************************
 * CANVAS / DOM
 *********************************************************/
const board = document.getElementById("board");
const ctx = board.getContext("2d");
const chartCanvas = document.getElementById("chart");

const hawksSlider = document.getElementById("hawksSlider");
const dovesSlider = document.getElementById("dovesSlider");
const grudgeSlider = document.getElementById("grudgeSlider");
const detectiveSlider = document.getElementById("detectiveSlider");
const hawksVal = document.getElementById("hawksVal");
const dovesVal = document.getElementById("dovesVal");
const grudgeVal = document.getElementById("grudgeVal");
const detectiveVal = document.getElementById("detectiveVal");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

/*********************************************************
 * CHART
 *********************************************************/
const chart = new Chart(chartCanvas, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      { label: "Hawks", borderColor: "red", data: [] },
      { label: "Doves", borderColor: "blue", data: [] },
      { label: "Grudges", borderColor: "gold", data: [] },
      { label: "Detectives", borderColor: "purple", data: [] }
    ]
  },
  options: {
    responsive: false,
    plugins: {
      legend: { position: "top" }
    },
    scales: {
      x: { title: { display: true, text: "Day" } },
      y: { title: { display: true, text: "Population" } }
    }
  }
});

/*********************************************************
 * ÉTAT GLOBAL
 *********************************************************/
const center = { x: 300, y: 300 };
const perimeterRadius = 280;

let creatures = [];
let foods = [];
let day = 0;
let loop = null;
let nextId = 0;
let animating = false;

/*********************************************************
 * UTILITAIRE
 *********************************************************/
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function count(species) {
  return creatures.filter(c => c.species === species).length;
}

/*********************************************************
 * CRÉATURE
 *********************************************************/
function createCreature(species, angle) {
  const x = center.x + perimeterRadius * Math.cos(angle);
  const y = center.y + perimeterRadius * Math.sin(angle);
  return {
    id: nextId++,
    species,
    angle,
    x, y,
    startX: x,
    startY: y,
    target: null,
    food: 0,
    hawksMemory:
        species === Species.GRUDGE || species === Species.DETECTIVE
            ? new Set()
            : null,
    round: species === Species.DETECTIVE ? 0 : null,
    someoneCheated: species === Species.DETECTIVE ? false : null
  };
}

/*********************************************************
 * NOURRITURE — 3 CERCLES / 4–6–8 PAIRES
 *********************************************************/
function generateFoodPairs() {
  foods = [];
  const config = [
    { r: 100, pairs: 4 },
    { r: 160, pairs: 6 },
    { r: 220, pairs: 8 }
  ];
  const offset = 10; // décalage perpendiculaire pour 2 boules

  config.forEach(({ r, pairs }) => {
    for (let i = 0; i < pairs; i++) {
      const a = (2 * Math.PI * i) / pairs;
      const cx = center.x + r * Math.cos(a);
      const cy = center.y + r * Math.sin(a);

      // vecteur perpendiculaire au rayon
      const dirX = cx - center.x;
      const dirY = cy - center.y;
      const dist = Math.hypot(dirX, dirY);
      const perpX = -dirY / dist;
      const perpY = dirX / dist;

      foods.push({
        assigned: [],
        eaten: false,
        r,
        cx,
        cy,
        items: [
          { x: cx + offset * perpX, y: cy + offset * perpY, eaten: false },
          { x: cx - offset * perpX, y: cy - offset * perpY, eaten: false }
        ]
      });
    }
  });

  const max = foods.length * 2;
  hawksSlider.max = dovesSlider.max = grudgeSlider.max = detectiveSlider.max = max;
}

/*********************************************************
 * INITIALISATION
 *********************************************************/
function initCreatures(h, d, g, det) {
  creatures = [];
  nextId = 0;
  day = 0;

  generateFoodPairs();

  const total = h + d + g + det;
  for (let i = 0; i < total; i++) {
    const angle = (2 * Math.PI * i) / total;
    let s =
        i < h ? Species.HAWK :
            i < h + d ? Species.DOVE :
                i < h + d + g ? Species.GRUDGE :
                    Species.DETECTIVE;

    creatures.push(createCreature(s, angle));
  }

  resetChart();
  draw();
}


/*********************************************************
 * DESSIN
 *********************************************************/
function draw() {
  ctx.clearRect(0, 0, board.width, board.height);

  // Cercle
  ctx.beginPath();
  ctx.arc(center.x, center.y, perimeterRadius, 0, 2 * Math.PI);
  ctx.stroke();

  // Nourriture
  foods.forEach(p => {
    if (p.eaten) return;
    p.items.forEach(f => {
      if (!f.eaten) {
        ctx.beginPath();
        ctx.arc(f.x, f.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = "green";
        ctx.fill();
      }
    });
  });

  // Créatures
  creatures.forEach(c => {
    ctx.beginPath();
    ctx.arc(c.x, c.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle =
        c.species === Species.HAWK ? "red" :
            c.species === Species.DOVE ? "blue" :
                c.species === Species.GRUDGE ? "gold" : "purple";
    ctx.fill();
  });
}

/*********************************************************
 * ASSIGNATION AUX PAIRES (sur le rayon centre → paire)
 *********************************************************/
function assignTargets() {
  foods.forEach(p => p.assigned = []);
  const shuffled = [...creatures].sort(() => Math.random() - 0.5);

  shuffled.forEach(c => {
    const free = foods.filter(p => p.assigned.length < 2 && !p.eaten);
    if (!free.length) return;

    const p = free[Math.floor(Math.random() * free.length)];
    p.assigned.push(c);

    // vecteur radial centre → centre de la paire
    const dx = p.cx - center.x;
    const dy = p.cy - center.y;
    const dist = Math.hypot(dx, dy);
    const dirX = dx / dist;
    const dirY = dy / dist;

    // position sur le rayon + sideOffset
    const radialOffset = p.assigned.length === 1 ? -16 : 16; // première devant, deuxième derrière
    const tx = center.x + dirX * (p.r + radialOffset);
    const ty = center.y + dirY * (p.r + radialOffset);

    c.target = { x: tx, y: ty, pair: p };
  });
}

/*********************************************************
 * DÉPLACEMENT ANIMÉ
 *********************************************************/
function animateMove(speed = 4) {
  return new Promise(resolve => {
    function step() {
      let done = true;
      creatures.forEach(c => {
        if (!c.target) return;
        const dx = c.target.x - c.x;
        const dy = c.target.y - c.y;
        const d = Math.hypot(dx, dy);
        if (d > speed) {
          c.x += dx / d * speed;
          c.y += dy / d * speed;
          done = false;
        }
      });
      draw();
      done ? resolve() : requestAnimationFrame(step);
    }
    step();
  });
}

/*********************************************************
 * COMPORTEMENT ALIMENTAIRE
 *********************************************************/
function behavesAsHawk(c, other) {
  if (c.species === Species.HAWK) return true;
  if (c.species === Species.DOVE) return false;
  if (c.species === Species.DETECTIVE) {
    if (c.round < 4) {
      return [false, true, false, false][c.round];
    }
    if (!c.someoneCheated) return true; // exploitation
    return other && c.hawksMemory?.has(other.id);
  }
  // Pour une Grudge, vérifier que le Hawk est vivant avant de considérer qu'il se comporte comme un Hawk
  if (c.species === Species.GRUDGE && other) {
    // Filtrer la mémoire pour ne garder que les Hawks encore présents
    c.hawksMemory = new Set([...c.hawksMemory].filter(id => creatures.some(h => h.id === id)));
    return c.hawksMemory.has(other.id);
  }

  return false;
}

function eatPair(pair) {
  if (pair.eaten) return;
  const assigned = pair.assigned;
  if (assigned.length === 0) return;

  if (assigned.length === 1) {
    assigned[0].food += 2;
  } else if (assigned.length === 2) {
    const [a, b] = assigned;
    const ha = behavesAsHawk(a, b);
    const hb = behavesAsHawk(b, a);

    // Detective observe
    if (a.species === Species.DETECTIVE && hb) {
      a.someoneCheated = true;
      a.hawksMemory ??= new Set();
      a.hawksMemory.add(b.id);
    }
    if (b.species === Species.DETECTIVE && ha) {
      b.someoneCheated = true;
      b.hawksMemory ??= new Set();
      b.hawksMemory.add(a.id);
    }

    //repartition de la nourriture
    if (ha && hb) {
      // Hawk vs Hawk ou Grudge déjà rencontré → 0:0
      a.food = 0;
      b.food = 0;
    } else if (ha && !hb) {
      // a = Hawk, b = Dove/Grudge
      a.food = 1.5;
      b.food = 0.5;
      if (b.species === Species.GRUDGE && !b.hawksMemory.has(a.id)) b.hawksMemory.add(a.id);
    } else if (!ha && hb) {
      // b = Hawk, a = Dove/Grudge
      a.food = 0.5;
      b.food = 1.5;
      if (a.species === Species.GRUDGE && !a.hawksMemory.has(b.id)) a.hawksMemory.add(b.id);
    } else {
      // Dove vs Dove ou Grudge vs Dove → 1:1
      a.food = 1;
      b.food = 1;
    }
  }

  // Marquer les items comme mangés
  pair.items.forEach(f => f.eaten = true);
  pair.eaten = true;
}

/*********************************************************
 * FIN DE JOURNÉE — GLISSEMENT SUR LE CERCLE
 *********************************************************/
async function animateReposition(next) {
  const step = (2 * Math.PI) / next.length;
  next.forEach((c, i) => c.targetAngle = i * step);

  return new Promise(resolve => {
    function stepAnim() {
      let done = true;
      next.forEach(c => {
        let diff = c.targetAngle - c.angle;
        if (Math.abs(diff) > 0.01) {
          c.angle += diff * 0.15;
          done = false;
        }
        c.x = center.x + perimeterRadius * Math.cos(c.angle);
        c.y = center.y + perimeterRadius * Math.sin(c.angle);
      });
      draw();
      if (done) {
        // Mettre à jour la position de départ après réorganisation finale
        next.forEach(c => {
          c.startX = c.x;
          c.startY = c.y;
        });
        resolve();
      } else requestAnimationFrame(stepAnim);
    }
    stepAnim();
  });
}

/*********************************************************
 * MORT & REPRODUCTION
 *********************************************************/
function applyMortalityAndReproduction() {
  const next = [];
  creatures.forEach(c => {
    let survives = false;
    let reproduce = false;

    //0=mort, 0.5=50% de chance de survie, 1=survie, 1.5=survie+50% de chance de repro, 2=repro
    if (c.food === 0) survives = false;
    else if (c.food === 0.5) survives = Math.random() < 0.5;
    else if (c.food === 1) survives = true;
    else if (c.food === 1.5) { survives = true; reproduce = Math.random() < 0.5; }
    else if (c.food >= 2) { survives = true; reproduce = true; }

    if (survives) next.push(c);
    if (reproduce) next.push(createCreature(c.species, c.angle));
  });
  return next;
}

/*********************************************************
 * STEP DAY COMPLET AVEC PAUSE SUR NOURRITURE
 *********************************************************/
async function stepDay() {
  if (animating) return;
  animating = true;
  day++;

  // Remise à zéro nourriture et régénération
  creatures.forEach(c => c.food = 0);
  generateFoodPairs();
  assignTargets();

  // Animation vers nourriture
  await animateMove();

  // Pause 1 seconde devant chaque paire
  await wait(1000);

  // Manger
  foods.forEach(eatPair);

  // Retour à leur position initiale
  creatures.forEach(c => c.target = { x: c.startX, y: c.startY });
  await animateMove();

  // Mort & reproduction
  creatures = applyMortalityAndReproduction();

  // Réorganisation finale
  await animateReposition(creatures);

  creatures.forEach(c => {
    if (c.species === Species.DETECTIVE) c.round++;
  });
  // Mise à jour du graphe
  updateChart();

  // **Arrêter la simulation si plus aucune créature**
  if (creatures.length === 0) {
    clearInterval(loop);
    loop = null;

    animating = false;
    return;
  }

  animating = false;
}

/*********************************************************
 * CHART
 *********************************************************/
function resetChart() {
  chart.data.labels = [0];
  chart.data.datasets[0].data = [count(Species.HAWK)];
  chart.data.datasets[1].data = [count(Species.DOVE)];
  chart.data.datasets[2].data = [count(Species.GRUDGE)];
  chart.data.datasets[3].data = [count(Species.DETECTIVE)];
  chart.update();
}

function updateChart() {
  chart.data.labels.push(day);
  chart.data.datasets[0].data.push(count(Species.HAWK));
  chart.data.datasets[1].data.push(count(Species.DOVE));
  chart.data.datasets[2].data.push(count(Species.GRUDGE));
  chart.data.datasets[3].data.push(count(Species.DETECTIVE));
  chart.update();
}

/*********************************************************
 * SLIDERS & BOUTONS
 *********************************************************/
function syncSliders() {
  hawksVal.textContent = hawksSlider.value.toString().padStart(2, '0');
  dovesVal.textContent = dovesSlider.value.toString().padStart(2, '0');
  grudgeVal.textContent = grudgeSlider.value.toString().padStart(2, '0');
  detectiveVal.textContent = detectiveSlider.value.padStart(2,'0');

  initCreatures(+hawksSlider.value, +dovesSlider.value, +grudgeSlider.value, +detectiveSlider.value);
}

hawksSlider.oninput = dovesSlider.oninput = grudgeSlider.oninput = detectiveSlider.oninput = syncSliders;

startBtn.onclick = () => loop ??= setInterval(stepDay, 3000);
stopBtn.onclick = () => { clearInterval(loop); loop = null; };
resetBtn.onclick = () => { stopBtn.onclick(); syncSliders(); };

/*********************************************************
 * INIT
 *********************************************************/
window.onload = syncSliders;