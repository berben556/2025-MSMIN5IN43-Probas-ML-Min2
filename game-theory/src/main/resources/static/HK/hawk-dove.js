const boardCanvas = document.getElementById('board');
const ctx = boardCanvas.getContext('2d');
const chartCanvas = document.getElementById('chart');

const chart = new Chart(chartCanvas, {
  type:'line',
  data:{labels:[], datasets:[
      {label:'Hawks', borderColor:'red', data:[]},
      {label:'Doves', borderColor:'blue', data:[]}
    ]},
  options:{
    responsive:false,
    plugins:{legend:{position:'top'}},
    scales:{
      x:{title:{display:true,text:'Day'}},
      y:{title:{display:true,text:'Population'}}
    }
  }
});

let creatures=[], foods=[], day=0, animating=false, intervalId=null, nextCreatureId=0;
const radius=280, center={x:300,y:300};

const hawksSlider = document.getElementById('hawksSlider');
const dovesSlider = document.getElementById('dovesSlider');
const hawksVal = document.getElementById('hawksVal');
const dovesVal = document.getElementById('dovesVal');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');

hawksSlider.oninput = () => { hawksVal.textContent = hawksSlider.value; resetCreaturesAndChart(); };
dovesSlider.oninput = () => { dovesVal.textContent = dovesSlider.value; resetCreaturesAndChart(); };

// ---------------------------
// INITIALISATION DES CREATURES
// ---------------------------
function initCreatures(hawks,doves){
  creatures=[];
  const total=parseInt(hawks)+parseInt(doves);
  for(let i=0;i<total;i++){
    const angle=(2*Math.PI*i)/total;
    const type=i<hawks?"HAWK":"DOVE";
    creatures.push({
      id: nextCreatureId++,
      type,
      angle,
      x:center.x+radius*Math.cos(angle),
      y:center.y+radius*Math.sin(angle),
      startX:center.x+radius*Math.cos(angle),
      startY:center.y+radius*Math.sin(angle),
      target:null,
      state:"atPerimeter",
      food:0,
      alpha:1
    });
  }
  draw();
}

function resetCreaturesAndChart(){
  day=0;
  initCreatures(hawksSlider.value, dovesSlider.value);
  chart.data.labels=[day];
  chart.data.datasets[0].data=[creatures.filter(c=>c.type==="HAWK").length];
  chart.data.datasets[1].data=[creatures.filter(c=>c.type==="DOVE").length];
  chart.update();
}

// ---------------------------
// DESSIN
// ---------------------------
function draw(){
  ctx.clearRect(0,0,boardCanvas.width,boardCanvas.height);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, 2*Math.PI);
  ctx.stroke();

  foods.forEach(pair => {
    pair.forEach(f => {
      if(!f.eaten){
        ctx.beginPath();
        ctx.arc(f.x,f.y,7,0,2*Math.PI);
        ctx.fillStyle='green';
        ctx.fill();
      }
    });
  });

  creatures.forEach(c => {
    ctx.beginPath();
    ctx.arc(c.x,c.y,8,0,2*Math.PI);
    ctx.fillStyle = c.type==="HAWK" ? 'rgba(255,0,0,'+c.alpha+')' : 'rgba(0,0,255,'+c.alpha+')';
    ctx.fill();
  });
}

// ---------------------------
// MOUVEMENT ET ANIMATION
// ---------------------------
function moveTowards(obj,target,speed){
  const dx=target.x-obj.x, dy=target.y-obj.y;
  const dist=Math.sqrt(dx*dx+dy*dy);
  if(dist<speed){ obj.x=target.x; obj.y=target.y; return true;}
  obj.x+=dx/dist*speed; obj.y+=dy/dist*speed; return false;
}

// ---------------------------
// GESTION JOUR
// ---------------------------
async function stepDay(){ /* ton code complet de stepDay() */ }
function updateChart(){ /* code chart update */ }
function generateFoodPairsStructured(){ /* code nourriture */ }
function assignTargetsToFoodStructured(){ /* code assign */ }
function eatFood(creature){ /* code eat */ }
function applyMortReproduction(){ /* code reproduction */ }
async function animateStep(state){ /* code animation */ }
async function animateEndOfDay(newCreatures){ /* code fin de jour */ }

// ---------------------------
// BOUTONS
// ---------------------------
startBtn.onclick = ()=>{ if(!intervalId) intervalId=setInterval(()=>stepDay(),3000); };
stopBtn.onclick = ()=>{ if(intervalId){ clearInterval(intervalId); intervalId=null; } };
resetBtn.onclick = ()=>{ if(intervalId){ clearInterval(intervalId); intervalId=null; } resetCreaturesAndChart(); };

// ---------------------------
// INIT
// ---------------------------
window.onload = ()=> resetCreaturesAndChart();
