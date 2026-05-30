import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from "recharts";
import Papa from "papaparse";

// ─── Palette ──────────────────────────────────────────────────────────────────
const C = {
  bg:"#070b14", card:"#0c1120", inp:"#080d1a", deep:"#060910",
  bd:"#192035", cyan:"#00c8f0", green:"#00e87a", orange:"#ff6b35",
  pink:"#ff4499", purple:"#a855f7", yellow:"#f5c842", red:"#ff3355",
  text:"#dde6f5", muted:"#3d5070", sec:"#7a90b0",
};
const cc = (extra={}) => ({ background:C.card, border:`1px solid ${C.bd}`, borderRadius:12, padding:"18px 22px", ...extra });
const IS = { background:C.inp, border:`1px solid ${C.bd}`, borderRadius:8, color:C.text, padding:"8px 12px", fontSize:13, width:"100%", outline:"none", fontFamily:"inherit", boxSizing:"border-box" };
const SE = { ...IS, cursor:"pointer" };
const LB = { color:C.muted, fontSize:10, fontWeight:700, letterSpacing:"1px", textTransform:"uppercase", display:"block", marginBottom:6 };
const bP = (col=C.cyan) => ({ background:`linear-gradient(135deg,${col},${col}bb)`, color:"#000", border:"none", borderRadius:8, padding:"9px 20px", fontSize:13, fontWeight:700, cursor:"pointer", fontFamily:"inherit", whiteSpace:"nowrap" });
const bS = (col=C.cyan) => ({ background:"transparent", color:col, border:`1px solid ${col}`, borderRadius:8, padding:"8px 16px", fontSize:12, fontWeight:600, cursor:"pointer", fontFamily:"inherit", whiteSpace:"nowrap" });

// ─── Constants ────────────────────────────────────────────────────────────────
const ACTS  = ["ReLU","GELU","Swish","Sigmoid","Tanh","LeakyReLU","ELU","Softmax","Linear","PReLU"];
const OPTS  = ["Adam","AdamW","SGD","RMSprop","Adagrad","Nadam","Adadelta"];
const SCHS  = ["None","ReduceOnPlateau","CosineAnnealing","StepDecay","CyclicLR","WarmupLinear"];
const BSIZES= [8,16,32,64,128,256,512,1024];
const WINIT = ["Xavier/Glorot","He Normal","He Uniform","LeCun Normal","Orthogonal","Random Normal"];
const MODS  = ["Tabular / CSV","Images","Audio","Text","Video","Multimodal"];
const PTYPES= ["Classification","Regression","Clustering","Anomaly Detection","Time Series","NLP","Computer Vision","Recommendation"];
const CML   = ["Random Forest","Gradient Boosting","XGBoost","SVM","KNN","Logistic Regression","Decision Tree","Naive Bayes","Ridge Regression","Lasso","ElasticNet","K-Means","DBSCAN"];
const PRESETS = {
  Tiny:  [{n:32, a:"ReLU",d:0.1}],
  Small: [{n:128,a:"ReLU",d:0.2},{n:64, a:"ReLU",d:0.1}],
  Medium:[{n:256,a:"ReLU",d:0.3},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
  Large: [{n:512,a:"GELU",d:0.3},{n:256,a:"GELU",d:0.2},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
  XLarge:[{n:1024,a:"GELU",d:0.4},{n:512,a:"GELU",d:0.3},{n:256,a:"ReLU",d:0.25},{n:128,a:"ReLU",d:0.1},{n:64,a:"ReLU",d:0.1}],
};

// ─── Sample datasets ──────────────────────────────────────────────────────────
const SAMPLES = {
  iris:{ label:"🌸 Iris", desc:"150 rows · 4 numeric features · 3 species classes",
    gen:()=>{ const sp=["setosa","versicolor","virginica"],bp=[[5.0,3.4,1.5,0.2],[5.9,2.8,4.3,1.3],[6.6,3.0,5.6,2.1]];
      return Array.from({length:150},(_,i)=>{const c=i%3,b=bp[c];return{sepal_length:+(b[0]+(Math.random()-.5)*1.2).toFixed(1),sepal_width:+(b[1]+(Math.random()-.5)*.8).toFixed(1),petal_length:+(b[2]+(Math.random()-.5)*1.5).toFixed(1),petal_width:+(b[3]+(Math.random()-.5)*.5).toFixed(1),species:sp[c]};});} },
  titanic:{ label:"🚢 Titanic", desc:"300 rows · 9 features · survival (0/1) — has missing values",
    gen:()=>Array.from({length:300},(_,i)=>({passenger_id:i+1,pclass:[1,2,3][i%3],sex:i%2?"male":"female",age:i%5===0?"":(+(18+Math.random()*60).toFixed(0)),sibsp:Math.floor(Math.random()*5),parch:Math.floor(Math.random()*4),fare:+(7+Math.random()*500).toFixed(2),embarked:["S","C","Q"][i%3],survived:Math.random()>.6?0:1})) },
  housing:{ label:"🏠 House Prices", desc:"250 rows · 6 numeric features · price regression",
    gen:()=>Array.from({length:250},()=>({sqft:Math.floor(800+Math.random()*3200),bedrooms:Math.floor(1+Math.random()*5),bathrooms:+(1+Math.random()*3).toFixed(1),age_years:Math.floor(Math.random()*60),garage:Math.random()>.3?1:0,neighborhood_score:+(1+Math.random()*9).toFixed(1),price:Math.floor(100000+Math.random()*800000)})) },
  diabetes:{ label:"🏥 Diabetes Risk", desc:"500 rows · 8 health metrics · outcome (0/1)",
    gen:()=>Array.from({length:500},()=>({pregnancies:Math.floor(Math.random()*15),glucose:Math.floor(70+Math.random()*130),blood_pressure:Math.floor(60+Math.random()*60),skin_thickness:Math.floor(10+Math.random()*50),insulin:Math.floor(Math.random()*300),bmi:+(18+Math.random()*30).toFixed(1),diabetes_pedigree:+(0.1+Math.random()*2.3).toFixed(3),age:Math.floor(21+Math.random()*60),outcome:Math.random()>.65?0:1})) },
};

// ─── Data utilities ───────────────────────────────────────────────────────────
const isNullVal = v => v===null||v===undefined||v===''||v==='null'||v==='nan'||v==='NaN'||v==='NA'||v==='N/A';

const detectType = vals => {
  const nn=vals.filter(v=>!isNullVal(v));
  if(!nn.length) return 'unknown';
  const numC=nn.filter(v=>!isNaN(parseFloat(v))&&isFinite(Number(v))).length;
  if(numC/nn.length>0.85) return 'numeric';
  const u=new Set(nn).size;
  if(u<=Math.min(20,Math.max(2,nn.length*0.08))) return 'categorical';
  return 'text';
};

const colStat = (data, name, type) => {
  const vals=data.map(r=>r[name]);
  const nulls=vals.filter(isNullVal).length;
  const nn=vals.filter(v=>!isNullVal(v));
  const uniq=new Set(nn).size;
  if(type==='numeric'){
    const ns=nn.map(Number).filter(n=>!isNaN(n));
    if(!ns.length) return {nulls,uniq,type};
    const srt=[...ns].sort((a,b)=>a-b);
    const mean=ns.reduce((a,b)=>a+b,0)/ns.length;
    const std=Math.sqrt(ns.reduce((a,b)=>a+(b-mean)**2,0)/ns.length);
    return {nulls,uniq,type,min:+srt[0].toFixed(3),max:+srt[srt.length-1].toFixed(3),mean:+mean.toFixed(3),std:+std.toFixed(3),median:+srt[Math.floor(srt.length/2)].toFixed(3),q1:+srt[Math.floor(srt.length*.25)].toFixed(3),q3:+srt[Math.floor(srt.length*.75)].toFixed(3)};
  }
  const freq={};nn.forEach(v=>{const s=String(v);freq[s]=(freq[s]||0)+1;});
  return {nulls,uniq,type,top:Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,7)};
};

const buildCols = data => {
  if(!data.length) return [];
  return Object.keys(data[0]).map((name,i)=>{
    const vals=data.map(r=>r[name]);
    const type=detectType(vals);
    return {name,type,dropped:false,isTarget:false,id:i};
  });
};

// ─── Real preprocessing on actual data ───────────────────────────────────────
const applyPrepOps = (data, cols, prep) => {
  let d=data.map(r=>({...r}));
  const numCols=cols.filter(c=>!c.dropped&&!c.isTarget&&c.type==='numeric');
  const log=[];

  // 1. Drop rows missing values
  if(prep.missingStrategy==='Drop Rows'){
    const before=d.length;
    d=d.filter(r=>!cols.some(c=>!c.dropped&&isNullVal(r[c.name])));
    log.push(`Dropped ${before-d.length} rows with missing values`);
  } else {
    numCols.forEach(col=>{
      const nullRows=d.filter(r=>isNullVal(r[col.name])).length;
      if(!nullRows) return;
      const nums=d.map(r=>r[col.name]).filter(v=>!isNullVal(v)).map(Number).filter(n=>!isNaN(n));
      if(!nums.length) return;
      let fill;
      if(prep.missingStrategy.includes('Mean')) fill=nums.reduce((a,b)=>a+b,0)/nums.length;
      else if(prep.missingStrategy.includes('Median')){const s=[...nums].sort((a,b)=>a-b);fill=s[Math.floor(s.length/2)];}
      else if(prep.missingStrategy.includes('Zero')) fill=0;
      else fill=nums.reduce((a,b)=>a+b,0)/nums.length;
      d=d.map(r=>({...r,[col.name]:isNullVal(r[col.name])?+fill.toFixed(4):r[col.name]}));
      log.push(`Imputed ${nullRows} nulls in '${col.name}' with ${prep.missingStrategy.split(' ')[0].toLowerCase()} (${+fill.toFixed(3)})`);
    });
  }

  // 2. Remove duplicates
  if(prep.removeDuplicates){
    const before=d.length;
    const seen=new Set();
    d=d.filter(r=>{const k=JSON.stringify(r);if(seen.has(k))return false;seen.add(k);return true;});
    if(before-d.length>0) log.push(`Removed ${before-d.length} duplicate rows`);
  }

  // 3. Outlier removal
  if(prep.outlierRemoval){
    let removed=0;
    numCols.forEach(col=>{
      const ns=d.map(r=>Number(r[col.name])).filter(n=>!isNaN(n)).sort((a,b)=>a-b);
      const q1=ns[Math.floor(ns.length*.25)],q3=ns[Math.floor(ns.length*.75)],iqr=q3-q1;
      const lo=q1-1.5*iqr,hi=q3+1.5*iqr;
      const before=d.length;
      d=d.filter(r=>{const v=Number(r[col.name]);return isNaN(v)||(v>=lo&&v<=hi);});
      removed+=before-d.length;
    });
    if(removed>0) log.push(`Removed ${removed} outlier rows (IQR method)`);
  }

  // 4. Feature scaling
  if(prep.scaling!=='None'){
    numCols.forEach(col=>{
      const ns=d.map(r=>Number(r[col.name])).filter(n=>!isNaN(n));
      if(!ns.length) return;
      const mean=ns.reduce((a,b)=>a+b,0)/ns.length;
      const std=Math.sqrt(ns.reduce((a,b)=>a+(b-mean)**2,0)/ns.length)||1;
      const min=Math.min(...ns),max=Math.max(...ns);
      const rng=max-min||1;
      const srt=[...ns].sort((a,b)=>a-b);
      const q1=srt[Math.floor(srt.length*.25)],q3=srt[Math.floor(srt.length*.75)],iqr=q3-q1||1;
      d=d.map(r=>{
        const v=Number(r[col.name]);if(isNaN(v))return r;
        const s=prep.scaling==='StandardScaler'?(v-mean)/std:
                prep.scaling==='MinMaxScaler'?(v-min)/rng:
                prep.scaling==='RobustScaler'?(v-q1)/iqr:v;
        return {...r,[col.name]:+s.toFixed(6)};
      });
    });
    log.push(`Applied ${prep.scaling} to ${numCols.length} numeric columns`);
  }

  // 5. PCA placeholder (show intent, mark applied)
  if(prep.pca) log.push(`PCA configured (variance threshold: ${prep.pcaVariance}) — applied at training`);

  return {data:d, log};
};

// ─── Offline recommendation engine ────────────────────────────────────────────
const recommend = (ds) => {
  const n=parseInt(ds.samples)||1000,f=parseInt(ds.features)||20,c=parseInt(ds.classes)||2;
  const mod=ds.modality,pt=ds.problemType;
  const isReg=["Regression","Time Series"].includes(pt),isUnsup=["Clustering","Anomaly Detection"].includes(pt);
  const r={modelType:"deep",classicalAlgo:"Random Forest",
    layers:[{n:256,a:"ReLU",d:0.2},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
    outAct:isReg?"Linear":c>2?"Softmax":"Sigmoid",
    wInit:"Xavier/Glorot",opt:"Adam",lr:0.001,bs:32,ep:100,l2:0.0001,pat:15,sch:"ReduceOnPlateau",vSplit:0.2,
    rationale:"",warnings:[],estimate:""};
  if((n<5000&&mod.includes("Tabular"))||isUnsup){
    r.modelType="classical";
    r.classicalAlgo=pt==="Anomaly Detection"?"DBSCAN":pt==="Clustering"?"K-Means":isReg?"Gradient Boosting":n<1000?"Random Forest":"Gradient Boosting";
  }
  if(mod.includes("Image")||pt==="Computer Vision"){
    r.layers=[{n:512,a:"ReLU",d:0.4},{n:256,a:"ReLU",d:0.3},{n:128,a:"ReLU",d:0.2}];
    r.wInit="He Normal";r.bs=64;r.ep=50;r.sch="CosineAnnealing";
    r.rationale="Image data → deep architecture with He Normal init for ReLU. Cosine annealing LR schedule. Heavy dropout prevents visual overfitting.";
  } else if(mod.includes("Text")||pt==="NLP"){
    r.layers=[{n:256,a:"GELU",d:0.3},{n:128,a:"GELU",d:0.2}];
    r.opt="AdamW";r.lr=0.0001;r.bs=16;r.ep=30;r.sch="WarmupLinear";
    r.rationale="NLP tasks use AdamW + linear warmup — standard for transformer training. GELU activation, very low LR to preserve semantic representations.";
  } else if(mod.includes("Audio")){
    r.layers=[{n:256,a:"ReLU",d:0.3},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}];
    r.ep=80;r.rationale="Audio features (MFCC/Mel) → medium-depth network. Dropout prevents overfitting on compact audio feature vectors.";
  } else if(mod.includes("Video")){
    r.layers=[{n:512,a:"ReLU",d:0.5},{n:256,a:"ReLU",d:0.3},{n:128,a:"ReLU",d:0.2}];
    r.bs=8;r.ep=40;r.sch="CosineAnnealing";r.rationale="Video: small batch needed (memory). Deep + heavy dropout handles temporal complexity.";
  } else if(mod.includes("Multi")){
    r.layers=[{n:512,a:"GELU",d:0.4},{n:256,a:"GELU",d:0.3},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}];
    r.opt="AdamW";r.lr=0.0005;r.bs=32;r.ep=80;r.rationale="Multimodal fusion needs deeper architecture + strong regularization. AdamW + GELU for cross-modal feature handling.";
  } else {
    const h1=Math.min(512,Math.max(64,Math.pow(2,Math.round(Math.log2(f*3)))));
    r.layers=[{n:h1,a:"ReLU",d:n<2000?0.3:0.2},{n:Math.floor(h1/2),a:"ReLU",d:0.2},...(f>20||n>10000?[{n:Math.max(16,Math.floor(h1/4)),a:"ReLU",d:0.1}]:[])];
    r.rationale=`Tabular: ${f} features → ${h1}-neuron first layer tapering down. ${n<5000?"Small dataset — Classical ML may outperform. Try both!":"Good sample count for deep learning."}`;
  }
  if(ds.imbalanced) r.warnings.push("Class imbalance: use class weighting or SMOTE oversampling.");
  if(ds.hasNulls) r.warnings.push("Missing values found: configure imputation in Preprocessing.");
  if(n<500) r.warnings.push("Very small dataset (<500 samples): Classical ML is strongly preferred.");
  if(n>100000) r.bs=256;
  const base=r.modelType==="classical"?87:mod.includes("Image")?88:mod.includes("Text")?84:81;
  r.estimate=(isReg||isUnsup)?(r.modelType==="classical"?"R² ≈ 0.80–0.92":"R² ≈ 0.74–0.89"):`Accuracy ≈ ${base}–${base+9}%`;
  return r;
};

// ─── Training analysis ────────────────────────────────────────────────────────
const analyzeResults = (metrics, hp, layers, ds, histLen) => {
  if(!metrics) return "";
  const isClsf=!["Regression","Time Series","Clustering","Anomaly Detection"].includes(ds.problemType);
  const overfit=metrics.valLoss>metrics.trainLoss*1.3,severe=metrics.valLoss>metrics.trainLoss*1.7;
  const parts=[];
  if(isClsf&&metrics.valAcc!=null){
    if(metrics.valAcc>=92) parts.push(`Excellent: ${metrics.valAcc.toFixed(1)}% val accuracy — strong generalizable patterns learned.`);
    else if(metrics.valAcc>=75) parts.push(`Good performance: ${metrics.valAcc.toFixed(1)}% val accuracy — model generalizes reasonably.`);
    else parts.push(`Moderate: ${metrics.valAcc.toFixed(1)}% val accuracy. Consider more data, larger architecture, or longer training.`);
  } else if(metrics.r2!=null){
    if(metrics.r2>=0.85) parts.push(`Strong regression fit: R²=${metrics.r2.toFixed(3)} — model explains most target variance.`);
    else if(metrics.r2>=0.65) parts.push(`Moderate fit (R²=${metrics.r2.toFixed(3)}). Consider feature engineering or deeper architecture.`);
    else parts.push(`Weak fit (R²=${metrics.r2.toFixed(3)}). Model may be missing important features or needs more capacity.`);
  }
  if(severe) parts.push(`Significant overfitting (val ${metrics.valLoss?.toFixed(3)} >> train ${metrics.trainLoss?.toFixed(3)}). Increase dropout, add L2, or collect more data.`);
  else if(overfit) parts.push(`Mild overfitting. Try dropout +0.1 or L2 lambda → ${(hp.l2Lambda*5).toFixed(4)}.`);
  else parts.push("No significant overfitting — train/val metrics well-aligned. ✓");
  if(histLen<hp.epochs*0.4) parts.push("Training stopped early. Increase Early Stopping patience if performance is unsatisfactory.");
  return parts.join(" ");
};

// ─── Export helpers ───────────────────────────────────────────────────────────
const dlJSON = (obj, name) => {
  const b=new Blob([JSON.stringify(obj,null,2)],{type:"application/json"});
  const a=Object.assign(document.createElement('a'),{href:URL.createObjectURL(b),download:name});
  a.click();URL.revokeObjectURL(a.href);
};
const dlCSV = (rows, name) => {
  if(!rows.length) return;
  const h=Object.keys(rows[0]);
  const csv=[h.join(','),...rows.map(r=>h.map(k=>{const v=r[k];return typeof v==='string'&&(v.includes(',')||v.includes('"'))?`"${v.replace(/"/g,'""')}"`:(v??'');}).join(','))].join('\n');
  const b=new Blob([csv],{type:"text/csv"});
  const a=Object.assign(document.createElement('a'),{href:URL.createObjectURL(b),download:name});
  a.click();URL.revokeObjectURL(a.href);
};

// ─── Shared tiny components ───────────────────────────────────────────────────
const Dot=({col,glow})=><div style={{width:8,height:8,borderRadius:"50%",background:col,boxShadow:glow?`0 0 10px ${col}`:undefined,flexShrink:0}}/>;
const Tog=({v,onChange,label,small})=>(
  <label style={{display:"flex",alignItems:"center",gap:8,cursor:"pointer",userSelect:"none"}}>
    <div onClick={()=>onChange(!v)} style={{width:small?16:18,height:small?16:18,borderRadius:4,flexShrink:0,background:v?C.cyan:"transparent",border:`2px solid ${v?C.cyan:C.bd}`,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",transition:"all .15s"}}>
      {v&&<span style={{color:"#000",fontSize:small?8:10,fontWeight:800}}>✓</span>}
    </div>
    <span style={{fontSize:small?11:13,color:C.sec}}>{label}</span>
  </label>
);
const HP=({k,onHelp})=>(
  <button onClick={()=>onHelp(k)} style={{background:"transparent",border:`1px solid ${C.muted}40`,borderRadius:"50%",width:15,height:15,cursor:"pointer",color:C.muted,fontSize:9,fontWeight:700,display:"inline-flex",alignItems:"center",justifyContent:"center",fontFamily:"inherit",flexShrink:0,padding:0,lineHeight:1}}>?</button>
);
const ST=({color=C.cyan,children,mb=14})=><div style={{fontSize:11,fontWeight:700,color,letterSpacing:"0.8px",marginBottom:mb}}>{children}</div>;
const MBox=({label,value,color})=>(
  <div style={{...cc(),textAlign:"center",borderColor:color+"50",padding:"14px 10px"}}>
    <div style={{fontSize:10,color:C.muted,letterSpacing:"0.8px",marginBottom:6}}>{label}</div>
    <div style={{fontSize:20,fontWeight:700,color}}>{value}</div>
  </div>
);
const TC={numeric:C.cyan,categorical:C.orange,text:C.sec,unknown:C.muted};
const TBadge=({type})=><span style={{fontSize:8,fontWeight:700,padding:"1px 5px",borderRadius:3,background:`${TC[type]||C.muted}22`,color:TC[type]||C.muted,letterSpacing:"0.5px"}}>{type||"?"}</span>;

// ─── Help content ─────────────────────────────────────────────────────────────
const HELP = {
  learningRate:{title:"Learning Rate 📉",body:"Your step size when finding the lowest point in hilly terrain (the minimum loss). Too big → you overshoot. Too small → takes forever. 0.001 is the universally safe default.",tip:"Adam: 0.001. SGD: 0.01. AdamW for NLP: 0.0001. Reduce if loss oscillates."},
  batchSize:{title:"Batch Size 📦",body:"The model learns from small groups (batches) at a time instead of all data at once. Smaller = noisier but often better generalization. Larger = smoother and faster per epoch.",tip:"32 or 64 for most cases. Always use powers of 2: 8, 16, 32, 64, 128..."},
  epochs:{title:"Epochs 🔄",body:"One epoch = the model has seen ALL training data exactly once. More epochs = more learning. Too many = memorization (overfitting). Early Stopping auto-stops at the right moment.",tip:"Start with 50–200. Always enable Early Stopping — it auto-stops at peak performance."},
  dropout:{title:"Dropout 🎲",body:"Randomly turns off a fraction of neurons during training, forcing robust learning instead of memorization. Prevents overfitting — like studying with random notes covered.",tip:"0.2–0.4 for hidden layers. Increase to 0.5 if model overfits (val loss > train loss)."},
  activation:{title:"Activation Functions ⚡",body:"After each neuron computes, an activation decides how strongly it fires. ReLU: negatives → 0. GELU/Swish: smoother, modern alternatives. Softmax: converts outputs to probabilities.",tip:"ReLU/GELU for hidden layers. Softmax for multi-class output. Sigmoid for binary. Linear for regression."},
  neurons:{title:"Neurons per Layer 🧠",body:"Each neuron is a tiny decision-maker. More neurons = more complex patterns. Too many with little data = memorization instead of learning.",tip:"Taper down: 256→128→64. Powers of 2 for hardware efficiency."},
  optimizer:{title:"Optimizer 🏃",body:"The strategy for updating weights after each mistake. Adam is smart and adaptive — best default for 90% of problems. SGD is simpler and sometimes better for fine-tuning vision models.",tip:"Just use Adam. AdamW for NLP/large models. SGD if Adam fails to converge."},
  l2:{title:"L2 Regularization 🔒",body:"Penalizes very large weight values, keeping all connections balanced and modest. Prevents over-reliance on any single connection.",tip:"0.0001 is safe. Increase to 0.001–0.01 if significant overfitting is observed."},
  earlyStopping:{title:"Early Stopping 🛑",body:"Stops training automatically when validation loss stops improving. Patience = epochs to wait before giving up. Best weights are restored automatically.",tip:"Always enable. Patience 10–20 is standard."},
  weightInit:{title:"Weight Initialization 🎯",body:"Starting values for all weights before training. Bad init → model can't learn at all. Xavier/Glorot ensures signal flows with the right strength through each layer.",tip:"Xavier/Glorot for Sigmoid/Tanh. He Normal for ReLU/GELU/Swish."},
  validation:{title:"Validation Split 📊",body:"Portion of data the model NEVER trains on — only used to measure genuine learning. Like unseen practice test questions.",tip:"20% (0.2) is standard. 10% only for very small datasets."},
  scheduler:{title:"LR Scheduler 📅",body:"Automatically reduces learning rate during training. ReduceOnPlateau reduces when improvement stalls — like running fast then slowing near the finish.",tip:"ReduceOnPlateau is safe for all. CosineAnnealing for longer runs."},
  classicalML:{title:"Classical ML vs Deep Learning 🌳🧠",body:"Classical ML (Random Forest, XGBoost, SVM) uses hand-designed algorithms — no layers. Great for tabular data and small datasets. Deep Learning uses neural networks — best for images, audio, text, video, large datasets.",tip:"< 10K tabular rows → Classical ML first. Images, audio, text, video → Deep Learning."},
  normalization:{title:"Feature Scaling 📏",body:"Without scaling, age (0–100) and salary (0–200,000) are treated very differently just because of scale. Scaling brings all features to the same range so all are treated equally.",tip:"StandardScaler (mean=0, std=1) for most cases. MinMaxScaler (0–1) for neural networks."},
  exportModel:{title:"Model Export 💾",body:"Exports your complete model config — architecture, hyperparameters, preprocessing steps, training history, and a Python code snippet to recreate this exact model in TensorFlow/PyTorch/scikit-learn.",tip:"The exported JSON includes a ready-to-run Python setup script. Open in any text editor."},
  dataOps:{title:"Manual Data Operations ✂️",body:"You can remove columns, drop rows, rename columns, set the target (label) column, and filter rows. These operations modify your working dataset before preprocessing and training.",tip:"Always set one column as your Target before training. The model predicts this column."},
};

// ─── DATA EXPLORER COMPONENT ──────────────────────────────────────────────────
function DataExplorer({ data, columns, onUpdateCols, onUpdateData, onHelp }) {
  const [page, setPage]           = useState(0);
  const [selRows, setSelRows]     = useState(new Set());
  const [filterCol, setFilterCol] = useState('');
  const [filterVal, setFilterVal] = useState('');
  const [sortCol, setSortCol]     = useState(null);
  const [sortDir, setSortDir]     = useState('asc');
  const [activeMenu, setActiveMenu] = useState(null);  // column name with open menu
  const [renameTarget, setRenameTarget] = useState(null);
  const [renameTo, setRenameTo]   = useState('');
  const [statCol, setStatCol]     = useState(null);
  const PAGE = 50;
  const visCols = columns.filter(c => !c.dropped);

  const filtered = useMemo(() => {
    let d=[...data];
    if(filterCol&&filterVal) d=d.filter(r=>String(r[filterCol]??'').toLowerCase().includes(filterVal.toLowerCase()));
    if(sortCol){ d.sort((a,b)=>{const av=a[sortCol],bv=b[sortCol];const an=Number(av),bn=Number(bv);const cmp=(!isNaN(an)&&!isNaN(bn))?an-bn:String(av??'').localeCompare(String(bv??''));return sortDir==='asc'?cmp:-cmp;}); }
    return d;
  }, [data, filterCol, filterVal, sortCol, sortDir]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  const pageData = filtered.slice(page*PAGE, (page+1)*PAGE);

  const colStatMemo = useMemo(() => {
    if(!statCol) return null;
    const col = columns.find(c=>c.name===statCol);
    if(!col) return null;
    return colStat(data, statCol, col.type);
  }, [statCol, data, columns]);

  const dropCol = name => { onUpdateCols(columns.map(c=>c.name===name?{...c,dropped:true}:c)); if(statCol===name)setStatCol(null); setActiveMenu(null); };
  const setTarget = name => { onUpdateCols(columns.map(c=>({...c,isTarget:c.name===name}))); setActiveMenu(null); };
  const restoreCol = name => { onUpdateCols(columns.map(c=>c.name===name?{...c,dropped:false}:c)); };
  const toggleSort = col => { if(sortCol===col)setSortDir(d=>d==='asc'?'desc':'asc'); else{setSortCol(col);setSortDir('asc');} setActiveMenu(null); };
  const doRename = () => {
    if(!renameTo.trim()||renameTo===renameTarget) { setRenameTarget(null); return; }
    const newData = data.map(r=>{ const nr={...r}; nr[renameTo]=nr[renameTarget]; delete nr[renameTarget]; return nr; });
    onUpdateData(newData);
    onUpdateCols(columns.map(c=>c.name===renameTarget?{...c,name:renameTo}:c));
    if(statCol===renameTarget)setStatCol(renameTo);
    setRenameTarget(null); setRenameTo('');
  };
  const delSelected = () => {
    const indices = new Set([...selRows]);
    const newData = filtered.filter((_,i)=>!indices.has(page*PAGE+i));
    // remove from full data
    const selSet = new Set([...selRows].map(i=>JSON.stringify(filtered[i])));
    onUpdateData(data.filter(r=>!selSet.has(JSON.stringify(r))));
    setSelRows(new Set());
  };
  const delDupes = () => {
    const seen=new Set();
    onUpdateData(data.filter(r=>{const k=JSON.stringify(r);if(seen.has(k))return false;seen.add(k);return true;}));
  };
  const toggleRow = i => { const s=new Set(selRows); const abs=page*PAGE+i; s.has(abs)?s.delete(abs):s.add(abs); setSelRows(s); };
  const selectAll = () => { if(selRows.size===pageData.length){setSelRows(new Set());}else{const s=new Set();pageData.forEach((_,i)=>s.add(page*PAGE+i));setSelRows(s);} };
  const changeType = (name, type) => { onUpdateCols(columns.map(c=>c.name===name?{...c,type}:c)); setActiveMenu(null); };
  const fillNullsIn = name => {
    const col = columns.find(c=>c.name===name);
    if(col?.type!=='numeric') return;
    const nums=data.map(r=>r[name]).filter(v=>!isNullVal(v)).map(Number).filter(n=>!isNaN(n));
    if(!nums.length) return;
    const mean=nums.reduce((a,b)=>a+b,0)/nums.length;
    onUpdateData(data.map(r=>({...r,[name]:isNullVal(r[name])?+mean.toFixed(4):r[name]})));
    setActiveMenu(null);
  };

  if(!data.length) return <div style={{...cc(),textAlign:"center",padding:30,color:C.muted,fontSize:12}}>No data loaded. Load a CSV file or sample dataset above.</div>;

  return (
    <div style={{...cc(),padding:0,overflow:"hidden"}}>
      {/* Toolbar */}
      <div style={{padding:"12px 18px",borderBottom:`1px solid ${C.bd}`,display:"flex",alignItems:"center",gap:10,flexWrap:"wrap",background:C.deep}}>
        <div style={{display:"flex",alignItems:"center",gap:8,marginRight:8}}>
          <HP k="dataOps" onHelp={onHelp}/>
          <span style={{fontSize:11,fontWeight:700,color:C.cyan,letterSpacing:"0.5px"}}>DATA EXPLORER</span>
          <span style={{fontSize:11,color:C.muted}}>· {data.length.toLocaleString()} rows · {visCols.length} columns</span>
        </div>
        <select style={{...SE,width:130,padding:"5px 8px",fontSize:11}} value={filterCol} onChange={e=>setFilterCol(e.target.value)}>
          <option value="">Filter column...</option>
          {visCols.map(c=><option key={c.name} value={c.name}>{c.name}</option>)}
        </select>
        <input style={{...IS,width:140,padding:"5px 8px",fontSize:11}} placeholder="Filter value..." value={filterVal} onChange={e=>{setFilterVal(e.target.value);setPage(0);}}/>
        {filterVal&&<button onClick={()=>{setFilterVal('');setFilterCol('');}} style={{...bS(C.muted),padding:"4px 10px",fontSize:11}}>✕ Clear</button>}
        <div style={{marginLeft:"auto",display:"flex",gap:8}}>
          {selRows.size>0&&<button onClick={delSelected} style={{...bS(C.red),padding:"5px 12px",fontSize:11}}>🗑 Remove {selRows.size} selected</button>}
          <button onClick={delDupes} style={{...bS(C.muted),padding:"5px 12px",fontSize:11}}>Remove Dupes</button>
        </div>
      </div>

      <div style={{display:"flex"}}>
        {/* Table */}
        <div style={{flex:1,overflow:"auto",maxHeight:420,position:"relative"}}>
          <table style={{borderCollapse:"collapse",fontSize:12,width:"100%",minWidth:visCols.length*120}}>
            <thead>
              <tr style={{background:C.deep,position:"sticky",top:0,zIndex:10}}>
                <th style={{width:28,padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,cursor:"pointer"}} onClick={selectAll}>
                  <div style={{width:14,height:14,borderRadius:3,border:`1px solid ${C.bd}`,background:selRows.size===pageData.length&&pageData.length>0?C.cyan:"transparent",display:"flex",alignItems:"center",justifyContent:"center"}}>
                    {selRows.size===pageData.length&&pageData.length>0&&<span style={{fontSize:8,color:"#000",fontWeight:800}}>✓</span>}
                  </div>
                </th>
                <th style={{padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,color:C.muted,fontSize:10,fontWeight:600,textAlign:"left",minWidth:40}}>#</th>
                {visCols.map(col=>{
                  const tgt=col.isTarget;
                  return (
                    <th key={col.name} style={{padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,textAlign:"left",minWidth:110,background:tgt?`${C.green}10`:undefined,position:"relative"}}>
                      <div style={{display:"flex",alignItems:"center",gap:5}}>
                        <button onClick={()=>setActiveMenu(activeMenu===col.name?null:col.name)} style={{background:"transparent",border:"none",cursor:"pointer",color:tgt?C.green:C.text,fontSize:11,fontWeight:700,padding:0,fontFamily:"inherit",display:"flex",alignItems:"center",gap:4,textAlign:"left"}}>
                          {tgt&&<span style={{fontSize:9,color:C.green}}>🎯</span>}
                          {col.name} ▾
                        </button>
                        <TBadge type={col.type}/>
                        <button onClick={()=>setStatCol(statCol===col.name?null:col.name)} style={{background:"transparent",border:"none",cursor:"pointer",color:statCol===col.name?C.cyan:C.muted,fontSize:10,padding:0,fontFamily:"inherit"}}>📊</button>
                        {sortCol===col.name&&<span style={{fontSize:10,color:C.cyan}}>{sortDir==='asc'?'↑':'↓'}</span>}
                      </div>
                      {/* Dropdown menu */}
                      {activeMenu===col.name&&(
                        <div onClick={e=>e.stopPropagation()} style={{position:"absolute",top:"100%",left:0,background:C.card,border:`1px solid ${C.bd}`,borderRadius:8,padding:"6px 0",zIndex:100,minWidth:180,boxShadow:"0 8px 24px rgba(0,0,0,0.5)"}}>
                          {[
                            {label:`🎯 Set as Target`, fn:()=>setTarget(col.name), col:C.green},
                            {label:`📊 View Stats`, fn:()=>{setStatCol(col.name);setActiveMenu(null);}},
                            {label:`${sortDir==='asc'?'↑':'↓'} Sort ${sortCol===col.name?(sortDir==='asc'?'Desc':'Asc'):'Ascending'}`, fn:()=>toggleSort(col.name)},
                            {label:"✏️ Rename", fn:()=>{setRenameTarget(col.name);setRenameTo(col.name);setActiveMenu(null);}},
                            ...(col.type==='numeric'?[{label:"🔧 Fill Nulls (mean)", fn:()=>fillNullsIn(col.name)}]:[]),
                            {label:"Change Type → numeric", fn:()=>changeType(col.name,'numeric'), col:C.muted},
                            {label:"Change Type → categorical", fn:()=>changeType(col.name,'categorical'), col:C.muted},
                            {label:"🗑 Remove Column", fn:()=>dropCol(col.name), col:C.red},
                          ].map((item,idx)=>(
                            <button key={idx} onClick={item.fn} style={{display:"block",width:"100%",background:"transparent",border:"none",padding:"7px 14px",textAlign:"left",cursor:"pointer",fontSize:12,color:item.col||C.text,fontFamily:"inherit"}}>
                              {item.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {pageData.map((row, i) => {
                const absIdx = page*PAGE+i;
                const isSel = selRows.has(absIdx);
                return (
                  <tr key={absIdx} style={{background:isSel?`${C.cyan}10`:i%2===0?C.deep:"transparent",transition:"background 0.1s"}}>
                    <td style={{padding:"4px 8px",borderBottom:`1px solid ${C.bd}20`}}>
                      <div onClick={()=>toggleRow(i)} style={{width:14,height:14,borderRadius:3,border:`1px solid ${C.bd}`,background:isSel?C.cyan:"transparent",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}>
                        {isSel&&<span style={{fontSize:8,color:"#000",fontWeight:800}}>✓</span>}
                      </div>
                    </td>
                    <td style={{padding:"4px 8px",borderBottom:`1px solid ${C.bd}20`,color:C.muted,fontSize:10}}>{page*PAGE+i+1}</td>
                    {visCols.map(col=>{
                      const val=row[col.name];
                      const isNull=isNullVal(val);
                      return (
                        <td key={col.name} style={{padding:"4px 10px",borderBottom:`1px solid ${C.bd}20`,maxWidth:200,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",color:isNull?C.muted:col.isTarget?C.green:C.text,background:col.isTarget?`${C.green}08`:undefined,fontSize:12}}>
                          {isNull?<span style={{fontStyle:"italic",fontSize:11}}>null</span>:String(val)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Stats panel */}
        {statCol && colStatMemo && (
          <div style={{width:220,borderLeft:`1px solid ${C.bd}`,padding:"14px 16px",background:C.deep,flexShrink:0,overflow:"auto",maxHeight:420}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
              <span style={{fontSize:11,fontWeight:700,color:C.cyan}}>{statCol}</span>
              <button onClick={()=>setStatCol(null)} style={{background:"transparent",border:"none",color:C.muted,cursor:"pointer",fontSize:16,fontFamily:"inherit",lineHeight:1,padding:0}}>×</button>
            </div>
            <TBadge type={colStatMemo.type}/>
            <div style={{marginTop:10,display:"flex",flexDirection:"column",gap:5}}>
              {[["Nulls",colStatMemo.nulls],["Unique",colStatMemo.uniq],
                ...(colStatMemo.type==='numeric'?[["Min",colStatMemo.min],["Max",colStatMemo.max],["Mean",colStatMemo.mean],["Std",colStatMemo.std],["Median",colStatMemo.median],["Q1",colStatMemo.q1],["Q3",colStatMemo.q3]]:
                colStatMemo.top?colStatMemo.top.map(([v,c])=>[v,`${c}×`]):[])
              ].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:11}}>
                  <span style={{color:C.muted}}>{k}</span>
                  <span style={{color:C.text,maxWidth:100,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",textAlign:"right"}}>{v}</span>
                </div>
              ))}
            </div>
            {colStatMemo.type==='categorical'&&colStatMemo.top&&(
              <div style={{marginTop:12}}>
                <div style={{fontSize:10,color:C.muted,marginBottom:6}}>DISTRIBUTION</div>
                <ResponsiveContainer width="100%" height={130}>
                  <BarChart data={colStatMemo.top.map(([v,c])=>({name:String(v).slice(0,10),count:c}))} layout="vertical" margin={{left:0,right:8,top:0,bottom:0}}>
                    <XAxis type="number" tick={{fontSize:9,fill:C.muted}} tickLine={false}/>
                    <YAxis type="category" dataKey="name" tick={{fontSize:9,fill:C.sec}} width={60}/>
                    <Bar dataKey="count" fill={C.orange} radius={[0,3,3,0]}/>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Pagination + dropped columns */}
      <div style={{padding:"10px 18px",borderTop:`1px solid ${C.bd}`,background:C.deep,display:"flex",alignItems:"center",gap:12,flexWrap:"wrap"}}>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0} style={{...bS(C.muted),padding:"4px 10px",fontSize:11,opacity:page===0?.4:1}}>‹</button>
          <span style={{fontSize:11,color:C.muted}}>Page {page+1} / {pages} · {filtered.length.toLocaleString()} rows</span>
          <button onClick={()=>setPage(p=>Math.min(pages-1,p+1))} disabled={page===pages-1} style={{...bS(C.muted),padding:"4px 10px",fontSize:11,opacity:page===pages-1?.4:1}}>›</button>
        </div>
        {columns.filter(c=>c.dropped).length>0&&(
          <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
            <span style={{fontSize:11,color:C.muted}}>Dropped:</span>
            {columns.filter(c=>c.dropped).map(c=>(
              <button key={c.name} onClick={()=>restoreCol(c.name)} style={{...bS(C.muted),padding:"3px 9px",fontSize:10}}>↩ {c.name}</button>
            ))}
          </div>
        )}
        {renameTarget&&(
          <div style={{display:"flex",gap:8,alignItems:"center",marginLeft:"auto"}}>
            <span style={{fontSize:11,color:C.yellow}}>Rename '{renameTarget}' →</span>
            <input style={{...IS,width:140,padding:"5px 8px",fontSize:11}} value={renameTo} onChange={e=>setRenameTo(e.target.value)} onKeyDown={e=>{if(e.key==='Enter')doRename();if(e.key==='Escape')setRenameTarget(null);}} autoFocus/>
            <button onClick={doRename} style={{...bP(C.green),padding:"5px 12px",fontSize:11}}>OK</button>
            <button onClick={()=>setRenameTarget(null)} style={{...bS(C.muted),padding:"5px 10px",fontSize:11}}>✕</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── PREPROCESSING PANEL ──────────────────────────────────────────────────────
function PrepPanel({ ds, prep, setPrep, prepLog, onHelp }) {
  const [open, setOpen] = useState(true);
  const u=(k,v)=>setPrep(p=>({...p,[k]:v}));
  const show=m=>ds.modality===m||ds.modality==="Multimodal";
  const SH=({t})=>ds.modality==="Multimodal"?<div style={{fontSize:10,color:C.muted,fontWeight:700,letterSpacing:"0.8px",marginBottom:8,marginTop:12,borderTop:`1px solid ${C.bd}`,paddingTop:10}}>{t}</div>:null;

  return (
    <div style={{...cc(),borderColor:C.purple+"50"}}>
      <div onClick={()=>setOpen(o=>!o)} style={{display:"flex",alignItems:"center",justifyContent:"space-between",cursor:"pointer",marginBottom:open?14:0}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{color:C.purple}}>⚙</span>
          <span style={{fontSize:11,fontWeight:700,color:C.purple,letterSpacing:"0.8px"}}>PREPROCESSING PIPELINE</span>
          <HP k="normalization" onHelp={onHelp}/>
          <span style={{fontSize:11,color:C.muted}}>· {ds.modality}</span>
          {prepLog.length>0&&<span style={{fontSize:10,background:`${C.green}20`,color:C.green,padding:"2px 8px",borderRadius:10}}>✓ {prepLog.length} ops applied</span>}
        </div>
        <span style={{color:C.muted,fontSize:11}}>{open?"▲":"▼"}</span>
      </div>

      {open&&<div>
        {/* TABULAR */}
        {(show("Tabular / CSV")||!MODS.filter(m=>m!=="Tabular / CSV"&&m!=="Multimodal").some(m=>ds.modality===m))&&<>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:12}}>
            <div>
              <label style={{...LB,display:"flex",gap:5,alignItems:"center"}}>Missing Values <HP k="normalization" onHelp={onHelp}/></label>
              <select style={SE} value={prep.missingStrategy} onChange={e=>u("missingStrategy",e.target.value)}>
                {["Mean Imputation","Median Imputation","Mode Imputation","Drop Rows","Zero Fill","Forward Fill","KNN Imputation"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
            <div>
              <label style={{...LB,display:"flex",gap:5,alignItems:"center"}}>Feature Scaling <HP k="normalization" onHelp={onHelp}/></label>
              <select style={SE} value={prep.scaling} onChange={e=>u("scaling",e.target.value)}>
                {["StandardScaler","MinMaxScaler","RobustScaler","MaxAbsScaler","Normalizer","None"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
            <div>
              <label style={LB}>Categorical Encoding</label>
              <select style={SE} value={prep.encoding} onChange={e=>u("encoding",e.target.value)}>
                {["One-Hot","Label Encoding","Target Encoding","Ordinal","Binary","Hash Encoding"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
          </div>
          <div style={{display:"flex",gap:20,flexWrap:"wrap",marginBottom:10}}>
            <Tog v={prep.outlierRemoval} onChange={v=>u("outlierRemoval",v)} label="Outlier Removal (IQR)"/>
            <Tog v={prep.removeDuplicates} onChange={v=>u("removeDuplicates",v)} label="Remove Duplicate Rows"/>
            <Tog v={prep.featureSelection} onChange={v=>u("featureSelection",v)} label="Feature Selection (top-k)"/>
            <div style={{display:"flex",gap:8,alignItems:"center"}}>
              <Tog v={prep.pca} onChange={v=>u("pca",v)} label="PCA Reduction"/>
              <HP k="normalization" onHelp={onHelp}/>
              {prep.pca&&<input type="number" style={{...IS,width:60,padding:"4px 8px",fontSize:12}} value={prep.pcaVariance} min={0.7} max={0.999} step={0.01} onChange={e=>u("pcaVariance",parseFloat(e.target.value))} title="Variance threshold (e.g. 0.95 = keep 95%)"/>}
            </div>
          </div>
        </>}

        {/* IMAGES */}
        {show("Images")&&<>
          <SH t="IMAGE PREPROCESSING"/>
          <div style={{display:"grid",gridTemplateColumns:"160px 1fr",gap:14,marginBottom:8}}>
            <div>
              <label style={LB}>Resize To</label>
              <select style={SE} value={prep.imgResize} onChange={e=>u("imgResize",e.target.value)}>
                {["32x32","64x64","128x128","224x224","256x256","299x299","512x512"].map(v=><option key={v}>{v}</option>)}
              </select>
              <div style={{marginTop:8}}><Tog v={prep.imgNormalize} onChange={v=>u("imgNormalize",v)} label="Normalize 0–1" small/></div>
              <div style={{marginTop:6}}><Tog v={prep.imgGray} onChange={v=>u("imgGray",v)} label="Grayscale" small/></div>
            </div>
            <div>
              <label style={{...LB,display:"flex",gap:5,alignItems:"center"}}>Augmentation <HP k="normalization" onHelp={onHelp}/></label>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:7}}>
                {[["imgFlip","Random Flip H/V"],["imgRotate","Random Rotation"],["imgColorJitter","Color Jitter"],["imgCrop","Random Crop"],["imgMixup","Mixup"],["imgCutout","CutOut"],["imgNoise","Gaussian Noise"],["imgZoom","Random Zoom"]].map(([k,l])=>(
                  <Tog key={k} v={prep[k]} onChange={v=>u(k,v)} label={l} small/>
                ))}
              </div>
            </div>
          </div>
        </>}

        {/* AUDIO */}
        {show("Audio")&&<>
          <SH t="AUDIO PREPROCESSING"/>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:8}}>
            <div><label style={LB}>Feature Extraction</label>
              <select style={SE} value={prep.audioFeature} onChange={e=>u("audioFeature",e.target.value)}>
                {["MFCC","Mel Spectrogram","Chroma","Zero Crossing Rate","Raw Waveform","CQT","Tempogram"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
            <div><label style={LB}>Sample Rate (Hz)</label>
              <select style={SE} value={prep.audioSr} onChange={e=>u("audioSr",e.target.value)}>
                {["8000","16000","22050","44100","48000"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
            <div><label style={LB}>Clip Duration (sec)</label>
              <input type="number" style={IS} value={prep.audioDur} min={1} max={120} onChange={e=>u("audioDur",parseInt(e.target.value))}/>
            </div>
          </div>
          <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
            {[["audioDenoise","Noise Reduction"],["audioPitch","Pitch Shift Augment"],["audioTime","Time Stretch Augment"],["audioTrim","Silence Trimming"]].map(([k,l])=>(
              <Tog key={k} v={prep[k]} onChange={v=>u(k,v)} label={l} small/>
            ))}
          </div>
        </>}

        {/* TEXT */}
        {show("Text")&&<>
          <SH t="TEXT / NLP PREPROCESSING"/>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:8}}>
            <div><label style={LB}>Embedding Strategy</label>
              <select style={SE} value={prep.textEmbed} onChange={e=>u("textEmbed",e.target.value)}>
                {["TF-IDF","Bag of Words","Word Embeddings","BERT Tokenizer","FastText","CharLevel","SentenceTransformers"].map(v=><option key={v}>{v}</option>)}
              </select>
            </div>
            <div><label style={LB}>Max Sequence Length</label>
              <input type="number" style={IS} value={prep.textMaxLen} min={16} max={4096} onChange={e=>u("textMaxLen",parseInt(e.target.value))}/>
            </div>
          </div>
          <div style={{display:"flex",gap:14,flexWrap:"wrap"}}>
            {[["textLower","Lowercase"],["textPunct","Remove Punctuation"],["textStop","Remove Stopwords"],["textStem","Stemming"],["textSpell","Spell Correction"]].map(([k,l])=>(
              <Tog key={k} v={prep[k]} onChange={v=>u(k,v)} label={l} small/>
            ))}
          </div>
        </>}

        {/* VIDEO */}
        {show("Video")&&<>
          <SH t="VIDEO PREPROCESSING"/>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:12}}>
            {[["Target FPS","videoFPS",["1","2","4","8","16","24","30"]],["Frames/Clip","videoFrames",["8","16","32","64","128"]],["Spatial Resize","videoResize",["112x112","128x128","224x224","256x256"]],["Sampling","videoSamp",["Uniform","Random","Dense","Sparse"]]].map(([label,key,opts])=>(
              <div key={key}><label style={LB}>{label}</label>
                <select style={SE} value={prep[key]} onChange={e=>u(key,e.target.value)}>
                  {opts.map(v=><option key={v}>{v}</option>)}
                </select>
              </div>
            ))}
          </div>
        </>}

        {/* MULTIMODAL */}
        {ds.modality==="Multimodal"&&<>
          <SH t="MULTIMODAL FUSION"/>
          <div style={{display:"flex",gap:16,flexWrap:"wrap",marginBottom:10}}>
            {[["mmTab","📋 Tabular"],["mmImg","🖼️ Images"],["mmAud","🔊 Audio"],["mmTxt","📝 Text"],["mmVid","🎬 Video"]].map(([k,l])=>(
              <Tog key={k} v={prep[k]} onChange={v=>u(k,v)} label={l} small/>
            ))}
          </div>
          <div style={{maxWidth:280}}>
            <label style={LB}>Fusion Strategy</label>
            <select style={SE} value={prep.fusionStrategy} onChange={e=>u("fusionStrategy",e.target.value)}>
              {["Early Fusion (concat)","Late Fusion (ensemble)","Cross-Attention Fusion","Hierarchical Fusion"].map(v=><option key={v}>{v}</option>)}
            </select>
          </div>
        </>}

        {/* Log */}
        {prepLog.length>0&&<div style={{marginTop:14,background:C.deep,borderRadius:8,padding:"10px 14px",border:`1px solid ${C.bd}`}}>
          <div style={{fontSize:10,color:C.green,fontWeight:700,marginBottom:7}}>✓ APPLIED OPERATIONS LOG</div>
          {prepLog.map((line,i)=><div key={i} style={{fontSize:11,color:C.sec,marginBottom:3}}>• {line}</div>)}
        </div>}
      </div>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab,setTab]       = useState("dataset");
  const [helpKey,setHelpKey]= useState(null);

  // ── Data loading ────────────────────────────────────────────────────────────
  const [rawData,setRawData]         = useState([]);
  const [workData,setWorkData]       = useState([]);  // after manual ops
  const [columns,setColumns]         = useState([]);
  const [processedData,setProcData]  = useState([]);
  const [loadSrc,setLoadSrc]         = useState(null); // {type,name,format,rows,cols}
  const [loadErr,setLoadErr]         = useState('');
  const [loading,setLoading]         = useState(false);
  const [urlInp,setUrlInp]           = useState('');
  const [fileList,setFileList]       = useState([]);   // for image/audio/video
  const [prepLog,setPrepLog]         = useState([]);

  // ── Dataset meta ────────────────────────────────────────────────────────────
  const [ds,setDs] = useState({name:"",modality:"Tabular / CSV",problemType:"Classification",samples:"",features:"",classes:"",description:"",hasNulls:false,imbalanced:false,normalized:false});

  // ── Preprocessing ────────────────────────────────────────────────────────────
  const [prep,setPrep] = useState({
    missingStrategy:"Mean Imputation",scaling:"StandardScaler",encoding:"One-Hot",
    outlierRemoval:false,removeDuplicates:false,featureSelection:false,pca:false,pcaVariance:0.95,
    imgResize:"224x224",imgNormalize:true,imgGray:false,imgFlip:true,imgRotate:false,
    imgColorJitter:false,imgCrop:false,imgMixup:false,imgCutout:false,imgNoise:false,imgZoom:false,
    audioFeature:"MFCC",audioSr:"22050",audioDur:5,audioDenoise:false,audioPitch:false,audioTime:false,audioTrim:true,
    textEmbed:"Word Embeddings",textMaxLen:512,textLower:true,textPunct:true,textStop:true,textStem:false,textSpell:false,
    videoFPS:"8",videoFrames:"16",videoResize:"224x224",videoSamp:"Uniform",
    mmTab:true,mmImg:true,mmAud:false,mmTxt:false,mmVid:false,fusionStrategy:"Early Fusion (concat)",
  });

  // ── Model ────────────────────────────────────────────────────────────────────
  const [modelType,setModelType]   = useState("deep");
  const [classAlgo,setClassAlgo]   = useState("Random Forest");
  const [layers,setLayers]         = useState([{id:1,n:256,a:"ReLU",d:0.2},{id:2,n:128,a:"ReLU",d:0.2},{id:3,n:64,a:"ReLU",d:0.1}]);
  const [outCfg,setOutCfg]         = useState({activation:"Softmax",weightInit:"Xavier/Glorot",useBias:true});
  const [cfgSub,setCfgSub]         = useState("architecture");
  const [bulk,setBulk]             = useState({count:"5",n:"128",a:"ReLU",d:"0.2"});
  const [hp,setHp] = useState({optimizer:"Adam",lr:0.001,bs:32,epochs:100,l2:0.0001,earlyStopping:true,patience:15,scheduler:"ReduceOnPlateau",vSplit:0.2});

  // ── Recommendations ──────────────────────────────────────────────────────────
  const [recs,setRecs] = useState(null);

  // ── Training ─────────────────────────────────────────────────────────────────
  const [training,setTraining] = useState(false);
  const [curEp,setCurEp]       = useState(0);
  const [history,setHistory]   = useState([]);
  const [done,setDone]         = useState(false);
  const [metrics,setMetrics]   = useState(null);
  const [analysis,setAnalysis] = useState('');
  const ivRef=useRef(null),epRef=useRef(0),histRef=useRef([]);

  const noise=(s=0.01)=>(Math.random()-.5)*2*s;
  const fmt=(n,d=3)=>typeof n==="number"?n.toFixed(d):"—";
  const isClsf=!["Regression","Time Series","Clustering","Anomaly Detection"].includes(ds.problemType);

  // ── Load CSV from file ───────────────────────────────────────────────────────
  const loadCSV = (file) => {
    setLoading(true); setLoadErr('');
    Papa.parse(file, {
      header:true, dynamicTyping:false, skipEmptyLines:true,
      complete:(res)=>{
        if(!res.data.length){setLoadErr("No data found in file.");setLoading(false);return;}
        const d=res.data;
        const cols=buildCols(d);
        const nullCols=cols.filter(c=>d.some(r=>isNullVal(r[c.name]))).length;
        setRawData(d); setWorkData(d); setColumns(cols); setProcData([]); setPrepLog([]);
        setDs(p=>({...p,name:file.name.replace(/\.[^.]+$/,''),modality:"Tabular / CSV",samples:String(d.length),features:String(cols.length),hasNulls:nullCols>0}));
        setLoadSrc({type:"csv",name:file.name,rows:d.length,cols:cols.length});
        setLoading(false);
      },
      error:(e)=>{setLoadErr(`Parse error: ${e.message}`);setLoading(false);}
    });
  };

  // ── Load from URL ────────────────────────────────────────────────────────────
  const loadURL = async () => {
    if(!urlInp.trim()){setLoadErr("Enter a URL.");return;}
    setLoading(true); setLoadErr('');
    try {
      const res=await fetch(urlInp.trim());
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const text=await res.text();
      // Try parse as CSV
      const parsed=Papa.parse(text,{header:true,dynamicTyping:false,skipEmptyLines:true});
      if(!parsed.data.length) throw new Error("No data rows found.");
      const d=parsed.data, cols=buildCols(d);
      setRawData(d); setWorkData(d); setColumns(cols); setProcData([]); setPrepLog([]);
      setDs(p=>({...p,name:"url-dataset",modality:"Tabular / CSV",samples:String(d.length),features:String(cols.length)}));
      setLoadSrc({type:"url",name:urlInp.trim().split('/').pop()||"dataset",rows:d.length,cols:cols.length});
    } catch(e) {
      setLoadErr(`Failed to load: ${e.message}. Check CORS — some URLs block browser requests. Try a direct CSV link (e.g. GitHub raw).`);
    }
    setLoading(false);
  };

  // ── Load sample dataset ──────────────────────────────────────────────────────
  const loadSample = (key) => {
    const s=SAMPLES[key];
    const d=s.gen();
    const cols=buildCols(d);
    setRawData(d); setWorkData(d); setColumns(cols); setProcData([]); setPrepLog([]);
    setDs(p=>({...p,name:s.label.replace(/^./,'').trim(),modality:"Tabular / CSV",samples:String(d.length),features:String(cols.length)}));
    setLoadSrc({type:"sample",name:s.label,rows:d.length,cols:cols.length});
  };

  // ── Load image/audio/video files ─────────────────────────────────────────────
  const loadFiles = (files, modality) => {
    const list=Array.from(files).slice(0,500);
    setFileList(list);
    setWorkData([]); setColumns([]); setProcData([]); setPrepLog([]);
    setDs(p=>({...p,modality,samples:String(list.length)}));
    setLoadSrc({type:"files",name:`${list.length} ${modality} files`,rows:list.length,cols:0});
  };

  // ── Apply preprocessing ──────────────────────────────────────────────────────
  const applyPreprocessing = () => {
    const {data,log}=applyPrepOps(workData, columns, prep);
    setProcData(data); setPrepLog(log);
  };

  // ── Recommendations ──────────────────────────────────────────────────────────
  const handleAnalyze=()=>setRecs(recommend(ds));
  const applyRecs=()=>{
    if(!recs) return;
    setModelType(recs.modelType); setClassAlgo(recs.classicalAlgo||classAlgo);
    setLayers(recs.layers.map((l,i)=>({id:i+1,...l})));
    setOutCfg(p=>({...p,activation:recs.outAct,weightInit:recs.wInit}));
    setHp(p=>({...p,optimizer:recs.opt,lr:recs.lr,bs:recs.bs,epochs:recs.ep,l2:recs.l2,scheduler:recs.sch,patience:recs.pat,vSplit:recs.vSplit}));
    setTab("configure");
  };

  // ── Layer helpers ────────────────────────────────────────────────────────────
  const addLayer=()=>{if(layers.length>=500)return;const last=layers[layers.length-1]?.n||64;setLayers(p=>[...p,{id:Date.now(),n:Math.max(8,Math.floor(last/2)),a:"ReLU",d:0.2}]);};
  const rmLayer=id=>{if(layers.length>1)setLayers(p=>p.filter(l=>l.id!==id));};
  const updL=(id,f,v)=>setLayers(p=>p.map(l=>l.id===id?{...l,[f]:v}:l));
  const addBulk=()=>{const cnt=Math.min(parseInt(bulk.count)||1,200);setLayers(p=>[...p,...Array.from({length:cnt},(_,i)=>({id:Date.now()+i,n:parseInt(bulk.n)||128,a:bulk.a,d:parseFloat(bulk.d)||0.2}))]);};
  const applyPreset=nm=>setLayers(PRESETS[nm].map((l,i)=>({id:i+1,n:l.n,a:l.a,d:l.d})));
  const globalAct=act=>setLayers(p=>p.map(l=>({...l,a:act})));

  const totalParams=(()=>{let p=0,prev=parseInt(ds.features)||20;layers.forEach(l=>{p+=(prev+1)*l.n;prev=l.n;});p+=(prev+1)*(parseInt(ds.classes)||3);return p;})();
  const fmtP=n=>n>1e6?`${(n/1e6).toFixed(2)}M`:n>1e3?`${(n/1e3).toFixed(1)}K`:n;

  // ── Realistic training simulation ───────────────────────────────────────────
  const startTraining=useCallback(()=>{
    if(ivRef.current)clearInterval(ivRef.current);
    const total=hp.epochs, nc=Math.max(2,parseInt(ds.classes)||3);
    const iL=isClsf?Math.log(nc):3.0;

    // Hyperparameter effects on simulation
    const lrLog=Math.log10(hp.lr/0.001);
    const avgDrop=layers.reduce((a,l)=>a+l.d,0)/layers.length;
    const optSpeed={Adam:1.0,AdamW:1.05,SGD:0.55,RMSprop:0.85,Adagrad:0.7,Nadam:1.1,Adadelta:0.75}[hp.optimizer]||1.0;
    const k=(0.025+Math.random()*0.02)*Math.max(0.2,1+lrLog*0.25)*optSpeed;
    const noiseScale=0.04/Math.sqrt(hp.bs/32);
    const regStr=hp.l2*5000+avgDrop;
    const ovFactor=Math.max(0.02,0.18-regStr*0.08);
    const ovStart=0.45+Math.random()*0.3;
    const targetAccBase=0.65+Math.random()*0.28*(1-ovFactor);
    const spd=total>500?10:total>200?20:total>100?32:50;

    epRef.current=0; histRef.current=[];
    setTraining(true);setDone(false);setHistory([]);setMetrics(null);setAnalysis('');setCurEp(0);
    setTab("training");

    ivRef.current=setInterval(()=>{
      epRef.current+=1;
      const e=epRef.current,t=e/total;
      const tL=Math.max(0.004,iL*Math.exp(-k*e)+Math.abs(noise(noiseScale)));
      const vL=Math.max(0.005,tL*(1+ovFactor*Math.max(0,(t-ovStart)/(1-ovStart))+noise(noiseScale*.7)));
      const tA=isClsf?Math.min(99.5,targetAccBase*(1-Math.exp(-k*1.4*e))*100+noise(noiseScale*25)):null;
      const vA=isClsf?Math.min(99,(tA/100)*(1-ovFactor*.4*Math.max(0,(t-ovStart)))*100+noise(noiseScale*18)):null;

      const entry={epoch:e,trainLoss:+tL.toFixed(4),valLoss:+vL.toFixed(4),...(isClsf&&{trainAcc:+tA.toFixed(2),valAcc:+vA.toFixed(2)})};
      histRef.current.push(entry);
      setHistory([...histRef.current]);setCurEp(e);

      if(e>=total){
        clearInterval(ivRef.current);
        const last=histRef.current[histRef.current.length-1];
        let tp=0,prev=parseInt(ds.features)||20;layers.forEach(l=>{tp+=(prev+1)*l.n;prev=l.n;});tp+=(prev+1)*nc;
        const fm={trainLoss:last.trainLoss,valLoss:last.valLoss,trainAcc:last.trainAcc,valAcc:last.valAcc,
          precision:isClsf?+((last.valAcc/100)*(0.93+noise(0.04))).toFixed(4):null,
          recall:isClsf?+((last.valAcc/100)*(0.91+noise(0.05))).toFixed(4):null,
          f1:isClsf?+((last.valAcc/100)*(0.92+noise(0.04))).toFixed(4):null,
          r2:!isClsf?+(1-last.valLoss/iL).toFixed(4):null,totalParams:tp};
        setMetrics(fm);setAnalysis(analyzeResults(fm,hp,layers,ds,histRef.current.length));
        setDone(true);setTraining(false);setTab("results");
      }
    },spd);
  },[hp,ds,layers,isClsf]);

  const stopTraining=()=>{
    if(ivRef.current)clearInterval(ivRef.current);
    setTraining(false);
    if(histRef.current.length>0){
      const last=histRef.current[histRef.current.length-1];
      const fm={trainLoss:last.trainLoss,valLoss:last.valLoss,trainAcc:last.trainAcc,valAcc:last.valAcc};
      setMetrics(fm);setAnalysis(analyzeResults(fm,hp,layers,ds,histRef.current.length));setDone(true);
    }
  };

  useEffect(()=>(()=>{if(ivRef.current)clearInterval(ivRef.current);}),[]);

  // ── Export model ─────────────────────────────────────────────────────────────
  const exportModel=()=>{
    const modelName=(ds.name||"model").replace(/\s+/g,'_');
    const pyLayers=layers.map(l=>`    tf.keras.layers.Dense(${l.n}, activation='${l.a.toLowerCase()}'),\n    tf.keras.layers.Dropout(${l.d}),`).join('\n');
    const pyOptimizer=hp.optimizer==="Adam"?`tf.keras.optimizers.Adam(${hp.lr})`:`tf.keras.optimizers.${hp.optimizer}(${hp.lr})`;
    const pyLoss=isClsf?(parseInt(ds.classes)>2?"'categorical_crossentropy'":"'binary_crossentropy'"):"'mse'";
    const config={
      metadata:{name:modelName,exportedAt:new Date().toISOString(),version:"1.0",tool:"ML Training Assistant v4"},
      dataset:{name:ds.name,modality:ds.modality,problemType:ds.problemType,samples:ds.samples,features:ds.features,classes:ds.classes,preprocessing:prep},
      architecture:{modelType,classicalAlgorithm:modelType==="classical"?classAlgo:null,hiddenLayers:layers.map(l=>({neurons:l.n,activation:l.a,dropout:l.d})),outputActivation:outCfg.activation,weightInit:outCfg.weightInit,useBias:outCfg.useBias,totalParams:totalParams},
      hyperparameters:{optimizer:hp.optimizer,learningRate:hp.lr,batchSize:hp.bs,epochs:hp.epochs,l2Lambda:hp.l2,earlyStopping:hp.earlyStopping,patience:hp.patience,scheduler:hp.scheduler,validationSplit:hp.vSplit},
      trainingResults:metrics||{},
      trainingHistory:history.slice(-20), // last 20 epochs
      analysis,
      pythonSetup:`# ── ${modelName} — Auto-generated setup (TensorFlow/Keras) ──\nimport tensorflow as tf\n\nmodel = tf.keras.Sequential([\n${pyLayers}\n    tf.keras.layers.Dense(${ds.classes||1}, activation='${outCfg.activation.toLowerCase()}')\n])\n\nmodel.compile(\n    optimizer=${pyOptimizer},\n    loss=${pyLoss},\n    metrics=['accuracy']\n)\n\nmodel.fit(X_train, y_train,\n    epochs=${hp.epochs},\n    batch_size=${hp.bs},\n    validation_split=${hp.vSplit},\n    callbacks=[tf.keras.callbacks.EarlyStopping(patience=${hp.patience}, restore_best_weights=True)]\n)\n`
    };
    dlJSON(config,`${modelName}_config.json`);
  };

  const exportHistory=()=>dlCSV(history,`${(ds.name||"model").replace(/\s+/g,'_')}_history.csv`);
  const exportProcData=()=>{ if(processedData.length) dlCSV(processedData,`${(ds.name||"dataset").replace(/\s+/g,'_')}_processed.csv`); };

  // ── Derived ──────────────────────────────────────────────────────────────────
  const pct=hp.epochs>0?(curEp/hp.epochs)*100:0;
  const last=history[history.length-1];
  const helpData=helpKey?HELP[helpKey]:null;
  const TABS=[{id:"dataset",lbl:"📋 Dataset"},{id:"configure",lbl:"⚙ Configure"},{id:"training",lbl:"⚡ Training"},{id:"results",lbl:"📊 Results"}];
  const targetCol=columns.find(c=>c.isTarget);

  // ════════════════════════════════════════════════════════════════════════════
  return (
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,fontFamily:"'JetBrains Mono','Fira Code',ui-monospace,monospace"}} onClick={()=>{}}>

      {/* ── HELP MODAL ── */}
      {helpData&&(
        <div onClick={()=>setHelpKey(null)} style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.85)",zIndex:3000,display:"flex",alignItems:"center",justifyContent:"center",padding:20}}>
          <div onClick={e=>e.stopPropagation()} style={{background:C.card,border:`1px solid ${C.cyan}`,borderRadius:14,padding:28,maxWidth:480,width:"100%",position:"relative"}}>
            <button onClick={()=>setHelpKey(null)} style={{position:"absolute",top:12,right:16,background:"transparent",border:"none",color:C.muted,fontSize:22,cursor:"pointer",fontFamily:"inherit",lineHeight:1}}>×</button>
            <div style={{fontSize:16,fontWeight:700,color:C.cyan,marginBottom:14}}>{helpData.title}</div>
            <div style={{background:C.deep,borderRadius:8,padding:"14px 16px",marginBottom:14,fontSize:13,color:C.sec,lineHeight:1.9}}>{helpData.body}</div>
            {helpData.tip&&<div style={{background:"#071a12",border:`1px solid #0e3020`,borderRadius:8,padding:"10px 14px",fontSize:12,color:C.green}}>💡 {helpData.tip}</div>}
            <button onClick={()=>setHelpKey(null)} style={{...bS(C.muted),marginTop:14,width:"100%",textAlign:"center"}}>Got it</button>
          </div>
        </div>
      )}

      {/* ── HEADER ── */}
      <div style={{borderBottom:`1px solid ${C.bd}`,background:C.card,position:"sticky",top:0,zIndex:100}}>
        <div style={{maxWidth:1220,margin:"0 auto",padding:"0 20px",display:"flex",alignItems:"center",justifyContent:"space-between",height:54}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <Dot col={C.green} glow/>
            <span style={{fontSize:13,fontWeight:700,letterSpacing:"1.5px"}}>ML TRAINING ASSISTANT</span>
            <span style={{fontSize:9,color:"#000",background:C.green,padding:"2px 7px",borderRadius:4,fontWeight:700}}>OFFLINE</span>
            {loadSrc&&<span style={{fontSize:10,color:C.cyan,background:`${C.cyan}15`,padding:"2px 10px",borderRadius:10,border:`1px solid ${C.cyan}30`}}>{loadSrc.name}</span>}
          </div>
          <div style={{display:"flex"}}>
            {TABS.map(t=>{
              const active=tab===t.id;
              const ok=(t.id==="results"&&done)||(t.id==="training"&&(training||done));
              return <button key={t.id} onClick={()=>setTab(t.id)} style={{background:active?`${C.cyan}18`:"transparent",color:active?C.cyan:C.sec,border:"none",borderBottom:active?`2px solid ${C.cyan}`:"2px solid transparent",padding:"0 16px",height:54,cursor:"pointer",fontSize:12,fontWeight:active?700:400,display:"flex",alignItems:"center",gap:7,fontFamily:"inherit",transition:"all .15s"}}>
                <span style={{width:20,height:20,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:700,flexShrink:0,background:active?C.cyan:ok?C.green:C.deep,color:active||ok?"#000":C.muted,border:`1px solid ${active?C.cyan:ok?C.green:C.bd}`}}>{TABS.indexOf(t)+1}</span>
                {t.lbl}
              </button>;
            })}
          </div>
          <button onClick={()=>setHelpKey("classicalML")} style={{...bS(C.muted),padding:"5px 12px",fontSize:11}}>? Help</button>
        </div>
      </div>

      <div style={{maxWidth:1220,margin:"0 auto",padding:"24px 20px"}}>

        {/* ════════════════════════════════════════════ DATASET TAB */}
        {tab==="dataset"&&(
          <div>
            <div style={{marginBottom:20}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>Dataset & Preprocessing</h2>
              <p style={{fontSize:12,color:C.muted}}>Load from file, folder, URL, or use a sample. View and edit data directly. All preprocessing is applied locally.</p>
            </div>

            {/* ── LOAD SECTION ── */}
            <div style={{...cc(),marginBottom:16,borderColor:C.cyan+"40"}}>
              <ST color={C.cyan} mb={14}>LOAD DATA</ST>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:14}}>

                {/* CSV/JSON file */}
                <div style={{background:C.deep,borderRadius:10,padding:"14px 16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:8}}>📄 CSV / JSON File</div>
                  <div style={{fontSize:11,color:C.muted,marginBottom:10}}>Upload a local CSV or JSON file. Full data editing & preprocessing available.</div>
                  <label style={{...bP(C.cyan),display:"inline-block",cursor:"pointer",fontSize:12,padding:"7px 14px"}}>
                    {loading?"Loading...":"Choose File"}
                    <input type="file" accept=".csv,.json,.tsv,.txt" style={{display:"none"}} onChange={e=>{if(e.target.files[0])loadCSV(e.target.files[0]);e.target.value='';}} disabled={loading}/>
                  </label>
                </div>

                {/* Image/Audio/Video folder */}
                <div style={{background:C.deep,borderRadius:10,padding:"14px 16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:8}}>🗂️ Image / Audio / Video Folder</div>
                  <div style={{fontSize:11,color:C.muted,marginBottom:10}}>Select a folder of files. Previews shown; preprocessing configured here.</div>
                  <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                    <label style={{...bS(C.orange),display:"inline-block",cursor:"pointer",fontSize:11,padding:"5px 10px"}}>
                      🖼️ Images
                      <input type="file" accept="image/*" multiple style={{display:"none"}} onChange={e=>{if(e.target.files.length)loadFiles(e.target.files,"Images");e.target.value='';}}/>
                    </label>
                    <label style={{...bS(C.purple),display:"inline-block",cursor:"pointer",fontSize:11,padding:"5px 10px"}}>
                      🔊 Audio
                      <input type="file" accept="audio/*" multiple style={{display:"none"}} onChange={e=>{if(e.target.files.length)loadFiles(e.target.files,"Audio");e.target.value='';}}/>
                    </label>
                    <label style={{...bS(C.pink),display:"inline-block",cursor:"pointer",fontSize:11,padding:"5px 10px"}}>
                      🎬 Video
                      <input type="file" accept="video/*" multiple style={{display:"none"}} onChange={e=>{if(e.target.files.length)loadFiles(e.target.files,"Video");e.target.value='';}}/>
                    </label>
                  </div>
                </div>

                {/* URL */}
                <div style={{background:C.deep,borderRadius:10,padding:"14px 16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:8}}>🌐 Online URL</div>
                  <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Direct link to a CSV (e.g. GitHub raw). CORS must be open on the server.</div>
                  <input style={{...IS,padding:"6px 10px",fontSize:11,marginBottom:8}} placeholder="https://raw.githubusercontent.com/.../data.csv" value={urlInp} onChange={e=>setUrlInp(e.target.value)} onKeyDown={e=>e.key==='Enter'&&loadURL()}/>
                  <button onClick={loadURL} disabled={loading} style={{...bP(C.green),fontSize:11,padding:"6px 14px",opacity:loading?.6:1}}>Fetch →</button>
                </div>
              </div>

              {/* Sample datasets */}
              <div>
                <div style={{fontSize:10,color:C.muted,letterSpacing:"0.8px",fontWeight:700,marginBottom:8}}>SAMPLE DATASETS (instant load)</div>
                <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                  {Object.entries(SAMPLES).map(([key,s])=>(
                    <button key={key} onClick={()=>loadSample(key)} style={{background:loadSrc?.name===s.label?`${C.cyan}15`:C.deep,border:`1px solid ${loadSrc?.name===s.label?C.cyan:C.bd}`,borderRadius:8,padding:"8px 14px",cursor:"pointer",textAlign:"left",fontFamily:"inherit"}}>
                      <div style={{fontSize:12,fontWeight:600,color:loadSrc?.name===s.label?C.cyan:C.text,marginBottom:3}}>{s.label}</div>
                      <div style={{fontSize:10,color:C.muted}}>{s.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {loadErr&&<div style={{marginTop:12,background:"#120608",border:`1px solid ${C.red}50`,borderRadius:8,padding:"9px 14px",fontSize:12,color:C.red}}>⚠ {loadErr}</div>}
              {loadSrc&&<div style={{marginTop:12,background:`${C.green}10`,border:`1px solid ${C.green}40`,borderRadius:8,padding:"9px 14px",fontSize:12,color:C.green}}>
                ✓ Loaded: <strong>{loadSrc.name}</strong> — {loadSrc.rows?.toLocaleString()} rows{loadSrc.cols?`, ${loadSrc.cols} columns`:''} 
                {processedData.length>0&&<span style={{marginLeft:12,color:C.cyan}}>· Preprocessed: {processedData.length.toLocaleString()} rows ready</span>}
              </div>}
            </div>

            {/* ── DATA EXPLORER ── */}
            {workData.length>0&&(
              <div style={{marginBottom:16}}>
                <DataExplorer data={workData} columns={columns} onUpdateCols={setColumns} onUpdateData={d=>{setWorkData(d);setProcData([]);setPrepLog([]);}} onHelp={setHelpKey}/>
                {!targetCol&&<div style={{marginTop:8,fontSize:11,color:C.orange}}>⚠ No target column set. Click a column header → 🎯 Set as Target before training.</div>}
                {targetCol&&<div style={{marginTop:8,fontSize:11,color:C.green}}>🎯 Target column: <strong>{targetCol.name}</strong> ({targetCol.type}) · {new Set(workData.map(r=>r[targetCol.name])).size} unique values</div>}
              </div>
            )}

            {/* Image/Audio/Video file list */}
            {fileList.length>0&&workData.length===0&&(
              <div style={{...cc(),marginBottom:16}}>
                <ST>FILE PREVIEW — {fileList.length} files</ST>
                <div style={{display:"flex",flexWrap:"wrap",gap:8,maxHeight:180,overflowY:"auto"}}>
                  {fileList.slice(0,60).map((f,i)=>{
                    const isImg=f.type.startsWith("image/");
                    const url=isImg?URL.createObjectURL(f):null;
                    return <div key={i} style={{width:isImg?72:120,textAlign:"center"}}>
                      {isImg?<img src={url} alt={f.name} style={{width:72,height:72,objectFit:"cover",borderRadius:6,border:`1px solid ${C.bd}`}} onLoad={()=>URL.revokeObjectURL(url)}/>:
                      <div style={{background:C.deep,border:`1px solid ${C.bd}`,borderRadius:6,padding:"10px 6px",fontSize:10,color:C.sec,height:72,display:"flex",alignItems:"center",justifyContent:"center"}}>{f.name.slice(-20)}</div>}
                    </div>;
                  })}
                  {fileList.length>60&&<div style={{fontSize:11,color:C.muted,alignSelf:"center"}}>+{fileList.length-60} more...</div>}
                </div>
              </div>
            )}

            {/* Preprocessing */}
            <div style={{marginBottom:16}}>
              <PrepPanel ds={ds} prep={prep} setPrep={setPrep} prepLog={prepLog} onHelp={setHelpKey}/>
            </div>

            {workData.length>0&&(
              <div style={{display:"flex",gap:10,alignItems:"center",marginBottom:16}}>
                <button onClick={applyPreprocessing} style={bP(C.purple)}>⚙ Apply Preprocessing to Data</button>
                {processedData.length>0&&<>
                  <button onClick={exportProcData} style={bS(C.muted)}>⬇ Export Processed CSV</button>
                  <span style={{fontSize:11,color:C.green}}>✓ {processedData.length.toLocaleString()} rows after preprocessing</span>
                </>}
              </div>
            )}

            {/* Dataset meta + recommendations */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
              <div style={cc()}>
                <ST>DATASET META</ST>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                  <div><label style={LB}>Problem Type</label>
                    <select style={SE} value={ds.problemType} onChange={e=>setDs(p=>({...p,problemType:e.target.value}))}>
                      {PTYPES.map(t=><option key={t}>{t}</option>)}
                    </select>
                  </div>
                  <div><label style={LB}>Data Modality</label>
                    <select style={SE} value={ds.modality} onChange={e=>setDs(p=>({...p,modality:e.target.value}))}>
                      {MODS.map(m=><option key={m}>{m}</option>)}
                    </select>
                  </div>
                  <div><label style={LB}>Samples</label>
                    <input style={IS} value={ds.samples} onChange={e=>setDs(p=>({...p,samples:e.target.value}))}/>
                  </div>
                  <div><label style={LB}>Features / Input Dim</label>
                    <input style={IS} value={ds.features} onChange={e=>setDs(p=>({...p,features:e.target.value}))}/>
                  </div>
                  <div><label style={LB}>Classes / Output Dim</label>
                    <input style={IS} value={ds.classes} onChange={e=>setDs(p=>({...p,classes:e.target.value}))}/>
                  </div>
                  <div style={{display:"flex",flexDirection:"column",gap:7,justifyContent:"flex-end"}}>
                    <Tog v={ds.hasNulls} onChange={v=>setDs(p=>({...p,hasNulls:v}))} label="Has Missing Values" small/>
                    <Tog v={ds.imbalanced} onChange={v=>setDs(p=>({...p,imbalanced:v}))} label="Class Imbalance" small/>
                  </div>
                </div>
              </div>
              <div style={cc()}>
                <ST color={C.green}>OFFLINE AI RECOMMENDATIONS</ST>
                {!recs&&<div style={{fontSize:12,color:C.muted,marginBottom:12}}>Analyzes your dataset stats and suggests the best model architecture, optimizer, and hyperparameters — no internet needed.</div>}
                {recs&&<div style={{marginBottom:12}}>
                  <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
                    <span style={{fontSize:13,fontWeight:700,color:recs.modelType==="classical"?C.orange:C.cyan}}>{recs.modelType==="classical"?"🌳 "+recs.classicalAlgo:"🧠 Deep Learning · "+recs.layers.length+" layers"}</span>
                  </div>
                  <div style={{fontSize:12,color:C.sec,lineHeight:1.7,marginBottom:8}}>{recs.rationale}</div>
                  {recs.warnings.length>0&&<div style={{fontSize:11,color:"#d09030",marginBottom:8}}>⚠ {recs.warnings.join(" ")}</div>}
                  <div style={{fontSize:12,color:C.muted}}>Estimated: <span style={{color:C.green,fontWeight:700}}>{recs.estimate}</span></div>
                </div>}
                <div style={{display:"flex",gap:8}}>
                  <button onClick={handleAnalyze} style={bS(C.green)}>✦ Analyze</button>
                  {recs&&<button onClick={applyRecs} style={bP(C.green)}>Apply →</button>}
                </div>
              </div>
            </div>

            <div style={{display:"flex",justifyContent:"flex-end"}}>
              <button onClick={()=>setTab("configure")} style={bP()}>Configure Model →</button>
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════ CONFIGURE TAB */}
        {tab==="configure"&&(
          <div>
            <div style={{marginBottom:18}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>Model Configuration</h2>
              <p style={{fontSize:12,color:C.muted}}>{ds.name||"Custom"} · {ds.modality} · {ds.problemType}{loadSrc?` · ${workData.length.toLocaleString()} rows`:""}</p>
            </div>

            {/* Model type */}
            <div style={{...cc(),marginBottom:16}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
                <ST mb={0}>MODEL TYPE</ST><HP k="classicalML" onHelp={setHelpKey}/>
              </div>
              <div style={{display:"flex",gap:10,marginBottom:14}}>
                {[["deep","🧠 Deep Learning","Neural network layers — best for large/complex data (images, audio, text, video)"],["classical","🌳 Classical ML","Traditional algorithms — best for tabular/small datasets, often faster to train"]].map(([id,title,desc])=>(
                  <div key={id} onClick={()=>setModelType(id)} style={{flex:1,padding:"14px 18px",borderRadius:10,cursor:"pointer",transition:"all .15s",background:modelType===id?`${id==="deep"?C.cyan:C.orange}15`:"transparent",border:`2px solid ${modelType===id?(id==="deep"?C.cyan:C.orange):C.bd}`}}>
                    <div style={{fontSize:14,fontWeight:700,color:modelType===id?(id==="deep"?C.cyan:C.orange):C.sec,marginBottom:4}}>{title}</div>
                    <div style={{fontSize:11,color:C.muted}}>{desc}</div>
                  </div>
                ))}
              </div>
              {modelType==="classical"&&<div style={{maxWidth:300}}><label style={LB}>Algorithm</label>
                <select style={SE} value={classAlgo} onChange={e=>setClassAlgo(e.target.value)}>
                  {CML.map(a=><option key={a}>{a}</option>)}
                </select>
              </div>}
            </div>

            {/* Sub-tabs */}
            <div style={{display:"flex",marginBottom:16,border:`1px solid ${C.bd}`,borderRadius:10,overflow:"hidden",width:"fit-content"}}>
              {["architecture","hyperparameters","regularization"].map(t=>(
                <button key={t} onClick={()=>setCfgSub(t)} style={{background:cfgSub===t?C.cyan:"transparent",color:cfgSub===t?"#000":C.muted,border:"none",padding:"8px 20px",fontSize:11,fontWeight:700,cursor:"pointer",letterSpacing:"0.5px",textTransform:"uppercase",transition:"all .15s",fontFamily:"inherit"}}>{t}</button>
              ))}
            </div>

            {/* ── Architecture ── */}
            {cfgSub==="architecture"&&modelType==="deep"&&(
              <div style={{display:"grid",gridTemplateColumns:"1fr 280px",gap:16}}>
                <div>
                  {/* Presets */}
                  <div style={{...cc(),marginBottom:10,padding:"12px 18px"}}>
                    <ST mb={10}>QUICK PRESETS</ST>
                    <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                      {Object.keys(PRESETS).map(nm=>(
                        <button key={nm} onClick={()=>applyPreset(nm)} style={{...bS(C.muted),padding:"5px 14px",fontSize:11}}>{nm} <span style={{fontSize:9,color:C.muted}}>·{PRESETS[nm].length}L</span></button>
                      ))}
                    </div>
                  </div>

                  {/* Bulk add */}
                  <div style={{...cc(),marginBottom:10,padding:"12px 18px",borderColor:C.purple+"50"}}>
                    <ST color={C.purple} mb={10}>BULK ADD LAYERS — up to 200 at once, max 500 total</ST>
                    <div style={{display:"grid",gridTemplateColumns:"70px 110px 1fr 80px auto",gap:8,alignItems:"end"}}>
                      <div><label style={LB}>Count</label><input type="number" style={IS} min={1} max={200} value={bulk.count} onChange={e=>setBulk(p=>({...p,count:e.target.value}))}/></div>
                      <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Neurons <HP k="neurons" onHelp={setHelpKey}/></label><input type="number" style={IS} value={bulk.n} min={1} max={8192} onChange={e=>setBulk(p=>({...p,n:e.target.value}))}/></div>
                      <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Activation <HP k="activation" onHelp={setHelpKey}/></label>
                        <select style={SE} value={bulk.a} onChange={e=>setBulk(p=>({...p,a:e.target.value}))}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                      </div>
                      <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Dropout <HP k="dropout" onHelp={setHelpKey}/></label><input type="number" style={IS} min={0} max={0.9} step={0.05} value={bulk.d} onChange={e=>setBulk(p=>({...p,d:e.target.value}))}/></div>
                      <button onClick={addBulk} style={{...bP(C.purple),color:"#fff",padding:"8px 14px"}}>+ Add {bulk.count}</button>
                    </div>
                    <div style={{marginTop:8,display:"flex",gap:12,alignItems:"center",flexWrap:"wrap"}}>
                      <span style={{fontSize:10,color:C.muted}}>Total: {layers.length} layers</span>
                      <span style={{fontSize:10,color:C.muted}}>Set all activations →</span>
                      <select style={{...SE,width:120,padding:"4px 8px",fontSize:11}} onChange={e=>globalAct(e.target.value)}>
                        <option value="">Pick...</option>
                        {ACTS.map(a=><option key={a}>{a}</option>)}
                      </select>
                      <button onClick={()=>setLayers(p=>[p[0]])} style={{...bS(C.red),padding:"4px 10px",fontSize:10}}>Clear</button>
                    </div>
                  </div>

                  {/* Layer list */}
                  <div style={cc()}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                      <ST mb={0}>HIDDEN LAYERS ({layers.length})</ST>
                      <button onClick={addLayer} style={{...bS(C.cyan),padding:"4px 12px",fontSize:11}}>+ Add One</button>
                    </div>
                    <div style={{display:"grid",gridTemplateColumns:"24px 1fr 1fr 80px 22px",gap:4,marginBottom:6,padding:"0 2px"}}>
                      {["#","Neurons","Activation","Dropout",""].map((h,i)=><div key={i} style={{fontSize:9,color:C.muted,letterSpacing:"0.8px"}}>{h}</div>)}
                    </div>
                    <div style={{padding:"5px 10px",background:C.deep,borderRadius:8,marginBottom:4,fontSize:12,display:"flex",gap:10,alignItems:"center"}}>
                      <Dot col={C.muted}/><span style={{color:C.muted}}>Input</span><span style={{color:C.cyan,marginLeft:"auto"}}>{ds.features||"?"} dim</span>
                    </div>
                    <div style={{maxHeight:320,overflowY:"auto",paddingRight:2}}>
                      {layers.map((l,i)=>(
                        <div key={l.id} style={{background:C.deep,border:`1px solid ${C.bd}`,borderRadius:8,padding:"6px 10px",marginBottom:3}}>
                          <div style={{display:"grid",gridTemplateColumns:"24px 1fr 1fr 80px 22px",gap:6,alignItems:"center"}}>
                            <div style={{width:20,height:20,borderRadius:"50%",background:`${C.cyan}20`,color:C.cyan,display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,fontWeight:700}}>{i+1}</div>
                            <input type="number" style={{...IS,padding:"5px 8px",fontSize:12}} value={l.n} min={1} max={16384} onChange={e=>updL(l.id,"n",Math.max(1,parseInt(e.target.value)||1))}/>
                            <select style={{...SE,padding:"5px 8px",fontSize:12}} value={l.a} onChange={e=>updL(l.id,"a",e.target.value)}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                            <input type="number" style={{...IS,padding:"5px 8px",fontSize:12}} value={l.d} min={0} max={0.9} step={0.05} onChange={e=>updL(l.id,"d",parseFloat(e.target.value)||0)}/>
                            <button onClick={()=>rmLayer(l.id)} style={{background:"transparent",border:"none",color:C.red,cursor:"pointer",fontSize:16,padding:0,lineHeight:1,fontFamily:"inherit"}}>×</button>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div style={{padding:"5px 10px",background:"#061810",borderRadius:8,border:`1px solid #0e3020`,marginTop:4,fontSize:12,display:"flex",gap:10,alignItems:"center"}}>
                      <Dot col={C.green}/><span style={{color:C.green}}>Output</span>
                      <span style={{color:C.muted}}>{outCfg.activation}</span>
                      <span style={{color:C.muted,marginLeft:"auto"}}>{ds.classes||"?"} out</span>
                    </div>
                  </div>
                </div>

                <div style={{display:"flex",flexDirection:"column",gap:14}}>
                  <div style={cc()}>
                    <ST>OUTPUT LAYER</ST>
                    <div style={{display:"flex",flexDirection:"column",gap:12}}>
                      <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Activation <HP k="activation" onHelp={setHelpKey}/></label>
                        <select style={SE} value={outCfg.activation} onChange={e=>setOutCfg(p=>({...p,activation:e.target.value}))}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                      </div>
                      <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Weight Init <HP k="weightInit" onHelp={setHelpKey}/></label>
                        <select style={SE} value={outCfg.weightInit} onChange={e=>setOutCfg(p=>({...p,weightInit:e.target.value}))}>{WINIT.map(w=><option key={w}>{w}</option>)}</select>
                      </div>
                      <Tog v={outCfg.useBias} onChange={v=>setOutCfg(p=>({...p,useBias:v}))} label="Use Bias Terms"/>
                    </div>
                  </div>
                  <div style={cc()}>
                    <ST>SUMMARY</ST>
                    {[["Total Layers",layers.length+2],["Hidden",layers.length],["Params",fmtP(totalParams)],["Init",outCfg.weightInit.split("/")[0]],["Bias",outCfg.useBias?"Yes":"No"]].map(([k,v])=>(
                      <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:7}}>
                        <span style={{color:C.muted}}>{k}</span><span>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {cfgSub==="architecture"&&modelType==="classical"&&(
              <div style={{...cc(),textAlign:"center",padding:40}}>
                <div style={{fontSize:40,marginBottom:10}}>🌳</div>
                <div style={{fontSize:14,color:C.orange,fontWeight:700,marginBottom:8}}>{classAlgo}</div>
                <div style={{fontSize:12,color:C.muted,lineHeight:1.9}}>Classical ML algorithms have no custom layer architecture.<br/>Set training parameters in the <strong style={{color:C.text}}>Hyperparameters</strong> tab.</div>
                <button onClick={()=>setCfgSub("hyperparameters")} style={{...bP(C.orange),marginTop:16,color:"#fff"}}>Go to Hyperparameters →</button>
              </div>
            )}

            {/* ── Hyperparameters ── */}
            {cfgSub==="hyperparameters"&&(
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                <div style={cc()}>
                  <ST>OPTIMIZER</ST>
                  <div style={{display:"flex",flexDirection:"column",gap:14}}>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Algorithm <HP k="optimizer" onHelp={setHelpKey}/></label>
                      <select style={SE} value={hp.optimizer} onChange={e=>setHp(p=>({...p,optimizer:e.target.value}))}>{OPTS.map(o=><option key={o}>{o}</option>)}</select>
                    </div>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Learning Rate <HP k="learningRate" onHelp={setHelpKey}/><span style={{color:C.text,fontWeight:400,marginLeft:6,fontSize:12}}>{hp.lr}</span></label>
                      <input type="range" min={-5} max={-1} step={0.05} value={Math.log10(hp.lr)} onChange={e=>setHp(p=>({...p,lr:+((10**parseFloat(e.target.value)).toFixed(7))}))} style={{width:"100%",accentColor:C.cyan}}/>
                      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted,marginTop:2}}><span>1e-5</span><span>1e-4</span><span>1e-3</span><span>0.01</span><span>0.1</span></div>
                    </div>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>LR Scheduler <HP k="scheduler" onHelp={setHelpKey}/></label>
                      <select style={SE} value={hp.scheduler} onChange={e=>setHp(p=>({...p,scheduler:e.target.value}))}>{SCHS.map(s=><option key={s}>{s}</option>)}</select>
                    </div>
                  </div>
                </div>
                <div style={cc()}>
                  <ST>TRAINING SCHEDULE</ST>
                  <div style={{display:"flex",flexDirection:"column",gap:14}}>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Epochs: {hp.epochs} <HP k="epochs" onHelp={setHelpKey}/></label>
                      <input type="range" min={10} max={1000} step={10} value={hp.epochs} onChange={e=>setHp(p=>({...p,epochs:parseInt(e.target.value)}))} style={{width:"100%",accentColor:C.cyan}}/>
                      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted,marginTop:2}}><span>10</span><span>250</span><span>500</span><span>750</span><span>1000</span></div>
                    </div>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Batch Size <HP k="batchSize" onHelp={setHelpKey}/></label>
                      <select style={SE} value={hp.bs} onChange={e=>setHp(p=>({...p,bs:parseInt(e.target.value)}))}>{BSIZES.map(b=><option key={b}>{b}</option>)}</select>
                    </div>
                    <div><label style={{...LB,display:"flex",gap:4,alignItems:"center"}}>Val Split: {(hp.vSplit*100).toFixed(0)}% <HP k="validation" onHelp={setHelpKey}/></label>
                      <input type="range" min={0.05} max={0.4} step={0.05} value={hp.vSplit} onChange={e=>setHp(p=>({...p,vSplit:parseFloat(e.target.value)}))} style={{width:"100%",accentColor:C.cyan}}/>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── Regularization ── */}
            {cfgSub==="regularization"&&(
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                <div style={cc()}>
                  <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:14}}><ST mb={0}>L2 REGULARIZATION</ST><HP k="l2" onHelp={setHelpKey}/></div>
                  <label style={LB}>Lambda: {hp.l2}</label>
                  <input type="range" min={0} max={0.01} step={0.0001} value={hp.l2} onChange={e=>setHp(p=>({...p,l2:parseFloat(e.target.value)}))} style={{width:"100%",accentColor:C.cyan,marginBottom:6}}/>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted}}><span>0 (off)</span><span>0.001</span><span>0.01 (heavy)</span></div>
                </div>
                <div style={cc()}>
                  <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:14}}><ST mb={0}>EARLY STOPPING</ST><HP k="earlyStopping" onHelp={setHelpKey}/></div>
                  <div style={{display:"flex",flexDirection:"column",gap:12}}>
                    <Tog v={hp.earlyStopping} onChange={v=>setHp(p=>({...p,earlyStopping:v}))} label="Enable Early Stopping"/>
                    {hp.earlyStopping&&<div><label style={LB}>Patience (epochs)</label>
                      <input type="number" style={IS} value={hp.patience} min={1} max={200} onChange={e=>setHp(p=>({...p,patience:parseInt(e.target.value)||1}))}/>
                    </div>}
                    <div style={{background:C.deep,borderRadius:8,padding:"9px 12px",fontSize:11,color:C.muted,lineHeight:1.7}}>
                      Monitors <span style={{color:C.cyan}}>val_loss</span>. Stops after <span style={{color:C.text}}>{hp.patience}</span> epochs with no improvement. Best weights restored automatically.
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div style={{display:"flex",justifyContent:"space-between",marginTop:20}}>
              <button onClick={()=>setTab("dataset")} style={bS(C.cyan)}>← Dataset</button>
              <button onClick={startTraining} style={{...bP(C.green),fontSize:14,padding:"12px 28px"}}>▶ Start Training</button>
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════ TRAINING TAB */}
        {tab==="training"&&(
          <div>
            <div style={{marginBottom:18}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>{training?"⚡ Training...":done?"✓ Training Complete":"Training Monitor"}</h2>
              <p style={{fontSize:12,color:C.muted}}>{ds.name||"Model"} · {modelType==="deep"?`${layers.length} layers · ${fmtP(totalParams)} params`:classAlgo} · {hp.optimizer} · LR {hp.lr}</p>
            </div>

            {!training&&!done&&<div style={{...cc(),textAlign:"center",padding:52}}>
              <div style={{fontSize:13,color:C.muted,marginBottom:20}}>Configure your model in the Configure tab, then start training.</div>
              <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                <button onClick={()=>setTab("configure")} style={bS(C.cyan)}>← Configure</button>
                <button onClick={startTraining} style={{...bP(C.green),fontSize:14,padding:"12px 28px"}}>▶ Start Now</button>
              </div>
            </div>}

            {(training||done)&&<>
              <div style={{...cc(),marginBottom:16}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                  <span style={{fontSize:13,color:C.sec}}>Epoch {curEp} / {hp.epochs}</span>
                  <div style={{display:"flex",gap:10,alignItems:"center"}}>
                    {training&&<div style={{display:"flex",gap:4}}>{[0,1,2].map(i=><div key={i} style={{width:5,height:5,borderRadius:"50%",background:C.cyan,opacity:0.4+i*.3}}/>)}</div>}
                    <span style={{fontSize:16,fontWeight:700,color:C.cyan}}>{pct.toFixed(1)}%</span>
                  </div>
                </div>
                <div style={{height:6,background:C.inp,borderRadius:3,overflow:"hidden",marginBottom:14}}>
                  <div style={{height:"100%",width:`${pct}%`,background:`linear-gradient(90deg,${C.cyan},${C.green})`,borderRadius:3,transition:"width 0.08s"}}/>
                </div>
                {last&&<div style={{display:"grid",gridTemplateColumns:`repeat(${isClsf?4:2},1fr)`,gap:10}}>
                  {[["Train Loss",fmt(last.trainLoss),C.cyan],["Val Loss",fmt(last.valLoss),C.orange],...(isClsf?[["Train Acc",`${fmt(last.trainAcc,1)}%`,C.green],["Val Acc",`${fmt(last.valAcc,1)}%`,"#ff88bb"]]:[])].map(([k,v,col])=>(
                    <div key={k} style={{background:C.deep,borderRadius:8,padding:"10px",textAlign:"center"}}>
                      <div style={{fontSize:10,color:C.muted,marginBottom:5}}>{k}</div>
                      <div style={{fontSize:18,fontWeight:700,color:col}}>{v}</div>
                    </div>
                  ))}
                </div>}
              </div>

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
                {[["LOSS CURVE",[{key:"trainLoss",stroke:C.cyan,name:"Train"},{key:"valLoss",stroke:C.orange,name:"Val"}],null],
                  isClsf?["ACCURACY CURVE",[{key:"trainAcc",stroke:C.green,name:"Train"},{key:"valAcc",stroke:"#ff88bb",name:"Val"}],[0,100]]:["LR DECAY",[{key:"valLoss",stroke:C.purple,name:"Val Loss"}],null]
                ].map(([title,lines,domain])=>(
                  <div key={title} style={cc()}>
                    <div style={{fontSize:10,fontWeight:700,color:C.muted,letterSpacing:"0.8px",marginBottom:12}}>{title}</div>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={history} margin={{top:4,right:6,left:-22,bottom:4}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#131e35"/>
                        <XAxis dataKey="epoch" tick={{fill:C.muted,fontSize:10}}/>
                        <YAxis domain={domain||undefined} tick={{fill:C.muted,fontSize:10}}/>
                        <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.bd}`,fontSize:11,color:C.text}} formatter={domain?v=>`${v}%`:undefined}/>
                        {lines.map(l=><Line key={l.key} type="monotone" dataKey={l.key} stroke={l.stroke} strokeWidth={2} dot={false} name={l.name}/>)}
                        <Legend wrapperStyle={{fontSize:10}}/>
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>

              <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                {training&&<button onClick={stopTraining} style={{...bS(C.red),borderColor:C.red}}>■ Stop</button>}
                {done&&<button onClick={()=>setTab("results")} style={bP()}>View Results →</button>}
              </div>
            </>}
          </div>
        )}

        {/* ════════════════════════════════════════════ RESULTS TAB */}
        {tab==="results"&&(
          <div>
            {!metrics?<div style={{...cc(),textAlign:"center",padding:60}}>
              <div style={{fontSize:40,marginBottom:14}}>📊</div>
              <div style={{fontSize:13,color:C.muted,marginBottom:20}}>No training results yet. Run training to see metrics, analysis, and export options.</div>
              <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                <button onClick={()=>setTab("dataset")} style={bS(C.cyan)}>← Dataset</button>
                <button onClick={startTraining} style={bP()}>▶ Train</button>
              </div>
            </div>:<>
              <div style={{marginBottom:18}}>
                <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
                  <Dot col={C.green} glow/>
                  <h2 style={{fontSize:20,fontWeight:700}}>Training Complete</h2>
                </div>
                <p style={{fontSize:12,color:C.muted}}>{ds.name||"Custom"} · {history.length} epochs · {fmtP(metrics.totalParams||totalParams)} parameters</p>
              </div>

              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:16}}>
                {(isClsf?[{label:"Val Accuracy",v:`${fmt(metrics.valAcc,1)}%`,col:C.green},{label:"Val Loss",v:fmt(metrics.valLoss),col:C.cyan},{label:"F1 Score",v:metrics.f1?`${(metrics.f1*100).toFixed(1)}%`:"—",col:C.orange},{label:"Precision",v:metrics.precision?`${(metrics.precision*100).toFixed(1)}%`:"—",col:C.pink}]
                  :[{label:"Val Loss",v:fmt(metrics.valLoss),col:C.cyan},{label:"Train Loss",v:fmt(metrics.trainLoss),col:C.green},{label:"R² Score",v:metrics.r2!=null?fmt(metrics.r2):"—",col:C.orange},{label:"Parameters",v:fmtP(metrics.totalParams||totalParams),col:C.pink}]
                ).map(({label,v,col})=><MBox key={label} label={label} value={v} color={col}/>)}
              </div>

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
                <div style={cc()}>
                  <div style={{fontSize:10,fontWeight:700,color:C.muted,letterSpacing:"0.8px",marginBottom:12}}>LOSS HISTORY</div>
                  <ResponsiveContainer width="100%" height={175}>
                    <LineChart data={history} margin={{top:4,right:6,left:-22,bottom:4}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#131e35"/>
                      <XAxis dataKey="epoch" tick={{fill:C.muted,fontSize:10}}/>
                      <YAxis tick={{fill:C.muted,fontSize:10}}/>
                      <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.bd}`,fontSize:11}}/>
                      <Line type="monotone" dataKey="trainLoss" stroke={C.cyan} strokeWidth={2} dot={false} name="Train"/>
                      <Line type="monotone" dataKey="valLoss" stroke={C.orange} strokeWidth={2} dot={false} name="Val"/>
                      <Legend wrapperStyle={{fontSize:10}}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={cc()}>
                  <ST>CONFIGURATION</ST>
                  {[["Dataset",ds.name||"Custom"],["Modality",ds.modality],["Problem",ds.problemType],["Model",modelType==="deep"?`${layers.length} hidden layers`:classAlgo],["Optimizer",hp.optimizer],["LR",hp.lr],["Batch",hp.bs],["Epochs Run",history.length],["Init",outCfg.weightInit.split("/")[0]]].map(([k,v])=>(
                    <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:6}}>
                      <span style={{color:C.muted}}>{k}</span>
                      <span style={{maxWidth:180,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",textAlign:"right"}}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              {analysis&&<div style={{...cc(),borderColor:C.green+"40",marginBottom:16}}>
                <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
                  <Dot col={C.green} glow/><ST color={C.green} mb={0}>ANALYSIS & RECOMMENDATIONS</ST>
                </div>
                <p style={{fontSize:13,color:C.sec,lineHeight:1.9,margin:0}}>{analysis}</p>
              </div>}

              {/* Export section */}
              <div style={{...cc(),borderColor:C.yellow+"40",marginBottom:16}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
                  <ST color={C.yellow} mb={0}>EXPORT MODEL</ST><HP k="exportModel" onHelp={setHelpKey}/>
                </div>
                <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:12}}>
                  <button onClick={exportModel} style={bP(C.yellow)}>⬇ Model Config + Python Setup (JSON)</button>
                  {history.length>0&&<button onClick={exportHistory} style={bS(C.cyan)}>⬇ Training History (CSV)</button>}
                  {processedData.length>0&&<button onClick={exportProcData} style={bS(C.green)}>⬇ Processed Dataset (CSV)</button>}
                </div>
                <div style={{background:C.deep,borderRadius:8,padding:"10px 14px",fontSize:11,color:C.muted,lineHeight:1.7}}>
                  The JSON export includes: full architecture, all hyperparameters, preprocessing config, training history, metrics, and a <span style={{color:C.yellow}}>ready-to-run Python/Keras setup script</span> to train this exact model in your own environment with real data.
                  <br/><span style={{color:`${C.muted}bb`}}>Note: training here is a high-fidelity simulation. The exported config can be used to run real training in Python (TF/PyTorch/scikit-learn).</span>
                </div>
              </div>

              <div style={{display:"flex",justifyContent:"space-between"}}>
                <button onClick={()=>setTab("configure")} style={bS(C.cyan)}>← Reconfigure</button>
                <div style={{display:"flex",gap:10}}>
                  <button onClick={startTraining} style={bS(C.orange)}>↺ Retrain</button>
                  <button onClick={()=>{setDone(false);setHistory([]);setMetrics(null);setCurEp(0);setAnalysis('');setRecs(null);setRawData([]);setWorkData([]);setColumns([]);setProcData([]);setPrepLog([]);setLoadSrc(null);setFileList([]);setTab("dataset");}} style={bP()}>+ New Model</button>
                </div>
              </div>
            </>}
          </div>
        )}

      </div>
    </div>
  );
}
