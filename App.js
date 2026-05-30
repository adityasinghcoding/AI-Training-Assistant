import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, BarChart, Bar
} from "recharts";

// ─── Config ──────────────────────────────────────────────────────────────────
const API = "http://localhost:5000/api";

// ─── Palette & Styles ────────────────────────────────────────────────────────
const C = {
  bg:"#070b14", card:"#0c1120", inp:"#080d1a", deep:"#060910",
  bd:"#192035", cyan:"#00c8f0", green:"#00e87a", orange:"#ff6b35",
  pink:"#ff4499", purple:"#a855f7", yellow:"#f5c842", red:"#ff3355",
  text:"#dde6f5", muted:"#3d5070", sec:"#7a90b0",
};
const cc  = (x={}) => ({ background:C.card, border:`1px solid ${C.bd}`, borderRadius:12, padding:"18px 22px", ...x });
const IS  = { background:C.inp, border:`1px solid ${C.bd}`, borderRadius:8, color:C.text, padding:"8px 12px", fontSize:13, width:"100%", outline:"none", fontFamily:"inherit", boxSizing:"border-box" };
const SE  = { ...IS, cursor:"pointer" };
const LB  = { color:C.muted, fontSize:10, fontWeight:700, letterSpacing:"1px", textTransform:"uppercase", display:"block", marginBottom:6 };
const bP  = (col=C.cyan) => ({ background:`linear-gradient(135deg,${col},${col}cc)`, color:"#000", border:"none", borderRadius:8, padding:"9px 20px", fontSize:13, fontWeight:700, cursor:"pointer", fontFamily:"inherit", whiteSpace:"nowrap" });
const bS  = (col=C.cyan) => ({ background:"transparent", color:col, border:`1px solid ${col}`, borderRadius:8, padding:"8px 16px", fontSize:12, fontWeight:600, cursor:"pointer", fontFamily:"inherit", whiteSpace:"nowrap" });

// ─── Constants ────────────────────────────────────────────────────────────────
const ACTS   = ["ReLU","GELU","Swish","Sigmoid","Tanh","LeakyReLU","ELU","Softmax","Linear","PReLU"];
const OPTS   = ["Adam","AdamW","SGD","RMSprop","Adagrad","Nadam","Adadelta"];
const SCHS   = ["None","ReduceOnPlateau","CosineAnnealing","StepDecay","CyclicLR","WarmupLinear"];
const BSIZES = [8,16,32,64,128,256,512,1024];
const WINIT  = ["Xavier/Glorot","He Normal","He Uniform","LeCun Normal","Orthogonal","Random Normal"];
const PTYPES = ["Classification","Regression","Clustering","Anomaly Detection","Time Series","NLP","Computer Vision","Recommendation"];
const CML    = ["Random Forest","Gradient Boosting","XGBoost","SVM","KNN","Logistic Regression","Decision Tree","Naive Bayes","Ridge Regression","Lasso","ElasticNet","K-Means","DBSCAN","Isolation Forest","Linear Regression","SVR","AdaBoost"];
const PRESETS= {
  Tiny:  [{n:32, a:"ReLU",d:0.1}],
  Small: [{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
  Medium:[{n:256,a:"ReLU",d:0.3},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
  Large: [{n:512,a:"GELU",d:0.3},{n:256,a:"GELU",d:0.2},{n:128,a:"ReLU",d:0.2},{n:64,a:"ReLU",d:0.1}],
  XLarge:[{n:1024,a:"GELU",d:0.4},{n:512,a:"GELU",d:0.3},{n:256,a:"ReLU",d:0.25},{n:128,a:"ReLU",d:0.1},{n:64,a:"ReLU",d:0.1}],
};
const SAMPLES=[
  {key:"iris",   label:"🌸 Iris",         desc:"150 rows · 4 features · 3 species"},
  {key:"wine",   label:"🍷 Wine Quality",  desc:"178 rows · 13 features · 3 classes"},
  {key:"diabetes",label:"🏥 Diabetes",     desc:"442 rows · 10 features · regression"},
  {key:"breast_cancer",label:"🔬 Breast Cancer",desc:"569 rows · 30 features · binary"},
];

// ─── Shared components ────────────────────────────────────────────────────────
const Dot = ({col,glow}) => <div style={{width:8,height:8,borderRadius:"50%",background:col,boxShadow:glow?`0 0 10px ${col}`:undefined,flexShrink:0}}/>;
const Tog = ({v,onChange,label,small}) => (
  <label style={{display:"flex",alignItems:"center",gap:8,cursor:"pointer",userSelect:"none"}}>
    <div onClick={()=>onChange(!v)} style={{width:small?16:18,height:small?16:18,borderRadius:4,flexShrink:0,background:v?C.cyan:"transparent",border:`2px solid ${v?C.cyan:C.bd}`,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",transition:"all .15s"}}>
      {v&&<span style={{color:"#000",fontSize:small?8:10,fontWeight:800}}>✓</span>}
    </div>
    <span style={{fontSize:small?11:13,color:C.sec}}>{label}</span>
  </label>
);
const ST = ({color=C.cyan,children,mb=14}) => <div style={{fontSize:11,fontWeight:700,color,letterSpacing:"0.8px",marginBottom:mb}}>{children}</div>;
const MBox = ({label,value,color}) => (
  <div style={{...cc(),textAlign:"center",borderColor:color+"50",padding:"14px 10px"}}>
    <div style={{fontSize:10,color:C.muted,letterSpacing:"0.8px",marginBottom:6}}>{label}</div>
    <div style={{fontSize:20,fontWeight:700,color}}>{value}</div>
  </div>
);
const TC = {numeric:C.cyan,categorical:C.orange,text:C.sec,unknown:C.muted};
const TBadge = ({type}) => <span style={{fontSize:8,fontWeight:700,padding:"1px 5px",borderRadius:3,background:`${TC[type]||C.muted}22`,color:TC[type]||C.muted,letterSpacing:"0.5px"}}>{type||"?"}</span>;

// ─── Data Explorer ────────────────────────────────────────────────────────────
function DataExplorer({ sid, columns, totalRows, onColsChange, onRowsChange }) {
  const [page,       setPage]       = useState(0);
  const [rows,       setRows]       = useState([]);
  const [loading,    setLoading]    = useState(false);
  const [selRows,    setSelRows]    = useState(new Set());
  const [filterCol,  setFilterCol]  = useState("");
  const [filterVal,  setFilterVal]  = useState("");
  const [sortCol,    setSortCol]    = useState("");
  const [sortDir,    setSortDir]    = useState("asc");
  const [activeMenu, setActiveMenu] = useState(null);
  const [renameFrom, setRenameFrom] = useState(null);
  const [renameTo,   setRenameTo]   = useState("");
  const [statCol,    setStatCol]    = useState(null);
  const [statData,   setStatData]   = useState(null);
  const [totalFiltered, setTotalFiltered] = useState(totalRows);
  const LIMIT = 50;
  const visCols = columns.filter(c => !c.dropped);
  const pages = Math.max(1, Math.ceil(totalFiltered / LIMIT));

  const fetchRows = useCallback(async () => {
    if (!sid) return;
    setLoading(true);
    const params = new URLSearchParams({ sid, page, limit: LIMIT });
    if (filterCol && filterVal) { params.set("filterCol", filterCol); params.set("filterVal", filterVal); }
    if (sortCol) { params.set("sortCol", sortCol); params.set("sortDir", sortDir); }
    try {
      const res = await fetch(`${API}/dataset/preview?${params}`);
      const d   = await res.json();
      setRows(d.rows || []);
      setTotalFiltered(d.total || 0);
    } catch { setRows([]); }
    setLoading(false);
  }, [sid, page, filterCol, filterVal, sortCol, sortDir]);

  useEffect(() => { fetchRows(); }, [fetchRows]);

  const fetchStat = async col => {
    setStatCol(col); setStatData(null);
    try {
      const res = await fetch(`${API}/dataset/stats/${encodeURIComponent(col)}?sid=${sid}`);
      setStatData(await res.json());
    } catch { setStatData(null); }
  };

  const doDropCol = async col => {
    await fetch(`${API}/column/drop`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,col}) });
    onColsChange(columns.map(c => c.name===col ? {...c,dropped:true} : c));
    if (statCol===col) setStatCol(null);
    setActiveMenu(null);
  };
  const doSetTarget = async col => {
    await fetch(`${API}/column/target`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,col}) });
    onColsChange(columns.map(c => ({...c, isTarget: c.name===col})));
    setActiveMenu(null);
  };
  const doRename = async () => {
    if (!renameTo.trim() || renameTo===renameFrom) { setRenameFrom(null); return; }
    await fetch(`${API}/column/rename`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,old:renameFrom,new:renameTo}) });
    onColsChange(columns.map(c => c.name===renameFrom ? {...c,name:renameTo} : c));
    setRenameFrom(null); setRenameTo(""); fetchRows();
  };
  const doFillNull = async col => {
    await fetch(`${API}/column/fillnull`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,col,strategy:"mean"}) });
    fetchRows(); setActiveMenu(null);
  };
  const doDedup = async () => {
    const res = await fetch(`${API}/rows/dedup`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid}) });
    const d   = await res.json();
    onRowsChange(d.rows);
    fetchRows();
  };
  const doDelSelected = async () => {
    if (!selRows.size) return;
    const indices = [...selRows].map(i => page * LIMIT + i);
    const res = await fetch(`${API}/rows/delete`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,indices}) });
    const d   = await res.json();
    onRowsChange(d.rows); setSelRows(new Set()); fetchRows();
  };
  const doChangeType = async (col, dtype) => {
    await fetch(`${API}/column/type`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sid,col,dtype}) });
    onColsChange(columns.map(c => c.name===col ? {...c,type:dtype} : c));
    setActiveMenu(null);
  };
  const doSort = col => {
    if (sortCol===col) setSortDir(d => d==="asc"?"desc":"asc"); else { setSortCol(col); setSortDir("asc"); }
    setPage(0); setActiveMenu(null);
  };
  const toggleRow = i => { const s=new Set(selRows); s.has(i)?s.delete(i):s.add(i); setSelRows(s); };
  const selectAll = () => selRows.size===rows.length ? setSelRows(new Set()) : setSelRows(new Set(rows.map((_,i)=>i)));

  if (!sid) return null;

  return (
    <div style={{...cc(),padding:0,overflow:"hidden"}}>
      {/* Toolbar */}
      <div style={{padding:"10px 16px",borderBottom:`1px solid ${C.bd}`,display:"flex",alignItems:"center",gap:10,flexWrap:"wrap",background:C.deep}}>
        <span style={{fontSize:11,fontWeight:700,color:C.cyan,letterSpacing:"0.5px"}}>DATA EXPLORER</span>
        <span style={{fontSize:11,color:C.muted}}>· {totalFiltered.toLocaleString()} rows · {visCols.length} cols</span>
        <select style={{...SE,width:130,padding:"5px 8px",fontSize:11}} value={filterCol} onChange={e=>setFilterCol(e.target.value)}>
          <option value="">Filter column...</option>
          {visCols.map(c=><option key={c.name} value={c.name}>{c.name}</option>)}
        </select>
        <input style={{...IS,width:130,padding:"5px 8px",fontSize:11}} placeholder="Filter value..." value={filterVal} onChange={e=>{setFilterVal(e.target.value);setPage(0);}}/>
        {filterVal && <button onClick={()=>{setFilterVal("");setFilterCol("");setPage(0);}} style={{...bS(C.muted),padding:"4px 10px",fontSize:10}}>✕</button>}
        <div style={{marginLeft:"auto",display:"flex",gap:8}}>
          {selRows.size>0 && <button onClick={doDelSelected} style={{...bS(C.red),padding:"5px 10px",fontSize:11}}>🗑 Delete {selRows.size}</button>}
          <button onClick={doDedup} style={{...bS(C.muted),padding:"5px 10px",fontSize:11}}>Remove Dupes</button>
          <button onClick={fetchRows} style={{...bS(C.muted),padding:"5px 10px",fontSize:11}}>↺ Refresh</button>
        </div>
      </div>

      <div style={{display:"flex"}}>
        {/* Table */}
        <div style={{flex:1,overflow:"auto",maxHeight:400}} onClick={()=>setActiveMenu(null)}>
          {loading && <div style={{padding:20,textAlign:"center",fontSize:12,color:C.muted}}>Loading...</div>}
          {!loading && (
            <table style={{borderCollapse:"collapse",fontSize:12,width:"100%",minWidth:visCols.length*110}}>
              <thead>
                <tr style={{background:C.deep,position:"sticky",top:0,zIndex:10}}>
                  <th style={{width:28,padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,cursor:"pointer"}} onClick={selectAll}>
                    <div style={{width:14,height:14,borderRadius:3,border:`1px solid ${C.bd}`,background:selRows.size===rows.length&&rows.length>0?C.cyan:"transparent",display:"flex",alignItems:"center",justifyContent:"center"}}>
                      {selRows.size===rows.length&&rows.length>0&&<span style={{fontSize:8,color:"#000",fontWeight:800}}>✓</span>}
                    </div>
                  </th>
                  <th style={{padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,color:C.muted,fontSize:10,textAlign:"left",minWidth:36}}>#</th>
                  {visCols.map(col => (
                    <th key={col.name} style={{padding:"6px 8px",borderBottom:`1px solid ${C.bd}`,textAlign:"left",minWidth:110,background:col.isTarget?`${C.green}10`:undefined,position:"relative"}}>
                      <div style={{display:"flex",alignItems:"center",gap:5}}>
                        <button onClick={e=>{e.stopPropagation();setActiveMenu(activeMenu===col.name?null:col.name);}} style={{background:"transparent",border:"none",cursor:"pointer",color:col.isTarget?C.green:C.text,fontSize:11,fontWeight:700,padding:0,fontFamily:"inherit",display:"flex",alignItems:"center",gap:4}}>
                          {col.isTarget&&"🎯"}{col.name} ▾
                        </button>
                        <TBadge type={col.type}/>
                        <button onClick={()=>fetchStat(col.name)} style={{background:"transparent",border:"none",cursor:"pointer",color:statCol===col.name?C.cyan:C.muted,fontSize:10,padding:0,fontFamily:"inherit"}}>📊</button>
                        {sortCol===col.name&&<span style={{fontSize:10,color:C.cyan}}>{sortDir==="asc"?"↑":"↓"}</span>}
                      </div>
                      {activeMenu===col.name && (
                        <div onClick={e=>e.stopPropagation()} style={{position:"absolute",top:"100%",left:0,background:C.card,border:`1px solid ${C.bd}`,borderRadius:8,padding:"5px 0",zIndex:100,minWidth:190,boxShadow:"0 8px 24px rgba(0,0,0,0.6)"}}>
                          {[
                            {l:"🎯 Set as Target",fn:()=>doSetTarget(col.name),col:C.green},
                            {l:`📊 View Statistics`,fn:()=>{fetchStat(col.name);setActiveMenu(null);}},
                            {l:`${sortDir==="asc"?"↑":"↓"} Sort`,fn:()=>doSort(col.name)},
                            {l:"✏️ Rename",fn:()=>{setRenameFrom(col.name);setRenameTo(col.name);setActiveMenu(null);}},
                            {l:"🔧 Fill Nulls (mean)",fn:()=>doFillNull(col.name)},
                            {l:"→ Cast to numeric",fn:()=>doChangeType(col.name,"numeric"),col:C.muted},
                            {l:"→ Cast to categorical",fn:()=>doChangeType(col.name,"categorical"),col:C.muted},
                            {l:"🗑 Remove Column",fn:()=>doDropCol(col.name),col:C.red},
                          ].map((item,idx)=>(
                            <button key={idx} onClick={item.fn} style={{display:"block",width:"100%",background:"transparent",border:"none",padding:"7px 14px",textAlign:"left",cursor:"pointer",fontSize:12,color:item.col||C.text,fontFamily:"inherit"}}>
                              {item.l}
                            </button>
                          ))}
                        </div>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row,i)=>{
                  const isSel=selRows.has(i);
                  return (
                    <tr key={i} style={{background:isSel?`${C.cyan}10`:i%2===0?C.deep:"transparent"}}>
                      <td style={{padding:"4px 8px",borderBottom:`1px solid ${C.bd}20`}}>
                        <div onClick={()=>toggleRow(i)} style={{width:14,height:14,borderRadius:3,border:`1px solid ${C.bd}`,background:isSel?C.cyan:"transparent",cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}>
                          {isSel&&<span style={{fontSize:8,color:"#000",fontWeight:800}}>✓</span>}
                        </div>
                      </td>
                      <td style={{padding:"4px 8px",borderBottom:`1px solid ${C.bd}20`,color:C.muted,fontSize:10}}>{page*LIMIT+i+1}</td>
                      {visCols.map(col=>{
                        const val=row[col.name];
                        const isNull=val===null||val===undefined;
                        return (
                          <td key={col.name} style={{padding:"4px 10px",borderBottom:`1px solid ${C.bd}20`,maxWidth:200,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",color:isNull?C.muted:col.isTarget?C.green:C.text,background:col.isTarget?`${C.green}08`:undefined,fontSize:12}}>
                            {isNull?<span style={{fontStyle:"italic",fontSize:10}}>null</span>:String(val)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Stats panel */}
        {statCol && statData && (
          <div style={{width:210,borderLeft:`1px solid ${C.bd}`,padding:"14px 14px",background:C.deep,flexShrink:0,overflow:"auto",maxHeight:400}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
              <span style={{fontSize:11,fontWeight:700,color:C.cyan}}>{statCol}</span>
              <button onClick={()=>setStatCol(null)} style={{background:"transparent",border:"none",color:C.muted,cursor:"pointer",fontSize:16,padding:0,lineHeight:1}}>×</button>
            </div>
            <TBadge type={statData.type}/>
            <div style={{marginTop:10,display:"flex",flexDirection:"column",gap:5}}>
              {[["Nulls",statData.nulls],["Unique",statData.unique],
                ...(statData.type==="numeric"?[["Min",statData.min],["Max",statData.max],["Mean",statData.mean],["Std",statData.std],["Median",statData.median],["Q1",statData.q1],["Q3",statData.q3]]:
                   (statData.top||[]).map(([v,c])=>[String(v).slice(0,14),`${c}×`]))
              ].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:11}}>
                  <span style={{color:C.muted}}>{k}</span>
                  <span style={{color:C.text,maxWidth:100,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",textAlign:"right"}}>{v}</span>
                </div>
              ))}
            </div>
            {statData.type==="categorical"&&statData.top&&(
              <div style={{marginTop:10}}>
                <div style={{fontSize:9,color:C.muted,marginBottom:4}}>DISTRIBUTION</div>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart data={statData.top.map(([v,c])=>({name:String(v).slice(0,10),count:c}))} layout="vertical" margin={{left:0,right:4,top:0,bottom:0}}>
                    <XAxis type="number" tick={{fontSize:9,fill:C.muted}} tickLine={false}/>
                    <YAxis type="category" dataKey="name" tick={{fontSize:9,fill:C.sec}} width={56}/>
                    <Bar dataKey="count" fill={C.orange} radius={[0,3,3,0]}/>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Pagination */}
      <div style={{padding:"9px 16px",borderTop:`1px solid ${C.bd}`,background:C.deep,display:"flex",alignItems:"center",gap:10,flexWrap:"wrap"}}>
        <button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0} style={{...bS(C.muted),padding:"4px 10px",fontSize:11,opacity:page===0?.4:1}}>‹ Prev</button>
        <span style={{fontSize:11,color:C.muted}}>Page {page+1} / {pages} · {totalFiltered.toLocaleString()} rows shown</span>
        <button onClick={()=>setPage(p=>Math.min(pages-1,p+1))} disabled={page>=pages-1} style={{...bS(C.muted),padding:"4px 10px",fontSize:11,opacity:page>=pages-1?.4:1}}>Next ›</button>
        {renameFrom && (
          <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center"}}>
            <span style={{fontSize:11,color:C.yellow}}>Rename '{renameFrom}' →</span>
            <input style={{...IS,width:130,padding:"5px 8px",fontSize:11}} value={renameTo} onChange={e=>setRenameTo(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")doRename();if(e.key==="Escape")setRenameFrom(null);}} autoFocus/>
            <button onClick={doRename} style={{...bP(C.green),padding:"5px 12px",fontSize:11}}>OK</button>
            <button onClick={()=>setRenameFrom(null)} style={{...bS(C.muted),padding:"5px 10px",fontSize:11}}>✕</button>
          </div>
        )}
        {/* Dropped columns */}
        {columns.filter(c=>c.dropped).map(c=>(
          <button key={c.name} onClick={async()=>{
            await fetch(`${API}/column/drop`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sid,col:"__restore_not_supported__"})});
            onColsChange(columns.map(x=>x.name===c.name?{...x,dropped:false}:x));
          }} style={{...bS(C.muted),padding:"3px 8px",fontSize:10}}>↩ {c.name}</button>
        ))}
      </div>
    </div>
  );
}

// ─── Preprocessing Panel ──────────────────────────────────────────────────────
function PrepPanel({ prep, setPrep, modality, onApply, prepLog, applying }) {
  const [open, setOpen] = useState(true);
  const u = (k,v) => setPrep(p => ({...p,[k]:v}));

  return (
    <div style={{...cc(),borderColor:C.purple+"50"}}>
      <div onClick={()=>setOpen(o=>!o)} style={{display:"flex",alignItems:"center",justifyContent:"space-between",cursor:"pointer",marginBottom:open?14:0}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{color:C.purple}}>⚙</span>
          <span style={{fontSize:11,fontWeight:700,color:C.purple,letterSpacing:"0.8px"}}>PREPROCESSING PIPELINE</span>
          {prepLog.length>0&&<span style={{fontSize:10,background:`${C.green}20`,color:C.green,padding:"2px 8px",borderRadius:10}}>✓ {prepLog.length} ops applied</span>}
        </div>
        <span style={{color:C.muted,fontSize:11}}>{open?"▲":"▼"}</span>
      </div>

      {open && <>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:12}}>
          <div>
            <label style={LB}>Missing Values Strategy</label>
            <select style={SE} value={prep.missingStrategy} onChange={e=>u("missingStrategy",e.target.value)}>
              {["Mean Imputation","Median Imputation","Mode Imputation","Drop Rows","Zero Fill","KNN Imputation"].map(v=><option key={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label style={LB}>Feature Scaling</label>
            <select style={SE} value={prep.scaling} onChange={e=>u("scaling",e.target.value)}>
              {["StandardScaler","MinMaxScaler","RobustScaler","MaxAbsScaler","None"].map(v=><option key={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label style={LB}>Categorical Encoding</label>
            <select style={SE} value={prep.encoding} onChange={e=>u("encoding",e.target.value)}>
              {["One-Hot","Label Encoding","None"].map(v=><option key={v}>{v}</option>)}
            </select>
          </div>
        </div>

        <div style={{display:"flex",gap:20,flexWrap:"wrap",marginBottom:12}}>
          <Tog v={prep.outlierRemoval} onChange={v=>u("outlierRemoval",v)} label="Outlier Removal (IQR)" small/>
          <Tog v={prep.removeDuplicates} onChange={v=>u("removeDuplicates",v)} label="Remove Duplicate Rows" small/>
          <Tog v={prep.featureSelection} onChange={v=>u("featureSelection",v)} label="Feature Selection (top-k)" small/>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <Tog v={prep.pca} onChange={v=>u("pca",v)} label="PCA Reduction" small/>
            {prep.pca && <input type="number" style={{...IS,width:58,padding:"4px 7px",fontSize:11}} value={prep.pcaVariance} min={0.7} max={0.999} step={0.01} onChange={e=>u("pcaVariance",parseFloat(e.target.value))} title="Variance threshold"/>}
          </div>
        </div>

        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <button onClick={onApply} disabled={applying} style={{...bP(C.purple),opacity:applying?.6:1,padding:"8px 20px"}}>
            {applying ? "⚙ Applying..." : "⚙ Apply Preprocessing"}
          </button>
          <span style={{fontSize:11,color:C.muted}}>Operations run on the actual dataset in the backend</span>
        </div>

        {prepLog.length>0 && (
          <div style={{marginTop:14,background:C.deep,borderRadius:8,padding:"10px 14px",border:`1px solid ${C.bd}`}}>
            <div style={{fontSize:10,color:C.green,fontWeight:700,marginBottom:7}}>✓ APPLIED OPS LOG</div>
            {prepLog.map((line,i)=><div key={i} style={{fontSize:11,color:C.sec,marginBottom:3}}>• {line}</div>)}
          </div>
        )}
      </>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab]       = useState("dataset");
  const [sid, setSid]       = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const [tfAvail, setTfAvail]     = useState(false);
  const [xgbAvail, setXgbAvail]   = useState(false);

  // Dataset state
  const [columns,   setColumns]   = useState([]);
  const [totalRows, setTotalRows] = useState(0);
  const [loadSrc,   setLoadSrc]   = useState(null);
  const [loadErr,   setLoadErr]   = useState("");
  const [loading,   setLoading]   = useState(false);
  const [urlInp,    setUrlInp]    = useState("");
  const [problemType, setProblemType] = useState("Classification");

  // Preprocessing
  const [prep, setPrep]     = useState({missingStrategy:"Mean Imputation",scaling:"StandardScaler",encoding:"One-Hot",outlierRemoval:false,removeDuplicates:false,featureSelection:false,pca:false,pcaVariance:0.95});
  const [prepLog, setPrepLog] = useState([]);
  const [applying, setApplying] = useState(false);

  // Model
  const [modelType, setModelType] = useState("classical");
  const [classAlgo, setClassAlgo] = useState("Random Forest");
  const [layers,    setLayers]    = useState([{id:1,n:256,a:"ReLU",d:0.2},{id:2,n:128,a:"ReLU",d:0.2},{id:3,n:64,a:"ReLU",d:0.1}]);
  const [outCfg,    setOutCfg]    = useState({activation:"Softmax",weightInit:"Xavier/Glorot",useBias:true});
  const [cfgSub,    setCfgSub]    = useState("architecture");
  const [bulk,      setBulk]      = useState({count:"5",n:"128",a:"ReLU",d:"0.2"});
  const [hp,        setHp]        = useState({optimizer:"Adam",lr:0.001,bs:32,epochs:100,l2:0.0001,earlyStopping:true,patience:10,scheduler:"ReduceOnPlateau",vSplit:0.2});

  // Training
  const [training,  setTraining]  = useState(false);
  const [history,   setHistory]   = useState([]);
  const [metrics,   setMetrics]   = useState(null);
  const [done,      setDone]      = useState(false);
  const [trainErr,  setTrainErr]  = useState("");
  const [statusMsg, setStatusMsg] = useState("");
  const [curEp,     setCurEp]     = useState(0);
  const esRef = useRef(null);

  const isClsf = !["Regression","Time Series","Clustering","Anomaly Detection"].includes(problemType);
  const targetCol = columns.find(c=>c.isTarget);

  // ── Check backend health ────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/health`).then(r=>r.json()).then(d=>{
      setBackendOk(true); setTfAvail(d.tensorflow||false); setXgbAvail(d.xgboost||false);
    }).catch(()=>setBackendOk(false));
  }, []);

  // ── Upload CSV ──────────────────────────────────────────────────────────────
  const uploadCSV = async file => {
    setLoading(true); setLoadErr(""); setPrepLog([]);
    const form = new FormData();
    form.append("file", file);
    if (sid) form.append("sid", sid);
    try {
      const res = await fetch(`${API}/upload`, { method:"POST", body:form });
      const d   = await res.json();
      if (d.error) { setLoadErr(d.error); setLoading(false); return; }
      setSid(d.sid); setColumns(d.columns); setTotalRows(d.rows);
      setLoadSrc({name:d.filename, rows:d.rows, cols:d.cols});
      setMetrics(null); setHistory([]); setDone(false);
    } catch(e) { setLoadErr(`Upload failed: ${e.message}`); }
    setLoading(false);
  };

  // ── Load URL ────────────────────────────────────────────────────────────────
  const loadURL = async () => {
    if (!urlInp.trim()) { setLoadErr("Enter a URL."); return; }
    setLoading(true); setLoadErr("");
    // fetch URL, re-upload as blob
    try {
      const res = await fetch(urlInp.trim());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], urlInp.split("/").pop()||"dataset.csv", {type:"text/csv"});
      await uploadCSV(file);
    } catch(e) { setLoadErr(`Could not fetch URL: ${e.message}`); setLoading(false); }
  };

  // ── Load sample ─────────────────────────────────────────────────────────────
  const loadSample = async key => {
    setLoading(true); setLoadErr(""); setPrepLog([]);
    const newSid = sid || `s_${Date.now()}`;
    try {
      const res = await fetch(`${API}/sample/${key}?sid=${newSid}`);
      const d   = await res.json();
      if (d.error) { setLoadErr(d.error); setLoading(false); return; }
      setSid(d.sid); setColumns(d.columns); setTotalRows(d.rows);
      setLoadSrc({name:d.filename, rows:d.rows, cols:d.cols});
      setMetrics(null); setHistory([]); setDone(false);
    } catch(e) { setLoadErr(`Error: ${e.message}`); }
    setLoading(false);
  };

  // ── Apply preprocessing ─────────────────────────────────────────────────────
  const applyPrep = async () => {
    if (!sid) return;
    setApplying(true);
    try {
      const res = await fetch(`${API}/preprocess`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sid,config:prep})});
      const d   = await res.json();
      if (d.error) { setLoadErr(d.error); setApplying(false); return; }
      setPrepLog(d.log||[]);
      setTotalRows(d.rows);
      if (d.columns) setColumns(cols => {
        const newNames = new Set(d.columns);
        return [...cols.filter(c=>newNames.has(c.name)||c.dropped),
                ...d.columns.filter(n=>!cols.find(c=>c.name===n)).map(n=>({name:n,type:"numeric",dropped:false,isTarget:false}))];
      });
    } catch(e) { setLoadErr(`Preprocessing error: ${e.message}`); }
    setApplying(false);
  };

  // ── Start training via SSE ──────────────────────────────────────────────────
  const startTraining = () => {
    if (!sid) { setTrainErr("Load a dataset first."); return; }
    if (!targetCol && !["Clustering","Anomaly Detection"].includes(problemType)) {
      setTrainErr("Set a target column first (click column header → 🎯 Set as Target)."); return;
    }
    if (esRef.current) esRef.current.close();

    setTraining(true); setDone(false); setHistory([]); setMetrics(null);
    setTrainErr(""); setStatusMsg("Connecting to backend..."); setCurEp(0);
    setTab("training");

    const body = {
      sid, problemType,
      modelType, algo:classAlgo,
      layers: layers.map(l=>({n:l.n,a:l.a,d:l.d})),
      outCfg, hp,
    };

    fetch(`${API}/train/stream`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
      .then(res => {
        if (!res.ok) { setTrainErr("Backend error. Is backend.py running?"); setTraining(false); return; }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";

        const pump = () => reader.read().then(({ done: streamDone, value }) => {
          if (streamDone) { setTraining(false); return; }
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split("\n\n");
          buf = lines.pop();
          lines.forEach(line => {
            const match = line.match(/^data: (.+)$/m);
            if (!match) return;
            try {
              const msg = JSON.parse(match[1]);
              if (msg.event === "progress") {
                setHistory(h => [...h, msg.data]);
                setCurEp(msg.data.epoch);
              } else if (msg.event === "metrics") {
                setMetrics(msg.data);
              } else if (msg.event === "status") {
                setStatusMsg(msg.data.message);
              } else if (msg.event === "error") {
                setTrainErr(msg.data.message);
                setTraining(false);
              } else if (msg.event === "done") {
                setTraining(false); setDone(true); setTab("results");
              }
            } catch {}
          });
          pump();
        }).catch(() => setTraining(false));
        pump();
      })
      .catch(() => { setTrainErr("Cannot reach backend. Run: python backend.py"); setTraining(false); });
  };

  const stopTraining = () => {
    if (esRef.current) esRef.current.close();
    setTraining(false);
    if (history.length) setDone(true);
  };

  // ── Layer helpers ──────────────────────────────────────────────────────────
  const addLayer    = () => { if(layers.length>=500)return; const last=layers[layers.length-1]?.n||64; setLayers(p=>[...p,{id:Date.now(),n:Math.max(8,Math.floor(last/2)),a:"ReLU",d:0.2}]); };
  const rmLayer     = id  => { if(layers.length>1)setLayers(p=>p.filter(l=>l.id!==id)); };
  const updL        = (id,f,v) => setLayers(p=>p.map(l=>l.id===id?{...l,[f]:v}:l));
  const addBulk     = () => { const n=Math.min(parseInt(bulk.count)||1,200); setLayers(p=>[...p,...Array.from({length:n},(_,i)=>({id:Date.now()+i,n:parseInt(bulk.n)||128,a:bulk.a,d:parseFloat(bulk.d)||0.2}))]); };
  const applyPreset = nm => setLayers(PRESETS[nm].map((l,i)=>({id:i+1,...l})));
  const globalAct   = act => setLayers(p=>p.map(l=>({...l,a:act})));
  const totalParams = (() => { let p=0,prev=20; layers.forEach(l=>{p+=(prev+1)*l.n;prev=l.n;}); p+=(prev+1)*3; return p; })();
  const fmtP = n => n>1e6?`${(n/1e6).toFixed(2)}M`:n>1e3?`${(n/1e3).toFixed(1)}K`:n;

  // ── Export helpers ─────────────────────────────────────────────────────────
  const exportModel   = () => window.open(`${API}/model/export?sid=${sid}`,"_blank");
  const exportHistory = () => window.open(`${API}/history/csv?sid=${sid}`,"_blank");
  const exportDataset = () => window.open(`${API}/dataset/export?sid=${sid}`,"_blank");

  const pct  = hp.epochs>0?(curEp/hp.epochs)*100:0;
  const last = history[history.length-1];
  const TABS = [{id:"dataset",lbl:"📋 Dataset"},{id:"configure",lbl:"⚙ Configure"},{id:"training",lbl:"⚡ Training"},{id:"results",lbl:"📊 Results"}];

  const fmt = (n,d=3) => typeof n==="number" ? n.toFixed(d) : "—";

  // ════════════════════════════════════════════════════════════════════════════
  return (
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,fontFamily:"'JetBrains Mono','Fira Code',ui-monospace,monospace"}}>

      {/* ── HEADER ── */}
      <div style={{borderBottom:`1px solid ${C.bd}`,background:C.card,position:"sticky",top:0,zIndex:100}}>
        <div style={{maxWidth:1280,margin:"0 auto",padding:"0 20px",display:"flex",alignItems:"center",justifyContent:"space-between",height:54}}>
          <div style={{display:"flex",alignItems:"center",gap:10}}>
            <Dot col={backendOk===true?C.green:backendOk===false?C.red:C.muted} glow={backendOk===true}/>
            <span style={{fontSize:13,fontWeight:700,letterSpacing:"1.5px"}}>ML TRAINING ASSISTANT</span>
            <span style={{fontSize:9,color:"#000",background:backendOk===true?C.green:backendOk===false?C.red:C.muted,padding:"2px 8px",borderRadius:4,fontWeight:700}}>
              {backendOk===null?"CHECKING...":backendOk?"BACKEND LIVE":"BACKEND OFFLINE"}
            </span>
            {tfAvail && <span style={{fontSize:9,color:C.purple,background:`${C.purple}20`,padding:"2px 7px",borderRadius:4,border:`1px solid ${C.purple}40`}}>TF ✓</span>}
            {xgbAvail && <span style={{fontSize:9,color:C.orange,background:`${C.orange}20`,padding:"2px 7px",borderRadius:4,border:`1px solid ${C.orange}40`}}>XGB ✓</span>}
            {loadSrc && <span style={{fontSize:10,color:C.cyan,background:`${C.cyan}15`,padding:"2px 10px",borderRadius:10,border:`1px solid ${C.cyan}30`}}>{loadSrc.name} · {totalRows.toLocaleString()} rows</span>}
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
          <div style={{fontSize:11,color:C.muted,textAlign:"right"}}>
            {backendOk===false && <span style={{color:C.red}}>Run: python backend.py</span>}
          </div>
        </div>
      </div>

      {/* Backend offline banner */}
      {backendOk===false && (
        <div style={{background:"#160008",borderBottom:`1px solid ${C.red}40`,padding:"10px 20px",textAlign:"center",fontSize:12,color:C.red}}>
          ⚠ Backend is not running. Open a terminal in your project folder and run: <strong>python backend.py</strong>
        </div>
      )}

      <div style={{maxWidth:1280,margin:"0 auto",padding:"24px 20px"}}>

        {/* ════════════ DATASET TAB */}
        {tab==="dataset" && (
          <div>
            <div style={{marginBottom:20}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>Dataset & Preprocessing</h2>
              <p style={{fontSize:12,color:C.muted}}>Upload local files, paste a URL, or use a built-in sample. All data is processed on your backend — works for any size including millions of rows.</p>
            </div>

            {/* Load */}
            <div style={{...cc(),marginBottom:16,borderColor:C.cyan+"40"}}>
              <ST color={C.cyan} mb={14}>LOAD DATASET</ST>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:16}}>
                {/* File upload */}
                <div style={{background:C.deep,borderRadius:10,padding:"16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:12,fontWeight:700,color:C.text,marginBottom:6}}>📄 Local CSV / JSON File</div>
                  <div style={{fontSize:11,color:C.muted,marginBottom:10}}>Upload any size file. Processed in chunks — handles millions of rows.</div>
                  <label style={{...bP(C.cyan),display:"inline-block",cursor:"pointer",fontSize:12,padding:"8px 16px"}}>
                    {loading?"Loading...":"Choose File"}
                    <input type="file" accept=".csv,.json,.tsv,.txt" style={{display:"none"}} onChange={e=>{if(e.target.files[0])uploadCSV(e.target.files[0]);e.target.value="";}} disabled={loading}/>
                  </label>
                </div>

                {/* URL */}
                <div style={{background:C.deep,borderRadius:10,padding:"16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:12,fontWeight:700,color:C.text,marginBottom:6}}>🌐 Online URL</div>
                  <div style={{fontSize:11,color:C.muted,marginBottom:8}}>Direct link to a CSV (GitHub raw URLs work great).</div>
                  <input style={{...IS,padding:"6px 10px",fontSize:11,marginBottom:8}} placeholder="https://raw.githubusercontent.com/.../data.csv" value={urlInp} onChange={e=>setUrlInp(e.target.value)} onKeyDown={e=>e.key==="Enter"&&loadURL()}/>
                  <button onClick={loadURL} disabled={loading} style={{...bP(C.green),fontSize:11,padding:"7px 14px",opacity:loading?.6:1}}>Fetch →</button>
                </div>

                {/* Backend info */}
                <div style={{background:C.deep,borderRadius:10,padding:"16px",border:`1px solid ${C.bd}`}}>
                  <div style={{fontSize:12,fontWeight:700,color:C.text,marginBottom:6}}>⚙ Backend Status</div>
                  <div style={{display:"flex",flexDirection:"column",gap:7}}>
                    {[["Backend",backendOk?"✓ Running on :5000":"✗ Offline — run backend.py",backendOk?C.green:C.red],
                      ["TensorFlow",tfAvail?"✓ Keras/DL available":"✗ pip install tensorflow",tfAvail?C.green:C.muted],
                      ["XGBoost",xgbAvail?"✓ Available":"✗ pip install xgboost",xgbAvail?C.green:C.muted],
                    ].map(([k,v,col])=>(
                      <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:11}}>
                        <span style={{color:C.muted}}>{k}</span>
                        <span style={{color:col}}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Samples */}
              <div>
                <div style={{fontSize:10,color:C.muted,letterSpacing:"0.8px",fontWeight:700,marginBottom:8}}>BUILT-IN SAMPLES (real sklearn datasets)</div>
                <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                  {SAMPLES.map(s=>(
                    <button key={s.key} onClick={()=>loadSample(s.key)} style={{background:loadSrc?.name===s.key?`${C.cyan}15`:C.deep,border:`1px solid ${loadSrc?.name===s.key?C.cyan:C.bd}`,borderRadius:8,padding:"8px 14px",cursor:"pointer",textAlign:"left",fontFamily:"inherit"}}>
                      <div style={{fontSize:12,fontWeight:600,color:loadSrc?.name===s.key?C.cyan:C.text,marginBottom:3}}>{s.label}</div>
                      <div style={{fontSize:10,color:C.muted}}>{s.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {loadErr && <div style={{marginTop:12,background:"#120608",border:`1px solid ${C.red}50`,borderRadius:8,padding:"9px 14px",fontSize:12,color:C.red}}>⚠ {loadErr}</div>}
              {loadSrc && <div style={{marginTop:12,background:`${C.green}10`,border:`1px solid ${C.green}40`,borderRadius:8,padding:"9px 14px",fontSize:12,color:C.green}}>
                ✓ Loaded: <strong>{loadSrc.name}</strong> — {totalRows.toLocaleString()} rows · {loadSrc.cols} columns
                {targetCol && <span style={{marginLeft:12,color:C.cyan}}>· Target: 🎯 {targetCol.name}</span>}
              </div>}
            </div>

            {/* Data Explorer */}
            {sid && columns.length>0 && (
              <div style={{marginBottom:16}}>
                <DataExplorer sid={sid} columns={columns} totalRows={totalRows}
                  onColsChange={setColumns}
                  onRowsChange={r=>{setTotalRows(r);}}/>
                {!targetCol && !["Clustering","Anomaly Detection"].includes(problemType) && (
                  <div style={{marginTop:8,fontSize:11,color:C.orange}}>⚠ No target column set. Click a column header ▾ → 🎯 Set as Target before training.</div>
                )}
                {targetCol && <div style={{marginTop:8,fontSize:11,color:C.green}}>🎯 Target: <strong>{targetCol.name}</strong> ({targetCol.type})</div>}
              </div>
            )}

            {/* Problem type + preprocessing */}
            {sid && (
              <>
                <div style={{...cc(),marginBottom:16}}>
                  <ST>PROBLEM TYPE</ST>
                  <select style={{...SE,maxWidth:280}} value={problemType} onChange={e=>setProblemType(e.target.value)}>
                    {PTYPES.map(t=><option key={t}>{t}</option>)}
                  </select>
                </div>
                <div style={{marginBottom:16}}>
                  <PrepPanel prep={prep} setPrep={setPrep} modality="Tabular" onApply={applyPrep} prepLog={prepLog} applying={applying}/>
                </div>
                {prepLog.length>0 && (
                  <div style={{display:"flex",gap:10,marginBottom:16}}>
                    <button onClick={exportDataset} style={bS(C.green)}>⬇ Export Processed Dataset (CSV)</button>
                  </div>
                )}
              </>
            )}

            <div style={{display:"flex",justifyContent:"flex-end",gap:10}}>
              <button onClick={()=>setTab("configure")} style={bP()}>Configure Model →</button>
            </div>
          </div>
        )}

        {/* ════════════ CONFIGURE TAB */}
        {tab==="configure" && (
          <div>
            <div style={{marginBottom:18}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>Model Configuration</h2>
              <p style={{fontSize:12,color:C.muted}}>{problemType}{loadSrc?` · ${totalRows.toLocaleString()} rows`:""}  {targetCol?`· Target: 🎯 ${targetCol.name}`:""}</p>
            </div>

            {/* Model type */}
            <div style={{...cc(),marginBottom:16}}>
              <ST mb={14}>MODEL TYPE</ST>
              <div style={{display:"flex",gap:10,marginBottom:14}}>
                {[["classical","🌳 Classical ML","Random Forest, XGBoost, SVM, KNN, LR — fastest, great for tabular"],
                  ["deep","🧠 Deep Learning","Real neural network training via TensorFlow/Keras"+(tfAvail?"":" — install TensorFlow first")]].map(([id,title,desc])=>(
                  <div key={id} onClick={()=>setModelType(id)} style={{flex:1,padding:"14px 18px",borderRadius:10,cursor:"pointer",background:modelType===id?`${id==="deep"?C.cyan:C.orange}18`:"transparent",border:`2px solid ${modelType===id?(id==="deep"?C.cyan:C.orange):C.bd}`,opacity:(id==="deep"&&!tfAvail)?0.5:1}}>
                    <div style={{fontSize:14,fontWeight:700,color:modelType===id?(id==="deep"?C.cyan:C.orange):C.sec,marginBottom:4}}>{title}</div>
                    <div style={{fontSize:11,color:C.muted}}>{desc}</div>
                  </div>
                ))}
              </div>
              {modelType==="classical" && (
                <div style={{maxWidth:320}}>
                  <label style={LB}>Algorithm</label>
                  <select style={SE} value={classAlgo} onChange={e=>setClassAlgo(e.target.value)}>
                    {CML.map(a=><option key={a}>{a}</option>)}
                  </select>
                </div>
              )}
            </div>

            {/* Sub-tabs */}
            <div style={{display:"flex",marginBottom:16,border:`1px solid ${C.bd}`,borderRadius:10,overflow:"hidden",width:"fit-content"}}>
              {["architecture","hyperparameters","regularization"].map(t=>(
                <button key={t} onClick={()=>setCfgSub(t)} style={{background:cfgSub===t?C.cyan:"transparent",color:cfgSub===t?"#000":C.muted,border:"none",padding:"8px 22px",fontSize:11,fontWeight:700,cursor:"pointer",letterSpacing:"0.5px",textTransform:"uppercase",transition:"all .15s",fontFamily:"inherit"}}>{t}</button>
              ))}
            </div>

            {/* Architecture */}
            {cfgSub==="architecture" && modelType==="deep" && (
              <div style={{display:"grid",gridTemplateColumns:"1fr 270px",gap:16}}>
                <div>
                  <div style={{...cc(),marginBottom:10,padding:"12px 18px"}}>
                    <ST mb={10}>QUICK PRESETS</ST>
                    <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                      {Object.keys(PRESETS).map(nm=>(
                        <button key={nm} onClick={()=>applyPreset(nm)} style={{...bS(C.muted),padding:"5px 12px",fontSize:11}}>{nm} <span style={{fontSize:9,color:C.muted}}>·{PRESETS[nm].length}L</span></button>
                      ))}
                    </div>
                  </div>

                  <div style={{...cc(),marginBottom:10,padding:"12px 18px",borderColor:C.purple+"50"}}>
                    <ST color={C.purple} mb={10}>BULK ADD LAYERS</ST>
                    <div style={{display:"grid",gridTemplateColumns:"60px 100px 1fr 80px auto",gap:8,alignItems:"end"}}>
                      <div><label style={LB}>Count</label><input type="number" style={IS} min={1} max={200} value={bulk.count} onChange={e=>setBulk(p=>({...p,count:e.target.value}))}/></div>
                      <div><label style={LB}>Neurons</label><input type="number" style={IS} value={bulk.n} min={1} max={8192} onChange={e=>setBulk(p=>({...p,n:e.target.value}))}/></div>
                      <div><label style={LB}>Activation</label>
                        <select style={SE} value={bulk.a} onChange={e=>setBulk(p=>({...p,a:e.target.value}))}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                      </div>
                      <div><label style={LB}>Dropout</label><input type="number" style={IS} min={0} max={0.9} step={0.05} value={bulk.d} onChange={e=>setBulk(p=>({...p,d:e.target.value}))}/></div>
                      <button onClick={addBulk} style={{...bP(C.purple),color:"#fff",padding:"8px 12px"}}>+ {bulk.count}</button>
                    </div>
                    <div style={{marginTop:8,display:"flex",gap:10,alignItems:"center"}}>
                      <span style={{fontSize:10,color:C.muted}}>Total: {layers.length}</span>
                      <span style={{fontSize:10,color:C.muted}}>Set all →</span>
                      <select style={{...SE,width:110,padding:"4px 8px",fontSize:11}} onChange={e=>e.target.value&&globalAct(e.target.value)}>
                        <option value="">Activation...</option>
                        {ACTS.map(a=><option key={a}>{a}</option>)}
                      </select>
                      <button onClick={()=>setLayers(p=>[p[0]])} style={{...bS(C.red),padding:"4px 10px",fontSize:10}}>Clear</button>
                    </div>
                  </div>

                  <div style={cc()}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                      <ST mb={0}>HIDDEN LAYERS ({layers.length})</ST>
                      <button onClick={addLayer} style={{...bS(C.cyan),padding:"4px 12px",fontSize:11}}>+ Add</button>
                    </div>
                    <div style={{display:"grid",gridTemplateColumns:"20px 1fr 1fr 70px 20px",gap:4,marginBottom:6}}>
                      {["#","Neurons","Activation","Dropout",""].map((h,i)=><div key={i} style={{fontSize:9,color:C.muted,letterSpacing:"0.8px"}}>{h}</div>)}
                    </div>
                    <div style={{padding:"5px 10px",background:C.deep,borderRadius:8,marginBottom:3,fontSize:12,display:"flex",gap:10,alignItems:"center"}}>
                      <Dot col={C.muted}/><span style={{color:C.muted}}>Input</span>
                    </div>
                    <div style={{maxHeight:320,overflowY:"auto",paddingRight:2}}>
                      {layers.map((l,i)=>(
                        <div key={l.id} style={{background:C.deep,border:`1px solid ${C.bd}`,borderRadius:8,padding:"6px 10px",marginBottom:3}}>
                          <div style={{display:"grid",gridTemplateColumns:"20px 1fr 1fr 70px 20px",gap:6,alignItems:"center"}}>
                            <div style={{width:18,height:18,borderRadius:"50%",background:`${C.cyan}20`,color:C.cyan,display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,fontWeight:700}}>{i+1}</div>
                            <input type="number" style={{...IS,padding:"5px 8px",fontSize:12}} value={l.n} min={1} max={16384} onChange={e=>updL(l.id,"n",Math.max(1,parseInt(e.target.value)||1))}/>
                            <select style={{...SE,padding:"5px 8px",fontSize:12}} value={l.a} onChange={e=>updL(l.id,"a",e.target.value)}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                            <input type="number" style={{...IS,padding:"5px 8px",fontSize:12}} value={l.d} min={0} max={0.9} step={0.05} onChange={e=>updL(l.id,"d",parseFloat(e.target.value)||0)}/>
                            <button onClick={()=>rmLayer(l.id)} style={{background:"transparent",border:"none",color:C.red,cursor:"pointer",fontSize:16,padding:0,lineHeight:1}}>×</button>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div style={{padding:"5px 10px",background:"#061810",borderRadius:8,border:`1px solid #0e3020`,marginTop:3,fontSize:12,display:"flex",gap:10,alignItems:"center"}}>
                      <Dot col={C.green}/><span style={{color:C.green}}>Output · {outCfg.activation}</span>
                    </div>
                  </div>
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:14}}>
                  <div style={cc()}>
                    <ST>OUTPUT LAYER</ST>
                    <div style={{display:"flex",flexDirection:"column",gap:12}}>
                      <div><label style={LB}>Activation</label>
                        <select style={SE} value={outCfg.activation} onChange={e=>setOutCfg(p=>({...p,activation:e.target.value}))}>{ACTS.map(a=><option key={a}>{a}</option>)}</select>
                      </div>
                      <div><label style={LB}>Weight Init</label>
                        <select style={SE} value={outCfg.weightInit} onChange={e=>setOutCfg(p=>({...p,weightInit:e.target.value}))}>{WINIT.map(w=><option key={w}>{w}</option>)}</select>
                      </div>
                      <Tog v={outCfg.useBias} onChange={v=>setOutCfg(p=>({...p,useBias:v}))} label="Use Bias Terms"/>
                    </div>
                  </div>
                  <div style={cc()}>
                    <ST>SUMMARY</ST>
                    {[["Layers",layers.length+2],["Hidden",layers.length],["Est. Params",fmtP(totalParams)]].map(([k,v])=>(
                      <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:7}}>
                        <span style={{color:C.muted}}>{k}</span><span>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {cfgSub==="architecture" && modelType==="classical" && (
              <div style={{...cc(),textAlign:"center",padding:40}}>
                <div style={{fontSize:40,marginBottom:10}}>🌳</div>
                <div style={{fontSize:14,color:C.orange,fontWeight:700,marginBottom:8}}>{classAlgo}</div>
                <div style={{fontSize:12,color:C.muted,lineHeight:1.9}}>Classical ML uses the algorithm selected above — no layer architecture needed.<br/>Epochs = number of estimators (trees, boosting rounds).<br/>Set training parameters in <strong style={{color:C.text}}>Hyperparameters</strong>.</div>
              </div>
            )}

            {cfgSub==="hyperparameters" && (
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                <div style={cc()}>
                  <ST>OPTIMIZER / ALGORITHM</ST>
                  <div style={{display:"flex",flexDirection:"column",gap:14}}>
                    {modelType==="deep" && <>
                      <div><label style={LB}>Optimizer</label>
                        <select style={SE} value={hp.optimizer} onChange={e=>setHp(p=>({...p,optimizer:e.target.value}))}>{OPTS.map(o=><option key={o}>{o}</option>)}</select>
                      </div>
                      <div><label style={LB}>Learning Rate: {hp.lr}</label>
                        <input type="range" min={-5} max={-1} step={0.05} value={Math.log10(hp.lr)} onChange={e=>setHp(p=>({...p,lr:+((10**parseFloat(e.target.value)).toFixed(7))}))} style={{width:"100%",accentColor:C.cyan}}/>
                        <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted,marginTop:2}}><span>1e-5</span><span>1e-3</span><span>0.1</span></div>
                      </div>
                      <div><label style={LB}>LR Scheduler</label>
                        <select style={SE} value={hp.scheduler} onChange={e=>setHp(p=>({...p,scheduler:e.target.value}))}>{SCHS.map(s=><option key={s}>{s}</option>)}</select>
                      </div>
                    </>}
                  </div>
                </div>
                <div style={cc()}>
                  <ST>TRAINING SCHEDULE</ST>
                  <div style={{display:"flex",flexDirection:"column",gap:14}}>
                    <div><label style={LB}>{modelType==="classical"?"Estimators / Iterations":"Epochs"}: {hp.epochs}</label>
                      <input type="range" min={10} max={1000} step={10} value={hp.epochs} onChange={e=>setHp(p=>({...p,epochs:parseInt(e.target.value)}))} style={{width:"100%",accentColor:C.cyan}}/>
                      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted,marginTop:2}}><span>10</span><span>500</span><span>1000</span></div>
                    </div>
                    {modelType==="deep" && <>
                      <div><label style={LB}>Batch Size</label>
                        <select style={SE} value={hp.bs} onChange={e=>setHp(p=>({...p,bs:parseInt(e.target.value)}))}>{BSIZES.map(b=><option key={b}>{b}</option>)}</select>
                      </div>
                      <div><label style={LB}>Validation Split: {(hp.vSplit*100).toFixed(0)}%</label>
                        <input type="range" min={0.05} max={0.4} step={0.05} value={hp.vSplit} onChange={e=>setHp(p=>({...p,vSplit:parseFloat(e.target.value)}))} style={{width:"100%",accentColor:C.cyan}}/>
                      </div>
                    </>}
                  </div>
                </div>
              </div>
            )}

            {cfgSub==="regularization" && (
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                <div style={cc()}>
                  <ST>L2 REGULARIZATION</ST>
                  <label style={LB}>Lambda: {hp.l2}</label>
                  <input type="range" min={0} max={0.01} step={0.0001} value={hp.l2} onChange={e=>setHp(p=>({...p,l2:parseFloat(e.target.value)}))} style={{width:"100%",accentColor:C.cyan,marginBottom:6}}/>
                </div>
                <div style={cc()}>
                  <ST>EARLY STOPPING</ST>
                  <div style={{display:"flex",flexDirection:"column",gap:12}}>
                    <Tog v={hp.earlyStopping} onChange={v=>setHp(p=>({...p,earlyStopping:v}))} label="Enable Early Stopping"/>
                    {hp.earlyStopping && <div><label style={LB}>Patience (epochs)</label>
                      <input type="number" style={IS} value={hp.patience} min={1} max={200} onChange={e=>setHp(p=>({...p,patience:parseInt(e.target.value)||1}))}/>
                    </div>}
                  </div>
                </div>
              </div>
            )}

            <div style={{display:"flex",justifyContent:"space-between",marginTop:20}}>
              <button onClick={()=>setTab("dataset")} style={bS(C.cyan)}>← Dataset</button>
              <div style={{display:"flex",gap:10,alignItems:"center"}}>
                {trainErr && <span style={{fontSize:12,color:C.red}}>{trainErr}</span>}
                <button onClick={startTraining} disabled={!backendOk} style={{...bP(C.green),fontSize:14,padding:"12px 28px",opacity:!backendOk?.4:1}}>▶ Start Real Training</button>
              </div>
            </div>
          </div>
        )}

        {/* ════════════ TRAINING TAB */}
        {tab==="training" && (
          <div>
            <div style={{marginBottom:18}}>
              <h2 style={{fontSize:20,fontWeight:700,marginBottom:4}}>{training?"⚡ Training on Real Data...":done?"✓ Training Complete":"Training Monitor"}</h2>
              <p style={{fontSize:12,color:C.muted}}>{loadSrc?.name||"Dataset"} · {modelType==="deep"?`${layers.length} layers`:classAlgo} · {hp.optimizer} lr={hp.lr}</p>
              {statusMsg && <div style={{fontSize:12,color:C.cyan,marginTop:6}}>↳ {statusMsg}</div>}
            </div>

            {trainErr && <div style={{...cc(),borderColor:C.red+"50",marginBottom:16}}>
              <div style={{fontSize:13,color:C.red}}>⚠ Training Error: {trainErr}</div>
              <div style={{fontSize:11,color:C.muted,marginTop:6}}>Make sure backend.py is running: <code style={{color:C.cyan}}>python backend.py</code></div>
            </div>}

            {!training && !done && !trainErr && (
              <div style={{...cc(),textAlign:"center",padding:52}}>
                <div style={{fontSize:13,color:C.muted,marginBottom:20}}>Configure your model then start training. Data is sent to the Python backend for real training.</div>
                <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                  <button onClick={()=>setTab("configure")} style={bS(C.cyan)}>← Configure</button>
                  <button onClick={startTraining} disabled={!backendOk} style={{...bP(C.green),fontSize:14,padding:"12px 26px"}}>▶ Start Training</button>
                </div>
              </div>
            )}

            {(training||done)&&history.length>0 && <>
              <div style={{...cc(),marginBottom:16}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                  <span style={{fontSize:13,color:C.sec}}>Epoch / Iteration: {curEp} / {hp.epochs}</span>
                  <div style={{display:"flex",gap:10,alignItems:"center"}}>
                    {training && <div style={{display:"flex",gap:4}}>{[0,1,2].map(i=><div key={i} style={{width:5,height:5,borderRadius:"50%",background:C.cyan,opacity:0.4+i*.3}}/>)}</div>}
                    <span style={{fontSize:16,fontWeight:700,color:C.cyan}}>{pct.toFixed(1)}%</span>
                  </div>
                </div>
                <div style={{height:6,background:C.inp,borderRadius:3,overflow:"hidden",marginBottom:14}}>
                  <div style={{height:"100%",width:`${pct}%`,background:`linear-gradient(90deg,${C.cyan},${C.green})`,borderRadius:3,transition:"width 0.15s"}}/>
                </div>
                {last && <div style={{display:"grid",gridTemplateColumns:`repeat(${isClsf&&last.valAcc!=null?4:2},1fr)`,gap:10}}>
                  {[["Train Loss",fmt(last.trainLoss,4),C.cyan],["Val Loss",fmt(last.valLoss,4),C.orange],
                    ...(isClsf&&last.valAcc!=null?[["Train Acc",`${fmt(last.trainAcc,1)}%`,C.green],["Val Acc",`${fmt(last.valAcc,1)}%`,"#ff88bb"]]:[])
                  ].map(([k,v,col])=>(
                    <div key={k} style={{background:C.deep,borderRadius:8,padding:"10px",textAlign:"center"}}>
                      <div style={{fontSize:10,color:C.muted,marginBottom:5}}>{k}</div>
                      <div style={{fontSize:18,fontWeight:700,color:col}}>{v}</div>
                    </div>
                  ))}
                </div>}
              </div>

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
                {[["LOSS",["trainLoss","valLoss"],[C.cyan,C.orange],null],
                  isClsf&&last?.valAcc!=null?["ACCURACY",["trainAcc","valAcc"],[C.green,"#ff88bb"],[0,100]]:["LOSS (TRAIN)",["trainLoss"],[C.cyan],null]
                ].map(([title,keys,colors,domain])=>(
                  <div key={title} style={cc()}>
                    <div style={{fontSize:10,fontWeight:700,color:C.muted,letterSpacing:"0.8px",marginBottom:12}}>{title}</div>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={history} margin={{top:4,right:6,left:-22,bottom:4}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#131e35"/>
                        <XAxis dataKey="epoch" tick={{fill:C.muted,fontSize:10}}/>
                        <YAxis domain={domain||undefined} tick={{fill:C.muted,fontSize:10}}/>
                        <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.bd}`,fontSize:11,color:C.text}} formatter={domain?v=>`${fmt(v,1)}%`:v=>fmt(v,4)}/>
                        {keys.map((k,i)=><Line key={k} type="monotone" dataKey={k} stroke={colors[i]} strokeWidth={2} dot={false} name={k}/>)}
                        <Legend wrapperStyle={{fontSize:10}}/>
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>

              <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                {training && <button onClick={stopTraining} style={{...bS(C.red),borderColor:C.red}}>■ Stop</button>}
                {done && <button onClick={()=>setTab("results")} style={bP()}>View Results →</button>}
              </div>
            </>}
          </div>
        )}

        {/* ════════════ RESULTS TAB */}
        {tab==="results" && (
          <div>
            {!metrics ? (
              <div style={{...cc(),textAlign:"center",padding:60}}>
                <div style={{fontSize:40,marginBottom:14}}>📊</div>
                <div style={{fontSize:13,color:C.muted,marginBottom:20}}>No results yet. Train a model to see real metrics here.</div>
                <div style={{display:"flex",gap:10,justifyContent:"center"}}>
                  <button onClick={()=>setTab("dataset")} style={bS(C.cyan)}>← Start Here</button>
                  <button onClick={startTraining} disabled={!sid||!backendOk} style={bP()}>▶ Train</button>
                </div>
              </div>
            ) : <>
              <div style={{marginBottom:18}}>
                <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
                  <Dot col={C.green} glow/>
                  <h2 style={{fontSize:20,fontWeight:700}}>Real Training Results</h2>
                </div>
                <p style={{fontSize:12,color:C.muted}}>{loadSrc?.name||"Dataset"} · {history.length} epochs · {modelType==="deep"?`${layers.length} layers`:classAlgo}</p>
              </div>

              <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:16}}>
                {(isClsf ? [
                  {label:"Val Accuracy", v:metrics.valAcc!=null?`${fmt(metrics.valAcc,1)}%`:"—", col:C.green},
                  {label:"Val Loss",     v:fmt(metrics.valLoss,4), col:C.cyan},
                  {label:"F1 Score",     v:metrics.f1!=null?fmt(metrics.f1,4):"—", col:C.orange},
                  {label:"Precision",    v:metrics.precision!=null?fmt(metrics.precision,4):"—", col:C.pink},
                ] : [
                  {label:"R² Score",  v:metrics.r2!=null?fmt(metrics.r2,4):"—", col:C.green},
                  {label:"Val Loss",  v:fmt(metrics.valLoss,6), col:C.cyan},
                  {label:"MAE",       v:metrics.mae!=null?fmt(metrics.mae,4):"—", col:C.orange},
                  {label:"RMSE",      v:metrics.rmse!=null?fmt(metrics.rmse,4):"—", col:C.pink},
                ]).map(({label,v,col})=><MBox key={label} label={label} value={v} color={col}/>)}
              </div>

              {metrics.silhouette!=null && (
                <div style={{...cc(),marginBottom:16,borderColor:C.purple+"40"}}>
                  <ST color={C.purple}>CLUSTERING METRICS</ST>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:12}}>
                    <MBox label="Silhouette Score" value={fmt(metrics.silhouette,4)} color={C.purple}/>
                    <MBox label="Clusters Found" value={metrics.n_clusters} color={C.orange}/>
                  </div>
                </div>
              )}

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:16}}>
                {history.length>0 && <div style={cc()}>
                  <div style={{fontSize:10,fontWeight:700,color:C.muted,letterSpacing:"0.8px",marginBottom:12}}>TRAINING HISTORY</div>
                  <ResponsiveContainer width="100%" height={175}>
                    <LineChart data={history} margin={{top:4,right:6,left:-22,bottom:4}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#131e35"/>
                      <XAxis dataKey="epoch" tick={{fill:C.muted,fontSize:10}}/>
                      <YAxis tick={{fill:C.muted,fontSize:10}}/>
                      <Tooltip contentStyle={{background:C.card,border:`1px solid ${C.bd}`,fontSize:11}}/>
                      <Line type="monotone" dataKey="trainLoss" stroke={C.cyan} strokeWidth={2} dot={false} name="Train Loss"/>
                      <Line type="monotone" dataKey="valLoss" stroke={C.orange} strokeWidth={2} dot={false} name="Val Loss"/>
                      <Legend wrapperStyle={{fontSize:10}}/>
                    </LineChart>
                  </ResponsiveContainer>
                </div>}
                <div style={cc()}>
                  <ST>CONFIGURATION</ST>
                  {[["Dataset",loadSrc?.name||"—"],["Problem",problemType],["Model",modelType==="deep"?`${layers.length} hidden layers`:classAlgo],
                    ["Target",targetCol?.name||"—"],["Optimizer",hp.optimizer],["LR",hp.lr],["Batch",hp.bs],["Epochs Run",history.length],
                    ["Preprocessing",prepLog.length?`${prepLog.length} ops applied`:"None"],
                    ["Train Acc",metrics.trainAcc!=null?`${fmt(metrics.trainAcc,1)}%`:"—"],
                    ["Val Acc",metrics.valAcc!=null?`${fmt(metrics.valAcc,1)}%`:"—"],
                  ].map(([k,v])=>(
                    <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12,marginBottom:5}}>
                      <span style={{color:C.muted}}>{k}</span>
                      <span style={{maxWidth:190,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",textAlign:"right"}}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Confusion matrix */}
              {metrics.confusionMatrix && (
                <div style={{...cc(),marginBottom:16}}>
                  <ST>CONFUSION MATRIX (real predictions on validation set)</ST>
                  <div style={{overflowX:"auto"}}>
                    <table style={{borderCollapse:"collapse",fontSize:11}}>
                      <tbody>
                        {metrics.confusionMatrix.map((row,i)=>(
                          <tr key={i}>
                            {row.map((val,j)=>(
                              <td key={j} style={{padding:"8px 12px",textAlign:"center",
                                background:i===j?`${C.green}30`:val>0?`${C.orange}15`:"transparent",
                                color:i===j?C.green:val>0?C.orange:C.muted,
                                border:`1px solid ${C.bd}`,minWidth:50,fontWeight:i===j?700:400}}>
                                {val}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{fontSize:10,color:C.muted,marginTop:6}}>Rows = Actual · Columns = Predicted · Green diagonal = correct predictions</div>
                  </div>
                </div>
              )}

              {/* Export */}
              <div style={{...cc(),borderColor:C.yellow+"40",marginBottom:16}}>
                <ST color={C.yellow} mb={14}>EXPORT TRAINED MODEL & DATA</ST>
                <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:10}}>
                  <button onClick={exportModel} style={bP(C.yellow)}>⬇ Download Trained Model (.pkl / .h5)</button>
                  {history.length>0 && <button onClick={exportHistory} style={bS(C.cyan)}>⬇ Training History (.csv)</button>}
                  {prepLog.length>0 && <button onClick={exportDataset} style={bS(C.green)}>⬇ Processed Dataset (.csv)</button>}
                </div>
                <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>
                  The <span style={{color:C.yellow}}>.pkl</span> file (scikit-learn) or <span style={{color:C.yellow}}>.h5</span> file (Keras) is the real trained model —
                  load it in Python with <code style={{color:C.cyan}}>pickle.load()</code> or <code style={{color:C.cyan}}>tf.keras.models.load_model()</code> and make real predictions.
                </div>
              </div>

              <div style={{display:"flex",justifyContent:"space-between"}}>
                <button onClick={()=>setTab("configure")} style={bS(C.cyan)}>← Reconfigure</button>
                <div style={{display:"flex",gap:10}}>
                  <button onClick={startTraining} disabled={!backendOk} style={bS(C.orange)}>↺ Retrain</button>
                  <button onClick={()=>{setDone(false);setHistory([]);setMetrics(null);setSid(null);setColumns([]);setLoadSrc(null);setPrepLog([]);setTotalRows(0);setTab("dataset");}} style={bP()}>+ New Model</button>
                </div>
              </div>
            </>}
          </div>
        )}

      </div>
    </div>
  );
}
