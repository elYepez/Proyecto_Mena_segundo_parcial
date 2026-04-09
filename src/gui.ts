
import type { Vec3 } from "./math";
import { makeObjectGPU, makeCustomGeo, parseOBJ, syncRotationSliders, uploadTexture } from "./main";

type Quat = [number,number,number,number];
function qIdentity(): Quat { return [0,0,0,1]; }

export interface SceneObject {
  id:          number;
  shape:       "cube" | "sphere" | "obj";
  label:       string;
  color:       string;
  ambient:     number;
  diffuse:     number;
  specular:    number;
  shininess:   number;
  position:    Vec3;
  quaternion:  Quat;   
  scale:       Vec3;
  useTexture:  boolean;
  gpu:         ReturnType<typeof makeObjectGPU>;
  customGeo?:  ReturnType<typeof makeCustomGeo>;
}

interface GlobalState {
  modelId:      number;
  lightX:       number;
  lightY:       number;
  lightZ:       number;
  autoRotLight: boolean;
  lightColor:   string;
}

const globalState: GlobalState = {
  modelId:0, lightX:3, lightY:4, lightZ:3, autoRotLight:true, lightColor:"#ffffff",
};
const objects: SceneObject[] = [];
let selectedIndex = -1;
let nextId = 1;
let nextX  = 0;

export function getGlobalState()   { return globalState; }
export function getObjects()       { return objects; }
export function getSelectedIndex() { return selectedIndex; }

export function hexToRgb(hex: string): [number,number,number] {
  const n=parseInt(hex.slice(1),16);
  return [(n>>16&255)/255,(n>>8&255)/255,(n&255)/255];
}

function makeObject(shape:"cube"|"sphere", label:string): SceneObject {
  const x=nextX; nextX+=3;
  return { id:nextId++, shape, label, color:"#4a9eff",
    ambient:0.12, diffuse:0.75, specular:0.60, shininess:32,
    position:[x,0,0], quaternion:qIdentity(), scale:[1,1,1],
    useTexture:false, gpu:makeObjectGPU() };
}

function slider(id:string, label:string, min:number, max:number, step:number, val:number) {
  const d=Number.isInteger(val)?String(val):val.toFixed(2);
  return `<div class="slider-row">
    <span class="slider-label">${label}</span>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}">
    <span class="slider-val" id="${id}-val">${d}</span></div>`;
}

function wire(id:string, set:(v:number)=>void) {
  const el=document.getElementById(id) as HTMLInputElement|null;
  const vl=document.getElementById(`${id}-val`);
  if (!el) return;
  el.addEventListener("input",()=>{
    const v=parseFloat(el.value); set(v);
    if (vl) vl.textContent=Number.isInteger(v)?String(v):v.toFixed(2);
  });
}

const MODEL_DESCS: Record<number,string> = {
  0:"Flat",
  1:"Gouraud",
  2:"Phong",
  3:"Blinn-Phong",
  4:"Normal Buffer",
  5:"Wireframe",
  6:"Depth",
  7:"Texture × Phong",
};


function renderSceneList() {
  const list=document.getElementById("scene-list")!;
  list.innerHTML="";
  objects.forEach((obj,i)=>{
    const btn=document.createElement("button");
    btn.className="scene-item"+(i===selectedIndex?" active":"");
    btn.textContent=`${i+1}. ${obj.label}`;
    btn.addEventListener("click",()=>{
      selectedIndex=(i===selectedIndex)?-1:i;
      renderSceneList(); renderInspector();
    });
    list.appendChild(btn);
  });
}


function renderInspector() {
  const panel=document.getElementById("inspector")!;
  if (selectedIndex<0||selectedIndex>=objects.length) {
    panel.innerHTML=`<div class="no-selection">No object selected<br><small>Drag to orbit camera · Scroll to zoom</small></div>`;
    return;
  }
  const obj=objects[selectedIndex];
  const [px,py,pz]=obj.position;
  const [sx,sy,sz]=obj.scale;

  // Convert quat to Euler for slider display
  const [qx,qy,qz,qw]=obj.quaternion;
  const rx=Math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy));
  const ry=Math.asin(Math.max(-1,Math.min(1,2*(qw*qy-qz*qx))));
  const rz=Math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));

  panel.innerHTML=`
    <div class="obj-panel-title">${obj.label.toUpperCase()}</div>

    <div class="gui-label">Position</div>
    ${slider("obj-px","X",-10,10,0.05,px)}
    ${slider("obj-py","Y",-10,10,0.05,py)}
    ${slider("obj-pz","Z",-10,10,0.05,pz)}

    <div class="gui-label" style="margin-top:8px">Rotation (drag canvas or use sliders)</div>
    ${slider("obj-rx","X",-3.15,3.15,0.01,rx)}
    ${slider("obj-ry","Y",-3.15,3.15,0.01,ry)}
    ${slider("obj-rz","Z",-3.15,3.15,0.01,rz)}

    <div class="gui-label" style="margin-top:8px">Scale</div>
    ${slider("obj-sx","X",0.05,6,0.05,sx)}
    ${slider("obj-sy","Y",0.05,6,0.05,sy)}
    ${slider("obj-sz","Z",0.05,6,0.05,sz)}

    <div class="gui-label" style="margin-top:8px">Material</div>
    ${slider("obj-ambient",  "Ambient (Ka)", 0,  1,  0.01, obj.ambient)}
    ${slider("obj-diffuse",  "Diffuse (Kd)", 0,  1,  0.01, obj.diffuse)}
    ${slider("obj-specular", "Specular (Ks)",0,  1,  0.01, obj.specular)}
    ${slider("obj-shininess","Shininess (n)",1,256,1,      obj.shininess)}

    <div class="gui-label" style="margin-top:8px">Color</div>
    <div class="color-row">
      <span>Object</span><input type="color" id="obj-color" value="${obj.color}">
    </div>

    <div class="gui-label" style="margin-top:8px">Texture (Spherical UV)</div>
    <input type="file" id="tex-upload" accept="image/*" class="file-input">
    <label class="checkbox-row" style="margin-top:4px">
      <input type="checkbox" id="use-texture" ${obj.useTexture?"checked":""}> Use texture (mode 7)
    </label>

    <div class="inspector-actions">
      <button id="btn-deselect">Deselect</button>
      <button class="remove-btn" id="btn-remove">Remove</button>
    </div>`;

  wire("obj-px",v=>{ obj.position[0]=v; });
  wire("obj-py",v=>{ obj.position[1]=v; });
  wire("obj-pz",v=>{ obj.position[2]=v; });


  const rebuildQuat=()=>{
    const rx2=parseFloat((document.getElementById("obj-rx") as HTMLInputElement).value);
    const ry2=parseFloat((document.getElementById("obj-ry") as HTMLInputElement).value);
    const rz2=parseFloat((document.getElementById("obj-rz") as HTMLInputElement).value);
    // Convert Euler YXZ → quaternion
    const cx=Math.cos(rx2/2), sx2=Math.sin(rx2/2);
    const cy=Math.cos(ry2/2), sy2=Math.sin(ry2/2);
    const cz=Math.cos(rz2/2), sz2=Math.sin(rz2/2);
    obj.quaternion=[
      sx2*cy*cz+cx*sy2*sz2,
      cx*sy2*cz-sx2*cy*sz2,
      cx*cy*sz2+sx2*sy2*cz,
      cx*cy*cz-sx2*sy2*sz2,
    ];
  };
  wire("obj-rx",()=>rebuildQuat());
  wire("obj-ry",()=>rebuildQuat());
  wire("obj-rz",()=>rebuildQuat());

  wire("obj-sx",v=>{ obj.scale[0]=v; });
  wire("obj-sy",v=>{ obj.scale[1]=v; });
  wire("obj-sz",v=>{ obj.scale[2]=v; });

  wire("obj-ambient",  v=>{ obj.ambient=v; });
  wire("obj-diffuse",  v=>{ obj.diffuse=v; });
  wire("obj-specular", v=>{ obj.specular=v; });
  wire("obj-shininess",v=>{ obj.shininess=v; });

  (document.getElementById("obj-color") as HTMLInputElement)
    .addEventListener("input",e=>{ obj.color=(e.target as HTMLInputElement).value; });

  // Texture upload
  document.getElementById("tex-upload")!.addEventListener("change", async e=>{
    const file=(e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    await uploadTexture(file, obj);
    obj.useTexture=true;
    (document.getElementById("use-texture") as HTMLInputElement).checked=true;
    // Auto-switch to texture mode
    globalState.modelId=7;
    document.querySelectorAll(".model-btn").forEach(b=>b.classList.remove("active"));
    document.querySelector('[data-id="7"]')?.classList.add("active");
    document.getElementById("model-desc")!.textContent=MODEL_DESCS[7];
  });

  (document.getElementById("use-texture") as HTMLInputElement)
    .addEventListener("change",e=>{ obj.useTexture=(e.target as HTMLInputElement).checked; });

  document.getElementById("btn-deselect")!.addEventListener("click",()=>{
    selectedIndex=-1; renderSceneList(); renderInspector();
  });
  document.getElementById("btn-remove")!.addEventListener("click",()=>{
    objects[selectedIndex].gpu.uniformBuf.destroy();
    objects.splice(selectedIndex,1);
    selectedIndex=objects.length>0?Math.min(selectedIndex,objects.length-1):-1;
    renderSceneList(); renderInspector();
  });
}


export function initGUI() {
  const overlay=document.createElement("div");
  overlay.id="gui";
  overlay.innerHTML=`
<div class="gui-panel gui-panel--left">
  <div class="gui-title">Pipeline</div>

  <div class="gui-section">
    <div class="gui-label">Add Object</div>
    <div class="model-btns">
      <button class="shape-btn" id="btn-add-sphere">Sphere</button>
      <button class="shape-btn" id="btn-add-cube">Cube</button>
    </div>
    <div class="gui-label" style="margin-top:8px">Load OBJ File</div>
    <input type="file" id="obj-upload" accept=".obj" class="file-input">
  </div>

  <div class="gui-section">
    <div class="gui-label">Render Mode (Global)</div>
    <div class="model-btns">
      <button class="model-btn active" data-id="0">Flat</button>
      <button class="model-btn" data-id="1">Gouraud</button>
      <button class="model-btn" data-id="2">Phong</button>
      <button class="model-btn" data-id="3">Blinn-Phong</button>
    </div>
    <div class="model-btns">
      <button class="model-btn" data-id="4">Normals</button>
      <button class="model-btn" data-id="5">Wireframe</button>
      <button class="model-btn" data-id="6">Depth</button>
      <button class="model-btn" data-id="7">Texture</button>
    </div>
    <div class="model-desc" id="model-desc"></div>
  </div>

  <div class="gui-section">
    <div class="gui-label">Global Light Color</div>
    <div class="color-row"><span>Light</span><input type="color" id="lightColor" value="#ffffff"></div>
  </div>

  <div class="gui-section">
    <div class="gui-label">Light Position</div>
    ${slider("lightX","X",-8,8,0.1,globalState.lightX)}
    ${slider("lightY","Y",-8,8,0.1,globalState.lightY)}
    ${slider("lightZ","Z",-8,8,0.1,globalState.lightZ)}
    <label class="checkbox-row">
      <input type="checkbox" id="autoRotLight" checked> Auto-rotate light
    </label>
  </div>

  <div class="gui-hint">
    Ypez proyecto a WebGpu
  </div>
</div>

<div class="gui-panel gui-panel--right">
  <div class="gui-title">Scene</div>
  <div id="scene-list" class="scene-list"></div>
  <div id="inspector" class="object-panel">
    <div class="no-selection">No object selected<br><small>Drag to orbit camera · Scroll to zoom</small></div>
  </div>
</div>`;

  document.body.appendChild(overlay);

  const updateDesc=()=>{ document.getElementById("model-desc")!.textContent=MODEL_DESCS[globalState.modelId]; };
  updateDesc();

  document.querySelectorAll<HTMLButtonElement>(".model-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      globalState.modelId=Number(btn.dataset.id);
      document.querySelectorAll(".model-btn").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active"); updateDesc();
    });
  });

  document.getElementById("btn-add-sphere")!.addEventListener("click",()=>{
    objects.push(makeObject("sphere","Sphere"));
    selectedIndex=objects.length-1; renderSceneList(); renderInspector();
  });
  document.getElementById("btn-add-cube")!.addEventListener("click",()=>{
    objects.push(makeObject("cube","Cube"));
    selectedIndex=objects.length-1; renderSceneList(); renderInspector();
  });

  document.getElementById("obj-upload")!.addEventListener("change", async e=>{
    const file=(e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const src=await file.text();
    const name=file.name.replace(".obj","");
    const { data }=parseOBJ(src);
    const x=nextX; nextX+=3;
    const obj: SceneObject={
      id:nextId++, shape:"obj", label:name, color:"#4a9eff",
      ambient:0.12, diffuse:0.75, specular:0.60, shininess:32,
      position:[x,0,0], quaternion:qIdentity(), scale:[1,1,1],
      useTexture:false, gpu:makeObjectGPU(), customGeo:makeCustomGeo(data),
    };
    objects.push(obj);
    selectedIndex=objects.length-1;
    renderSceneList(); renderInspector();
    (e.target as HTMLInputElement).value="";
  });

  (["lightX","lightY","lightZ"] as const).forEach(id=>{
    const el=document.getElementById(id) as HTMLInputElement;
    el.addEventListener("input",()=>{
      (globalState as Record<string,unknown>)[id]=parseFloat(el.value);
      document.getElementById(`${id}-val`)!.textContent=el.value;
    });
  });
  (document.getElementById("autoRotLight") as HTMLInputElement)
    .addEventListener("change",e=>{ globalState.autoRotLight=(e.target as HTMLInputElement).checked; });
  (document.getElementById("lightColor") as HTMLInputElement)
    .addEventListener("input",e=>{ globalState.lightColor=(e.target as HTMLInputElement).value; });
}