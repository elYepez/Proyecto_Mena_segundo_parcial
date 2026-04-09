/// <reference types="@webgpu/types" />

import "./style.css";
import shaderCode from "./shader.wgsl?raw";
import { mat4 } from "./math";
import type { Vec3 } from "./math";
import { hexToRgb, initGUI, getGlobalState, getObjects, getSelectedIndex } from "./gui";


if (!navigator.gpu) throw new Error("WebGPU not supported");
const canvas = document.querySelector("#gfx-main") as HTMLCanvasElement;
if (!canvas) throw new Error("Canvas #gfx-main not found");
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter found");
const device  = await adapter.requestDevice();
const context = canvas.getContext("webgpu")!;
const format  = navigator.gpu.getPreferredCanvasFormat();

let depthTexture: GPUTexture | null = null;
function resize() {
  canvas.width  = Math.max(1, Math.floor(window.innerWidth  * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(window.innerHeight * devicePixelRatio));
  context.configure({ device, format, alphaMode: "premultiplied" });
  depthTexture?.destroy();
  depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}
resize();
window.addEventListener("resize", resize);


function generateCube(): Float32Array {
  type V3 = [number,number,number];
  const faces: Array<{ n: V3; verts: number[][] }> = [
    { n:[ 0, 0, 1], verts:[[-1,-1,1,0,1],[1,-1,1,1,1],[1,1,1,1,0],[-1,1,1,0,0]] },
    { n:[ 0, 0,-1], verts:[[1,-1,-1,0,1],[-1,-1,-1,1,1],[-1,1,-1,1,0],[1,1,-1,0,0]] },
    { n:[-1, 0, 0], verts:[[-1,-1,-1,0,1],[-1,-1,1,1,1],[-1,1,1,1,0],[-1,1,-1,0,0]] },
    { n:[ 1, 0, 0], verts:[[1,-1,1,0,1],[1,-1,-1,1,1],[1,1,-1,1,0],[1,1,1,0,0]] },
    { n:[ 0, 1, 0], verts:[[-1,1,1,0,1],[1,1,1,1,1],[1,1,-1,1,0],[-1,1,-1,0,0]] },
    { n:[ 0,-1, 0], verts:[[-1,-1,-1,0,1],[1,-1,-1,1,1],[1,-1,1,1,0],[-1,-1,1,0,0]] },
  ];
  const verts: number[] = [], idxs: number[] = [];
  let base = 0;
  for (const face of faces) {
    for (const v of face.verts) verts.push(v[0],v[1],v[2],...face.n,v[3],v[4]);
    idxs.push(base,base+1,base+2, base,base+2,base+3);
    base += 4;
  }
  const data: number[] = [];
  for (const i of idxs) { const v=i*8; for (let k=0;k<8;k++) data.push(verts[v+k]); }
  return new Float32Array(data);
}

function generateSphere(stacks=64, slices=64): Float32Array {
  const data: number[] = [];
  for (let i=0; i<stacks; i++) {
    const phi0=(i/stacks)*Math.PI, phi1=((i+1)/stacks)*Math.PI;
    for (let j=0; j<slices; j++) {
      const t0=(j/slices)*2*Math.PI, t1=((j+1)/slices)*2*Math.PI;
      const p=[
        [Math.sin(phi0)*Math.cos(t0),Math.cos(phi0),Math.sin(phi0)*Math.sin(t0)],
        [Math.sin(phi0)*Math.cos(t1),Math.cos(phi0),Math.sin(phi0)*Math.sin(t1)],
        [Math.sin(phi1)*Math.cos(t1),Math.cos(phi1),Math.sin(phi1)*Math.sin(t1)],
        [Math.sin(phi1)*Math.cos(t0),Math.cos(phi1),Math.sin(phi1)*Math.sin(t0)],
      ];
      const uv=[[j/slices,i/stacks],[(j+1)/slices,i/stacks],[(j+1)/slices,(i+1)/stacks],[j/slices,(i+1)/stacks]];
      const push=(idx:number)=>{ const [x,y,z]=p[idx]; data.push(x,y,z,x,y,z,uv[idx][0],uv[idx][1]); };
      push(0);push(2);push(1);
      push(0);push(3);push(2);
    }
  }
  return new Float32Array(data);
}


export function parseOBJ(src: string): { data: Float32Array; center: Vec3; radius: number } {
  const pos: number[][] = [], norm: number[][] = [], uvs: number[][] = [], out: number[] = [];
  for (const raw of src.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const p = line.split(/\s+/);
    if (p[0]==="v")  pos.push([+p[1],+p[2],+p[3]]);
    if (p[0]==="vn") norm.push([+p[1],+p[2],+p[3]]);
    if (p[0]==="vt") uvs.push([+p[1],1-(+p[2])]);
    if (p[0]==="f") {
      const verts = p.slice(1).map(tok => {
        const [pi,ti,ni] = tok.split("/").map(s=>s?parseInt(s)-1:-1);
        return { pi, ti, ni };
      });
      for (let i=1; i+1<verts.length; i++) {
        for (const vt of [verts[0],verts[i],verts[i+1]]) {
          const pp = pos[vt.pi]??[0,0,0];
          const n  = vt.ni>=0 ? norm[vt.ni] : [0,1,0];
          const uv = vt.ti>=0 ? uvs[vt.ti]  : [0,0];
          out.push(pp[0],pp[1],pp[2], n[0],n[1],n[2], uv[0],uv[1]);
        }
      }
    }
  }

  const vCount = out.length / 8;
  let cx=0,cy=0,cz=0;
  for (let i=0;i<vCount;i++) { cx+=out[i*8]; cy+=out[i*8+1]; cz+=out[i*8+2]; }
  cx/=vCount; cy/=vCount; cz/=vCount;
  let radius=0;
  for (let i=0;i<vCount;i++) {
    out[i*8]-=cx; out[i*8+1]-=cy; out[i*8+2]-=cz;
    const d=Math.hypot(out[i*8],out[i*8+1],out[i*8+2]);
    if (d>radius) radius=d;
  }
  return { data: new Float32Array(out), center:[cx,cy,cz], radius };
}


function injectBarycentric(src: Float32Array): Float32Array {
  const triCount = src.length / 8 / 3;
  const dst = new Float32Array(triCount * 3 * 11);
  const bary = [[1,0,0],[0,1,0],[0,0,1]];
  for (let t=0;t<triCount;t++) {
    for (let v=0;v<3;v++) {
      const si=(t*3+v)*8, di=(t*3+v)*11;
      dst[di+0]=src[si+0]; dst[di+1]=src[si+1]; dst[di+2]=src[si+2]; // pos
      dst[di+3]=src[si+3]; dst[di+4]=src[si+4]; dst[di+5]=src[si+5]; // norm
      dst[di+6]=bary[v][0]; dst[di+7]=bary[v][1]; dst[di+8]=bary[v][2]; // bary
      dst[di+9]=src[si+6];  dst[di+10]=src[si+7];                        // uv
    }
  }
  return dst;
}


function makeVBuf(data: Float32Array) {
  const buf = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, data);
  return { buf, count: data.length / 11 };
}
const cubeGeo   = makeVBuf(injectBarycentric(generateCube()));
const sphereGeo = makeVBuf(injectBarycentric(generateSphere()));
export function makeCustomGeo(data: Float32Array) { return makeVBuf(injectBarycentric(data)); }
export function getGeometry(shape: "cube"|"sphere") { return shape==="cube" ? cubeGeo : sphereGeo; }


function makeWhiteTexture(): GPUTexture {
  const tex = device.createTexture({ size:[1,1], format:"rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture:tex }, new Uint8Array([255,255,255,255]), { bytesPerRow:4 }, [1,1]);
  return tex;
}
const defaultSampler = device.createSampler({ magFilter:"linear", minFilter:"linear",
  addressModeU:"repeat", addressModeV:"repeat" });


const UNIFORM_SIZE = 288;
const bgl = device.createBindGroupLayout({ entries:[
  { binding:0, visibility: GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT, buffer:{type:"uniform"} },
  { binding:1, visibility: GPUShaderStage.FRAGMENT, sampler:{type:"filtering"} },
  { binding:2, visibility: GPUShaderStage.FRAGMENT, texture:{sampleType:"float"} },
]});


const shader = device.createShaderModule({ code: shaderCode });
const pipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts:[bgl] }),
  vertex: {
    module: shader, entryPoint:"vs_main",
    buffers:[{ arrayStride:44, attributes:[
      { shaderLocation:0, offset: 0, format:"float32x3" }, // position
      { shaderLocation:1, offset:12, format:"float32x3" }, // normal
      { shaderLocation:2, offset:24, format:"float32x3" }, // barycentric
      { shaderLocation:3, offset:36, format:"float32x2" }, // uv
    ]}],
  },
  fragment:{ module:shader, entryPoint:"fs_main", targets:[{format}] },
  primitive:{ topology:"triangle-list", cullMode:"back" },
  depthStencil:{ format:"depth24plus", depthWriteEnabled:true, depthCompare:"less" },
});

export function makeObjectGPU(texture?: GPUTexture) {
  const uab        = new ArrayBuffer(UNIFORM_SIZE);
  const uniformBuf = device.createBuffer({ size:UNIFORM_SIZE, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
  const tex        = texture ?? makeWhiteTexture();
  const bindGroup  = device.createBindGroup({ layout:bgl, entries:[
    { binding:0, resource:{ buffer:uniformBuf } },
    { binding:1, resource: defaultSampler },
    { binding:2, resource: tex.createView() },
  ]});
  return { uniformBuf, bindGroup, uab, uf32:new Float32Array(uab), uu32:new Uint32Array(uab), tex };
}


export async function uploadTexture(file: File, obj: ReturnType<typeof import("./gui").getObjects>[0]) {
  const bitmap = await createImageBitmap(file, { colorSpaceConversion:"none" });
  const tex = device.createTexture({ size:[bitmap.width, bitmap.height], format:"rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT });
  device.queue.copyExternalImageToTexture({ source:bitmap }, { texture:tex }, [bitmap.width, bitmap.height]);
  // Rebuild bind group with new texture
  obj.gpu.tex.destroy();
  obj.gpu.bindGroup = device.createBindGroup({ layout:bgl, entries:[
    { binding:0, resource:{ buffer:obj.gpu.uniformBuf } },
    { binding:1, resource: defaultSampler },
    { binding:2, resource: tex.createView() },
  ]});
  (obj.gpu as Record<string,unknown>).tex = tex;
}


type Quat = [number,number,number,number]; // [x,y,z,w]

function qIdentity(): Quat { return [0,0,0,1]; }

function qMul(a: Quat, b: Quat): Quat {
  const [ax,ay,az,aw] = a, [bx,by,bz,bw] = b;
  return [
    aw*bx + ax*bw + ay*bz - az*by,
    aw*by - ax*bz + ay*bw + az*bx,
    aw*bz + ax*by - ay*bx + az*bw,
    aw*bw - ax*bx - ay*by - az*bz,
  ];
}

function qNorm(q: Quat): Quat {
  const l = Math.hypot(q[0],q[1],q[2],q[3]) || 1;
  return [q[0]/l, q[1]/l, q[2]/l, q[3]/l];
}

function qToMat4(q: Quat): Float32Array {
  const [x,y,z,w] = q;
  const m = new Float32Array(16);
  m[0]=1-2*(y*y+z*z); m[4]=2*(x*y-z*w);   m[8] =2*(x*z+y*w);   m[12]=0;
  m[1]=2*(x*y+z*w);   m[5]=1-2*(x*x+z*z); m[9] =2*(y*z-x*w);   m[13]=0;
  m[2]=2*(x*z-y*w);   m[6]=2*(y*z+x*w);   m[10]=1-2*(x*x+y*y); m[14]=0;
  m[3]=0;             m[7]=0;              m[11]=0;              m[15]=1;
  return m;
}


function projectOnSphere(nx: number, ny: number): [number,number,number] {
  const r2 = nx*nx + ny*ny;
  if (r2 <= 1.0) return [nx, ny, Math.sqrt(1 - r2)];
  const r = Math.sqrt(r2);
  return [nx/r, ny/r, 0];
}

function arcballQuat(p1: [number,number,number], p2: [number,number,number]): Quat {
  
  const axis: [number,number,number] = [
    p1[1]*p2[2] - p1[2]*p2[1],
    p1[2]*p2[0] - p1[0]*p2[2],
    p1[0]*p2[1] - p1[1]*p2[0],
  ];
  const dot  = Math.min(1, p1[0]*p2[0]+p1[1]*p2[1]+p1[2]*p2[2]);
  const angle = Math.acos(dot);
  const s    = Math.sin(angle/2);
  const len  = Math.hypot(...axis) || 1;
  return qNorm([axis[0]/len*s, axis[1]/len*s, axis[2]/len*s, Math.cos(angle/2)]);
}


let camAzimuth   = 0.4;
let camElevation = 0.3;
let camDistance  = 9;


export const clipPlanes = { near: 0.1, far: 100 };

let isDragging = false;
let lastMX=0, lastMY=0;
let arcStart: [number,number,number] | null = null;

function toNDC(clientX: number, clientY: number): [number,number] {
  const r = canvas.getBoundingClientRect();
  return [(clientX-r.left)/r.width*2-1, -((clientY-r.top)/r.height)*2+1];
}

canvas.addEventListener("mousedown", e => {
  if (e.button!==0) return;
  isDragging=true; lastMX=e.clientX; lastMY=e.clientY;
  const [nx,ny] = toNDC(e.clientX, e.clientY);
  arcStart = projectOnSphere(nx, ny);
});
window.addEventListener("mouseup", () => { isDragging=false; arcStart=null; });
window.addEventListener("mousemove", e => {
  if (!isDragging) return;
  const dx=(e.clientX-lastMX)/canvas.clientWidth;
  const dy=(e.clientY-lastMY)/canvas.clientHeight;
  const objects=getObjects(), selIdx=getSelectedIndex();
  const selObj=selIdx>=0&&selIdx<objects.length ? objects[selIdx] : null;

  if (selObj && arcStart) {
  
    const [nx,ny] = toNDC(e.clientX, e.clientY);
    const arcEnd  = projectOnSphere(nx, ny);
    const dq      = arcballQuat(arcStart, arcEnd);
    selObj.quaternion = qNorm(qMul(dq, selObj.quaternion));
    arcStart = arcEnd; // incremental: new start = current end
    syncRotationSliders(selObj);
  } else {
  
    camAzimuth   += dx*Math.PI*2;
    camElevation  = Math.max(-Math.PI/2+0.01, Math.min(Math.PI/2-0.01, camElevation-dy*Math.PI));
  }
  lastMX=e.clientX; lastMY=e.clientY;
});


canvas.addEventListener("wheel", e => {
  e.preventDefault();
  camDistance = Math.max(1, Math.min(200, camDistance*(1+e.deltaY*0.001)));
  
  clipPlanes.near = camDistance * 0.01;
  clipPlanes.far  = camDistance * 20;
}, { passive:false });

export function syncRotationSliders(obj: { quaternion: Quat }) {
  
  const [x,y,z,w] = obj.quaternion;
  const rx = Math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y));
  const ry = Math.asin(Math.max(-1,Math.min(1, 2*(w*y-z*x))));
  const rz = Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
  const set = (id:string, v:number) => {
    const el=document.getElementById(id) as HTMLInputElement|null;
    const vl=document.getElementById(id+"-val");
    if (el)  el.value=String(v);
    if (vl) vl.textContent=v.toFixed(2);
  };
  set("obj-rx",rx); set("obj-ry",ry); set("obj-rz",rz);
}

function getCamera(target: Vec3) {
  const ce=Math.cos(camElevation), se=Math.sin(camElevation);
  const sa=Math.sin(camAzimuth),   ca=Math.cos(camAzimuth);
  const pos: Vec3=[target[0]+camDistance*ce*sa, target[1]+camDistance*se, target[2]+camDistance*ce*ca];
  return { pos, view:mat4.lookAt(pos,target,[0,1,0]) };
}


function buildModel(obj: { position:Vec3; quaternion:Quat; scale:Vec3 }): Float32Array {
  const T  = mat4.translation(obj.position[0], obj.position[1], obj.position[2]);
  const R  = qToMat4(obj.quaternion);
  const S  = mat4.scaling(obj.scale[0], obj.scale[1], obj.scale[2]);
  return mat4.multiply(T, mat4.multiply(R, S));
}


initGUI();


const startTime = performance.now();

function frame(now: number) {
  const t      = (now-startTime)/1000;
  const global = getGlobalState();
  const objects= getObjects();
  const selIdx = getSelectedIndex();

  const selObj = selIdx>=0&&selIdx<objects.length ? objects[selIdx] : null;
  const target: Vec3 = selObj ? selObj.position : [0,0,0];
  const { pos:camPos, view } = getCamera(target);

  const aspect = canvas.width/canvas.height;
  const proj = mat4.perspective((60*Math.PI)/180, aspect, clipPlanes.near, clipPlanes.far);

  let lx=global.lightX, ly=global.lightY, lz=global.lightZ;
  if (global.autoRotLight) { lx=Math.cos(t*0.8)*4.5; lz=Math.sin(t*0.8)*4.5; }
  const [lr,lg,lb]=hexToRgb(global.lightColor);

  const encoder=device.createCommandEncoder();
  const pass=encoder.beginRenderPass({
    colorAttachments:[{ view:context.getCurrentTexture().createView(),
      clearValue:{r:.08,g:.08,b:.12,a:1}, loadOp:"clear", storeOp:"store" }],
    depthStencilAttachment:{ view:depthTexture!.createView(),
      depthClearValue:1, depthLoadOp:"clear", depthStoreOp:"store" },
  });

  pass.setPipeline(pipeline);

  for (let i=0;i<objects.length;i++) {
    const obj=objects[i];
    const geo=obj.customGeo ?? getGeometry(obj.shape);
    const { uf32, uu32, uniformBuf, uab, bindGroup }=obj.gpu;

    const model=buildModel(obj);
    const normM=mat4.normalMatrix(model);
    const mvp  =mat4.multiply(mat4.multiply(proj,view),model);
    const [or,og,ob]=hexToRgb(obj.color);

    uf32.set(mvp,   0);
    uf32.set(model, 16);
    uf32.set(normM, 32);
    uf32[48]=lx;          uf32[49]=ly;          uf32[50]=lz;  uf32[51]=0;
    uf32[52]=lr;          uf32[53]=lg;          uf32[54]=lb;  uf32[55]=0;
    uf32[56]=obj.ambient; uf32[57]=obj.diffuse; uf32[58]=obj.specular; uf32[59]=obj.shininess;
    uf32[60]=camPos[0];   uf32[61]=camPos[1];   uf32[62]=camPos[2];
    uu32[63]=global.modelId;
    uf32[64]=or;          uf32[65]=og;          uf32[66]=ob;
    uf32[67]=t;
    uu32[68]=obj.useTexture ? 1 : 0; 

    device.queue.writeBuffer(uniformBuf,0,uab);
    pass.setBindGroup(0,bindGroup);
    pass.setVertexBuffer(0,geo.buf);
    pass.draw(geo.count);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);