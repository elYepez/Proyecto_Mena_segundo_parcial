/// <reference types="@webgpu/types" />

import "./style.css";
import shaderCode from "./shader.wgsl?raw";
import { Camera } from "./camera";
import { mat4 } from "./math";
import type { Vec3 } from "./math";
import { gui, hexToRgb, initGUI, updateLightDisplay } from "./gui";


//WebGPU init
if (!navigator.gpu) throw new Error("WebGPU not supported");

const canvas = document.querySelector("#gfx-main") as HTMLCanvasElement;
if (!canvas) throw new Error("Canvas #gfx-main not found");

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter found");

const device = await adapter.requestDevice();
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

// ─────────────────────────────────────────────────────────────────────────────
// Vertex format: [x, y, z,  nx, ny, nz,  u, v]
//                 position    normal       uv
// stride = 8 floats = 32 bytes
// ─────────────────────────────────────────────────────────────────────────────

// ── Cube geometry ───────────────────────────────────────────────
// Each face is 2 triangles
// Normals are constant per face so flat and smooth shading look identical on a cube.

function generateCube(): Float32Array {
  const faces: Array<{ n: Vec3; verts: number[][] }> = [
    { n: [ 0,  0,  1], verts: [[-1,-1, 1,0,1],[1,-1, 1,1,1],[1, 1, 1,1,0],[-1,-1, 1,0,1],[1, 1, 1,1,0],[-1, 1, 1,0,0]] },
    { n: [ 0,  0, -1], verts: [[ 1,-1,-1,0,1],[-1,-1,-1,1,1],[-1, 1,-1,1,0],[1,-1,-1,0,1],[-1, 1,-1,1,0],[1, 1,-1,0,0]] },
    { n: [-1,  0,  0], verts: [[-1,-1,-1,0,1],[-1,-1, 1,1,1],[-1, 1, 1,1,0],[-1,-1,-1,0,1],[-1, 1, 1,1,0],[-1, 1,-1,0,0]] },
    { n: [ 1,  0,  0], verts: [[ 1,-1, 1,0,1],[ 1,-1,-1,1,1],[ 1, 1,-1,1,0],[1,-1, 1,0,1],[1, 1,-1,1,0],[1, 1, 1,0,0]] },
    { n: [ 0,  1,  0], verts: [[-1, 1, 1,0,1],[ 1, 1, 1,1,1],[ 1, 1,-1,1,0],[-1, 1, 1,0,1],[1, 1,-1,1,0],[-1, 1,-1,0,0]] },
    { n: [ 0, -1,  0], verts: [[-1,-1,-1,0,1],[ 1,-1,-1,1,1],[ 1,-1, 1,1,0],[-1,-1,-1,0,1],[1,-1, 1,1,0],[-1,-1, 1,0,0]] },
  ];

  const data: number[] = [];
  for (const face of faces) {
    for (const v of face.verts) {
      data.push(v[0], v[1], v[2]);// position
      data.push(...face.n); // normal (same for all verts on a face)
      data.push(v[3], v[4]);// uv
    }
  }
  return new Float32Array(data);
}

function generateSphere(stacks: number, slices: number): Float32Array {
  const data: number[] = [];

  for (let i = 0; i < stacks; i++) {
    const phi0 = (i / stacks) * Math.PI;
    const phi1 = ((i + 1) / stacks) * Math.PI;

    for (let j = 0; j < slices; j++) {
      const theta0 = (j / slices) * 2 * Math.PI;
      const theta1 = ((j + 1) / slices) * 2 * Math.PI;

      // 4 esquinas del quad
      const p = [
        [Math.sin(phi0) * Math.cos(theta0), Math.cos(phi0), Math.sin(phi0) * Math.sin(theta0)],
        [Math.sin(phi0) * Math.cos(theta1), Math.cos(phi0), Math.sin(phi0) * Math.sin(theta1)],
        [Math.sin(phi1) * Math.cos(theta1), Math.cos(phi1), Math.sin(phi1) * Math.sin(theta1)],
        [Math.sin(phi1) * Math.cos(theta0), Math.cos(phi1), Math.sin(phi1) * Math.sin(theta0)],
      ];

      const uv = [
        [j / slices,       i / stacks],
        [(j + 1) / slices, i / stacks],
        [(j + 1) / slices, (i + 1) / stacks],
        [j / slices,       (i + 1) / stacks],
      ];

      // En una esfera unitaria, la normal es igual a la posición
      const pushVert = (idx: number) => {
        const [x, y, z] = p[idx];
        data.push(x, y, z);              // position
        data.push(x, y, z);              // normal
        data.push(uv[idx][0], uv[idx][1]); // uv
      };

      // Dos triángulos por quad: 0-1-2 y 0-2-3
      pushVert(0); pushVert(1); pushVert(2);
      pushVert(0); pushVert(2); pushVert(3);
    }
  }

  return new Float32Array(data);
}


// Geometry buffers — rebuilt when the user switches shape
let activeShape: "cube" | "sphere" = "cube";

function buildVertexBuffer(shape: "cube" | "sphere"): { buf: GPUBuffer; count: number } {
  const data = shape === "cube" ? generateCube() : generateSphere(64, 64);
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return { buf, count: data.length / 8 };
}

let { buf: vertexBuffer, count: vertexCount } = buildVertexBuffer("cube");


// Uniform buffer  structure
//
// Layout (byte offsets):
//   0   mvp        mat4   64 B
//   64  model      mat4   64 B
//   128 normalMat  mat4   64 B
//   192 lightPos   vec3   12 B  + 4 pad
//   208 lightColor vec3   12 B  + 4 pad
//   224 ambient    f32     4 B
//   228 diffuse    f32     4 B
//   232 specular   f32     4 B
//   236 shininess  f32     4 B
//   240 camPos     vec3   12 B
//   252 model_id   u32     4 B  ← packed with camPos pad
//   256 objectColor vec3  12 B
//   268 time       f32     4 B
// ─────────────────────────────────────────────────────────────────────────────
const UNIFORM_SIZE = 288;

const uniformBuffer = device.createBuffer({
  size: UNIFORM_SIZE,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const uArrayBuf = new ArrayBuffer(UNIFORM_SIZE);
const uData     = new Float32Array(uArrayBuf);
const uData32   = new Uint32Array(uArrayBuf);

// Pipeline
const shader = device.createShaderModule({ label: "Lighting Shader", code: shaderCode });

const pipeline = device.createRenderPipeline({
  label: "Lighting Pipeline",
  layout: "auto",
  vertex: {
    module: shader,
    entryPoint: "vs_main",
    buffers: [{
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0,     format: "float32x3" }, // position
        { shaderLocation: 1, offset: 3 * 4, format: "float32x3" }, // normal
        { shaderLocation: 2, offset: 6 * 4, format: "float32x2" }, // uv
      ],
    }],
  },
  fragment: { module: shader, entryPoint: "fs_main", targets: [{ format }] },
  primitive: { topology: "triangle-list", cullMode: "back" },
  depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
});


// GUI
initGUI(shape => {
  activeShape = shape;
  vertexBuffer.destroy();
  ({ buf: vertexBuffer, count: vertexCount } = buildVertexBuffer(shape));
});

// Camera
const camera = new Camera();
camera.position = [0, 0, 5];
const keys = new Set<string>();
window.addEventListener("keydown", e => keys.add(e.key));
window.addEventListener("keyup",   e => keys.delete(e.key));


// Render loop
let lastTime    = performance.now();
const startTime = performance.now();

function frame(now: number) {
  const dt = Math.min(0.033, (now - lastTime) / 1000);
  lastTime = now;
  const t  = (now - startTime) / 1000;

  camera.update(keys, dt);

  const aspect = canvas.width / canvas.height;
  const proj   = mat4.perspective((60 * Math.PI) / 180, aspect, 0.1, 100);
  const view   = camera.getViewMatrix();
  const model  = mat4.identity();
  const normM  = mat4.normalMatrix(model);
  const mvp    = mat4.multiply(mat4.multiply(proj, view), model);

  let lx = gui.lightX, ly = gui.lightY, lz = gui.lightZ;
  if (gui.autoRotLight) {
    lx = Math.cos(t * 0.8) * 4.5;
    lz = Math.sin(t * 0.8) * 4.5;
    updateLightDisplay(lx, lz);
  }

  const [or, og, ob] = hexToRgb(gui.objectColor);
  const [lr, lg, lb] = hexToRgb(gui.lightColor);

  uData.set(mvp,   0);
  uData.set(model, 16);
  uData.set(normM, 32);
  uData[48] = lx;          uData[49] = ly;          uData[50] = lz; uData[51] = 0;
  uData[52] = lr;          uData[53] = lg;           uData[54] = lb; uData[55] = 0;
  uData[56] = gui.ambient; uData[57] = gui.diffuse;  uData[58] = gui.specular; uData[59] = gui.shininess;
  uData[60] = camera.position[0]; uData[61] = camera.position[1]; uData[62] = camera.position[2];
  uData32[63] = gui.modelId;//<-must be u32 bits
  uData[64] = or; uData[65] = og; uData[66] = ob;
  uData[67] = t;

  device.queue.writeBuffer(uniformBuffer, 0, uArrayBuf);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexture!.createView(),
      depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store",
    },
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount);
  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
