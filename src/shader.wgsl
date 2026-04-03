// shader.wgsl
// model_id:
//   0 = Flat         4 = Normal Buffer
//   1 = Gouraud      5 = Wireframe
//   2 = Phong        6 = Depth
//   3 = Blinn-Phong  7 = Texture (spherical UV × Phong)
//
// Stride: [pos3, norm3, bary3, uv2] = 11 floats = 44 bytes

struct Uniforms {
  mvp        : mat4x4<f32>,
  model      : mat4x4<f32>,
  normalMat  : mat4x4<f32>,
  lightPos   : vec3<f32>,  _p0 : f32,
  lightColor : vec3<f32>,  _p1 : f32,
  ambient    : f32,
  diffuse    : f32,
  specular   : f32,
  shininess  : f32,
  camPos     : vec3<f32>,
  model_id   : u32,
  objectColor : vec3<f32>,
  time        : f32,
  use_texture : u32,        // 1 = sample texture, 0 = use objectColor
};

@group(0) @binding(0) var<uniform> u       : Uniforms;
@group(0) @binding(1) var          texSamp : sampler;
@group(0) @binding(2) var          texImg  : texture_2d<f32>;

struct VSIn {
  @location(0) position    : vec3<f32>,
  @location(1) normal      : vec3<f32>,
  @location(2) barycentric : vec3<f32>,
  @location(3) uv          : vec2<f32>,
};

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos      : vec3<f32>,
  @location(1) worldNormal   : vec3<f32>,
  @location(2) barycentric   : vec3<f32>,
  @location(3) uv            : vec2<f32>,
  @location(4) gouraudColor  : vec3<f32>,
  @location(5) ndcDepth      : f32,
};

// ── Lighting ──────────────────────────────────────────────────────────────────
fn phongSpec(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>) -> f32 {
  let R = reflect(-L, N);
  return pow(max(dot(R, V), 0.0), u.shininess);
}

fn blinnSpec(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>) -> f32 {
  let H = normalize(L + V);
  return pow(max(dot(N, H), 0.0), u.shininess);
}

fn lighting(N: vec3<f32>, worldPos: vec3<f32>, baseColor: vec3<f32>, blinn: bool) -> vec3<f32> {
  let L     = normalize(u.lightPos - worldPos);
  let V     = normalize(u.camPos   - worldPos);
  let NdotL = max(dot(N, L), 0.0);
  let amb   = u.ambient  * u.lightColor;
  let diff  = u.diffuse  * NdotL * u.lightColor;
  var spec  = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let s = select(phongSpec(N,L,V), blinnSpec(N,L,V), blinn);
    spec  = u.specular * s * u.lightColor;
  }
  return (amb + diff + spec) * baseColor;
}

fn flatShading(fragWorldPos: vec3<f32>) -> vec3<f32> {
  let faceN = normalize(cross(dpdx(fragWorldPos), dpdy(fragWorldPos)));
  return lighting(faceN, fragWorldPos, u.objectColor, false);
}

// ── Spherical UV mapping ──────────────────────────────────────────────────────
// Maps a world-space normal direction to (u,v) in [0,1]² using spherical coords.
// u = azimuth  (atan2(z,x) / 2π + 0.5)
// v = elevation (asin(y)   / π   + 0.5)
// This gives a continuous UV parameterization over the whole surface,
// independent of the mesh's UV attributes — useful for any shape (sphere, OBJ, cube).
fn sphericalUV(N: vec3<f32>) -> vec2<f32> {
  let u_coord = atan2(N.z, N.x) / (2.0 * 3.14159265) + 0.5;
  let v_coord = asin(clamp(N.y, -1.0, 1.0)) / 3.14159265 + 0.5;
  return vec2<f32>(u_coord, v_coord);
}

// ── Vertex shader ─────────────────────────────────────────────────────────────
@vertex
fn vs_main(input: VSIn) -> VSOut {
  var out: VSOut;
  let wp4  = u.model     * vec4<f32>(input.position, 1.0);
  let wn4  = u.normalMat * vec4<f32>(input.normal,   0.0);
  let clip = u.mvp       * vec4<f32>(input.position, 1.0);

  out.clipPos     = clip;
  out.worldPos    = wp4.xyz;
  out.worldNormal = normalize(wn4.xyz);
  out.barycentric = input.barycentric;
  out.uv          = input.uv;
  out.ndcDepth    = clip.z / clip.w;

  if u.model_id == 1u {
    out.gouraudColor = lighting(out.worldNormal, out.worldPos, u.objectColor, false);
  } else {
    out.gouraudColor = vec3<f32>(0.0);
  }
  return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(input.worldNormal);
  var color: vec3<f32>;

  switch u.model_id {

    // 0 — Flat: una normal por triángulo, derivada de dpdx/dpdy
    case 0u: { color = flatShading(input.worldPos); }

    // 1 — Gouraud: iluminación calculada por vértice, GPU interpola el color
    case 1u: { color = input.gouraudColor; }

    // 2 — Phong: normal interpolada por fragmento, especular R·V
    case 2u: { color = lighting(N, input.worldPos, u.objectColor, false); }

    // 3 — Blinn-Phong: igual que Phong pero especular H·N (half-vector)
    case 3u: { color = lighting(N, input.worldPos, u.objectColor, true); }

    // 4 — Normal buffer: normal world-space como RGB remapeada [-1,1]→[0,1]
    //     R=+X(derecha)  G=+Y(arriba)  B=+Z(hacia cámara)
    //     La interpolación baricéntrica del hardware da la normal suavizada por fragmento.
    case 4u: { color = N * 0.5 + vec3<f32>(0.5); }

    // 5 — Wireframe con hidden surface removal
    //     λ0,λ1,λ2 son [1,0,0][0,1,0][0,0,1] en los vértices, interpolados por la GPU.
    //     min(λ) ≈ 0 → borde → negro; min(λ) > 0 → relleno → blanco.
    //     El z-buffer elimina superficies ocultas automáticamente.
    case 5u: {
      let e = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
      color = vec3<f32>(1.0 - (1.0 - smoothstep(0.0, 0.02, e)));
    }

    // 6 — Depth: z/w remapeado [−1,1]→[0,1], visualiza el z-buffer como gris
    case 6u: { color = vec3<f32>((input.ndcDepth + 1.0) * 0.5); }

    // 7 — Textura con UV esférico × iluminación Phong
    //     Las coordenadas UV se calculan a partir de la normal world-space en esferal coords:
    //       u = atan2(Nz, Nx) / 2π + 0.5   (azimuth)
    //       v = asin(Ny)      / π  + 0.5   (elevation)
    //     Esto da un mapeo continuo sobre cualquier malla, no solo esferas.
    //     La textura se multiplica por la iluminación Phong para que reaccione a la luz.
    default: {
      var uv2 = input.uv;
      if u.use_texture == 1u {
        // Override UVs with spherical mapping from world-space normal
        uv2 = sphericalUV(N);
      }
      let texColor = textureSample(texImg, texSamp, uv2).rgb;
      let baseColor = select(u.objectColor, texColor, u.use_texture == 1u);
      color = lighting(N, input.worldPos, baseColor, false);
    }
  }

  return vec4<f32>(color, 1.0);
}
