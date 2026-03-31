// shader.wgsl
// The uniform struct and vertex pipeline are already wired up for you.
// model_id values:
//   0 = Flat implemented
//   1 = Gouraud TODO
//   2 = Phong TODO
//   3 = Blinn-Phong TODO
//
// Useful WGSL built-ins:
//   normalize(v) — returns unit vector
//   dot(a, b) — scalar dot product
//   reflect(I, N) — reflects incident vector I around normal N
//   max(a, b) — component-wise max
//   pow(base, exp) — power function
//   dpdx(v), dpdy(v) — screen-space partial derivatives (fragment stage only)
//   cross(a, b)— cross product
// ── Uniform block
struct Uniforms {
  mvp        : mat4x4<f32>,
  model      : mat4x4<f32>,
  normalMat  : mat4x4<f32>,

  lightPos   : vec3<f32>,
  _p0        : f32,

  lightColor : vec3<f32>,
  _p1        : f32,

  ambient    : f32,
  diffuse    : f32,
  specular   : f32,
  shininess  : f32,

  camPos     : vec3<f32>,
  model_id   : u32,

  objectColor : vec3<f32>,
  time        : f32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

// ── Vertex shader I/O
struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
};

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos      : vec3<f32>,
  @location(1) worldNormal   : vec3<f32>,
  @location(2) uv            : vec2<f32>,
  @location(3) gouraudColor  : vec3<f32>,
};

// ── Flat shading (referencia — no modificar)
fn flatShading(fragWorldPos: vec3<f32>) -> vec3<f32> {
  let dx    = dpdx(fragWorldPos);
  let dy    = dpdy(fragWorldPos);
  let faceN = normalize(cross(dx, dy));

  let L = normalize(u.lightPos - fragWorldPos);
  let V = normalize(u.camPos   - fragWorldPos);

  let ambientC = u.ambient * u.lightColor;

  let NdotL    = max(dot(faceN, L), 0.0);
  let diffuseC = u.diffuse * NdotL * u.lightColor;

  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let R     = reflect(-L, faceN);
    let RdotV = max(dot(R, V), 0.0);
    specularC = u.specular * pow(RdotV, u.shininess) * u.lightColor;
  }

  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Gouraud shading
// Se llama UNA VEZ POR VÉRTICE en vs_main.
// La GPU interpola el color resultante entre vértices antes de llegar al fragment shader.
// Efecto: iluminación "suavizada" pero los reflejos especulares pueden verse mal
// porque la interpolación lineal no captura bien la curva especular entre vértices.
fn gouraudLighting(N: vec3<f32>, vertWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - vertWorldPos);  // dirección AL foco de luz
  let V = normalize(u.camPos   - vertWorldPos);  // dirección A la cámara

  // Ambiente: luz base constante
  let ambientC = u.ambient * u.lightColor;

  // Difuso: Lambertian — cuánta luz llega según el ángulo con la normal
  let NdotL    = max(dot(N, L), 0.0);
  let diffuseC = u.diffuse * NdotL * u.lightColor;

  // Especular: reflexión de Phong calculada en el vértice
  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let R     = reflect(-L, N);               // dirección de reflexión perfecta
    let RdotV = max(dot(R, V), 0.0);
    specularC = u.specular * pow(RdotV, u.shininess) * u.lightColor;
  }

  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Phong shading
// Se llama UNA VEZ POR FRAGMENTO en fs_main.
// Usa la normal interpolada por la GPU — mucho más precisa que Gouraud.
// Diferencia clave vs Blinn-Phong: el especular usa reflect(-L, N) y R·V.
fn phongLighting(N: vec3<f32>, fragWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - fragWorldPos);
  let V = normalize(u.camPos   - fragWorldPos);

  // Ambiente
  let ambientC = u.ambient * u.lightColor;

  // Difuso
  let NdotL    = max(dot(N, L), 0.0);
  let diffuseC = u.diffuse * NdotL * u.lightColor;

  // Especular: ángulo entre el rayo reflejado R y la dirección a la cámara V
  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let R     = reflect(-L, N);               // R = reflexión de L alrededor de N
    let RdotV = max(dot(R, V), 0.0);
    specularC = u.specular * pow(RdotV, u.shininess) * u.lightColor;
  }

  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Blinn-Phong shading
// Se llama UNA VEZ POR FRAGMENTO en fs_main.
// Diferencia clave vs Phong: en lugar de R·V usa el half-vector H·N.
// H = normalize(L + V) — vector a medio camino entre luz y cámara.
// Ventaja: más eficiente y físicamente más correcto para ángulos grandes;
// los reflejos especulares se ven más "suaves" y realistas.
fn blinnPhongLighting(N: vec3<f32>, fragWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - fragWorldPos);
  let V = normalize(u.camPos   - fragWorldPos);

  // Ambiente
  let ambientC = u.ambient * u.lightColor;

  // Difuso
  let NdotL    = max(dot(N, L), 0.0);
  let diffuseC = u.diffuse * NdotL * u.lightColor;

  // Especular Blinn-Phong: half-vector H entre L y V
  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let H     = normalize(L + V);             // half-vector — la diferencia vs Phong
    let NdotH = max(dot(N, H), 0.0);
    specularC = u.specular * pow(NdotH, u.shininess) * u.lightColor;
  }

  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Vertex shader
@vertex
fn vs_main(input: VSIn) -> VSOut {
  var out: VSOut;

  let worldPos4    = u.model     * vec4<f32>(input.position, 1.0);
  let worldNormal4 = u.normalMat * vec4<f32>(input.normal,   0.0);

  out.clipPos     = u.mvp * vec4<f32>(input.position, 1.0);
  out.worldPos    = worldPos4.xyz;
  out.worldNormal = normalize(worldNormal4.xyz);
  out.uv          = input.uv;

  // Gouraud: calcular iluminación aquí, la GPU interpola el resultado al fragment
  if u.model_id == 1u {
    out.gouraudColor = gouraudLighting(out.worldNormal, out.worldPos);
  } else {
    out.gouraudColor = vec3<f32>(0.0);
  }

  return out;
}

// ── Fragment shader
@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  var color: vec3<f32>;
  let N = normalize(input.worldNormal);

  switch u.model_id {
    case 0u: {
      color = flatShading(input.worldPos);
    }
    case 1u: {
      color = input.gouraudColor;
    }
    case 2u: {
      color = phongLighting(N, input.worldPos);
    }
    default: {
      color = blinnPhongLighting(N, input.worldPos);
    }
  }

  return vec4<f32>(color, 1.0);
}
