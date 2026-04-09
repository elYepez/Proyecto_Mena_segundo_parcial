
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
  use_texture : u32,        
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

fn sphericalUV(N: vec3<f32>) -> vec2<f32> {
  let u_coord = atan2(N.z, N.x) / (2.0 * 3.14159265) + 0.5;
  let v_coord = asin(clamp(N.y, -1.0, 1.0)) / 3.14159265 + 0.5;
  return vec2<f32>(u_coord, v_coord);
}


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

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(input.worldNormal);
  var color: vec3<f32>;

  switch u.model_id {

    case 0u: { color = flatShading(input.worldPos); }


    case 1u: { color = input.gouraudColor; }


    case 2u: { color = lighting(N, input.worldPos, u.objectColor, false); }

 
    case 3u: { color = lighting(N, input.worldPos, u.objectColor, true); }


    case 4u: { color = N * 0.5 + vec3<f32>(0.5); }


    case 5u: {
      let e = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
      color = vec3<f32>(1.0 - (1.0 - smoothstep(0.0, 0.02, e)));
    }

 
    case 6u: { color = vec3<f32>((input.ndcDepth + 1.0) * 0.5); }

    default: {
      var uv2 = input.uv;
      if u.use_texture == 1u {
        uv2 = sphericalUV(N);
      }
      let texColor = textureSample(texImg, texSamp, uv2).rgb;
      let baseColor = select(u.objectColor, texColor, u.use_texture == 1u);
      color = lighting(N, input.worldPos, baseColor, false);
    }
  }

  return vec4<f32>(color, 1.0);
}
