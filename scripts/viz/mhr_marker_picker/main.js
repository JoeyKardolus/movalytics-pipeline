/**
 * MHR Mesh ↔ OpenSim Markers: interactive vertex labeling tool.
 *
 * Adapted from SAM4Dcap-core smpl_mapper viewer (mesh-agnostic).
 * Takes {vertices, faces} JSON for any body mesh, displays alongside
 * OpenSim skeleton + markers, lets user click-to-assign vertex indices.
 *
 * Three.js r0.160.0 via importmap.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

const elements = {
  opencapScenePath: document.getElementById("opencapScenePath"),
  smplMeshPath: document.getElementById("smplMeshPath"),
  load: document.getElementById("load"),
  exportJson: document.getElementById("exportJson"),
  exportCsv: document.getElementById("exportCsv"),
  importJson: document.getElementById("importJson"),
  meta: document.getElementById("meta"),

  layoutMode: document.getElementById("layoutMode"),
  splitDistance: document.getElementById("splitDistance"),
  smplYawPreset: document.getElementById("smplYawPreset"),
  lockScaleToBsm: document.getElementById("lockScaleToBsm"),
  bboxAlign: document.getElementById("bboxAlign"),
  fitAlign: document.getElementById("fitAlign"),
  resetSmpl: document.getElementById("resetSmpl"),
  fitCamera: document.getElementById("fitCamera"),

  showOpenCapMesh: document.getElementById("showOpenCapMesh"),
  showOpenCapSkeleton: document.getElementById("showOpenCapSkeleton"),
  showOpenCapMarkers: document.getElementById("showOpenCapMarkers"),
  showOpenCapLabels: document.getElementById("showOpenCapLabels"),
  showSmplMesh: document.getElementById("showSmplMesh"),
  showSmplWireframe: document.getElementById("showSmplWireframe"),
  showMappedPoints: document.getElementById("showMappedPoints"),
  showGrid: document.getElementById("showGrid"),
  showAxes: document.getElementById("showAxes"),

  markerSize: document.getElementById("markerSize"),
  markerFilter: document.getElementById("markerFilter"),
  markerList: document.getElementById("markerList"),
  markerCountTitle: document.getElementById("markerCountTitle"),

  selectedMarker: document.getElementById("selectedMarker"),
  selectedSmpl: document.getElementById("selectedSmpl"),
  clearCurrent: document.getElementById("clearCurrent"),
  clearAll: document.getElementById("clearAll"),
  autoAdvance: document.getElementById("autoAdvance"),
  autoSymmetry: document.getElementById("autoSymmetry"),
  autoSite: document.getElementById("autoSite"),
  loadAtlas: document.getElementById("loadAtlas"),

  viewer: document.getElementById("viewer"),

  scaleGlobal: document.getElementById("scaleGlobal"),
  scaleGlobalVal: document.getElementById("scaleGlobalVal"),
  scaleTorso: document.getElementById("scaleTorso"),
  scaleTorsoVal: document.getElementById("scaleTorsoVal"),
  scaleHumerus: document.getElementById("scaleHumerus"),
  scaleHumerusVal: document.getElementById("scaleHumerusVal"),
  scaleUlna: document.getElementById("scaleUlna"),
  scaleUlnaVal: document.getElementById("scaleUlnaVal"),
  scaleHand: document.getElementById("scaleHand"),
  scaleHandVal: document.getElementById("scaleHandVal"),
  scaleFemur: document.getElementById("scaleFemur"),
  scaleFemurVal: document.getElementById("scaleFemurVal"),
  scaleTibia: document.getElementById("scaleTibia"),
  scaleTibiaVal: document.getElementById("scaleTibiaVal"),
  scaleClavicle: document.getElementById("scaleClavicle"),
  scaleClavicleVal: document.getElementById("scaleClavicleVal"),
  scaleFoot: document.getElementById("scaleFoot"),
  scaleFootVal: document.getElementById("scaleFootVal"),
};

function setMeta(text) {
  elements.meta.textContent = text;
}

function makeRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.replaceChildren(renderer.domElement);
  return renderer;
}

function makeLabelRenderer(container) {
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
  labelRenderer.domElement.style.position = "absolute";
  labelRenderer.domElement.style.top = "0";
  labelRenderer.domElement.style.left = "0";
  labelRenderer.domElement.style.pointerEvents = "none";
  container.appendChild(labelRenderer.domElement);
  return labelRenderer;
}

function resize(renderer, labelRenderer, camera, container) {
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  labelRenderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

async function loadJson(url, hint) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${hint} fetch failed: ${res.status} ${res.statusText} (${url})`);
  return res.json();
}

function computeBounds(points) {
  const box = new THREE.Box3();
  for (const p of points) box.expandByPoint(p);
  return box;
}

function fitCameraToBox(camera, controls, box) {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.6 + 0.2;

  controls.target.copy(center);
  controls.update();

  const dir = new THREE.Vector3(1, 0.6, 1).normalize();
  camera.position.copy(center.clone().addScaledVector(dir, radius * 3.0));
  camera.near = Math.max(0.01, radius / 200);
  camera.far = Math.max(1000, radius * 50);
  camera.updateProjectionMatrix();
}

function buildScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0f14);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  camera.position.set(2.5, 1.6, 2.5);

  const renderer = makeRenderer(container);
  const labelRenderer = makeLabelRenderer(container);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.0, 1.0, 0.0);
  controls.enableZoom = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.update();

  // Manual zoom — clamp scroll delta for consistent speed
  renderer.domElement.addEventListener("wheel", (e) => {
    e.preventDefault();
    const dir = e.deltaY > 0 ? 1 : -1;
    const dist = camera.position.distanceTo(controls.target);
    const step = dist * 0.04; // 4% of current distance per tick
    const offset = camera.position.clone().sub(controls.target).normalize();
    camera.position.addScaledVector(offset, dir * step);
  }, { passive: false });

  const light1 = new THREE.DirectionalLight(0xffffff, 1.05);
  light1.position.set(3, 6, 2);
  scene.add(light1);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  const grid = new THREE.GridHelper(10, 20, 0x2b3340, 0x1b2230);
  grid.name = "grid";
  scene.add(grid);

  const axes = new THREE.AxesHelper(0.3);
  axes.name = "axes";
  scene.add(axes);

  const root = new THREE.Group();
  root.name = "root";
  scene.add(root);

  const openCapLayout = new THREE.Group();
  openCapLayout.name = "opencapLayout";
  const openCap = new THREE.Group();
  openCap.name = "opencap";
  openCapLayout.add(openCap);
  root.add(openCapLayout);

  const smplLayout = new THREE.Group();
  smplLayout.name = "mhrLayout";
  const smpl = new THREE.Group();
  smpl.name = "mhrAligned";
  smplLayout.add(smpl);
  root.add(smplLayout);

  window.addEventListener("resize", () => resize(renderer, labelRenderer, camera, container));
  resize(renderer, labelRenderer, camera, container);

  return { scene, camera, renderer, labelRenderer, controls, grid, axes, root, openCapLayout, openCap, smplLayout, smpl };
}

const GROUP_COLORS = {
  "Pelvis": 0x8bd3ff,
  "Knee": 0x7cffa7,
  "Ankle": 0x66e0a0,
  "Foot": 0x66e0a0,
  "Shoulder": 0xffc27a,
  "Spine": 0xb8a0ff,
  "Elbow": 0xffc27a,
  "Wrist": 0xffc27a,
  "Thigh Cluster": 0x7cffa7,
  "Shank Cluster": 0x7cffa7,
  // Legacy (backwards-compatible)
  "Waist": 0x8bd3ff,
  "Thigh": 0x7cffa7,
  "Shank": 0x7cffa7,
  "Torso": 0xb8a0ff,
  "Scapula": 0xb8a0ff,
  "Neck": 0xd6d6d6,
  "Head": 0xd6d6d6,
  "Upper Arm": 0xffc27a,
  "Forearm": 0xffc27a,
};

function markerColor(name, group) {
  if (group && GROUP_COLORS[group]) return GROUP_COLORS[group];
  return 0xd6d6d6;
}

function buildOpenCapSkeleton(root, bodies, edges) {
  const group = new THREE.Group();
  group.name = "opencapSkeleton";

  const validEdges = (edges || []).filter(([a, b]) => bodies.has(a) && bodies.has(b));

  const positions = new Float32Array(validEdges.length * 2 * 3);
  for (let i = 0; i < validEdges.length; i++) {
    const [a, b] = validEdges[i];
    const pa = bodies.get(a);
    const pb = bodies.get(b);
    const base = i * 2 * 3;
    positions[base + 0] = pa.x;
    positions[base + 1] = pa.y;
    positions[base + 2] = pa.z;
    positions[base + 3] = pb.x;
    positions[base + 4] = pb.y;
    positions[base + 5] = pb.z;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x66c2ff });
  const lines = new THREE.LineSegments(geometry, material);
  group.add(lines);

  root.add(group);
  return { group };
}

function buildOpenCapMarkers(root, markers, markerRadius, showLabels) {
  const group = new THREE.Group();
  group.name = "opencapMarkers";

  const sphere = new THREE.SphereGeometry(1, 12, 10);
  const markerMap = new Map();

  for (const { name, position, group: segGroup } of markers || []) {
    const color = markerColor(name, segGroup);
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.65,
      metalness: 0.05,
      emissive: 0x000000,
    });
    const mesh = new THREE.Mesh(sphere, material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.scale.setScalar(markerRadius);
    mesh.name = `marker:${name}`;
    mesh.userData.markerName = name;
    mesh.userData.segGroup = segGroup || "Other";

    const labelEl = document.createElement("div");
    labelEl.className = "label";
    labelEl.textContent = name;
    const label = new CSS2DObject(labelEl);
    label.position.set(0, markerRadius * 1.2, 0);
    label.visible = Boolean(showLabels);
    mesh.add(label);

    group.add(mesh);
    markerMap.set(name, { mesh, label, material, segGroup: segGroup || "Other", position: new THREE.Vector3(position[0], position[1], position[2]) });
  }

  root.add(group);
  return { group, markerMap };
}

function buildMhrMesh(root, meshData) {
  const positions = Float32Array.from(meshData.vertices || []);
  const facesRaw = meshData.faces || meshData.triangles || [];
  const faces = Uint32Array.from(facesRaw);
  const index = positions.length / 3 > 65535 ? new Uint32Array(faces) : new Uint16Array(faces);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const material = new THREE.MeshStandardMaterial({
    color: 0xd8c1a3,
    roughness: 0.85,
    metalness: 0.02,
    transparent: true,
    opacity: 0.92,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "mhrMesh";

  root.add(mesh);
  return { mesh, geometry, material, positions, index };
}

function barycentricCoords(a, b, c, p) {
  const v0 = b.clone().sub(a);
  const v1 = c.clone().sub(a);
  const v2 = p.clone().sub(a);
  const d00 = v0.dot(v0);
  const d01 = v0.dot(v1);
  const d11 = v1.dot(v1);
  const d20 = v2.dot(v0);
  const d21 = v2.dot(v1);
  const denom = d00 * d11 - d01 * d01;
  if (Math.abs(denom) < 1e-12) return [1, 0, 0];
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  const u = 1 - v - w;
  return [u, v, w];
}

function downloadText(filename, text, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function fmtVec3(v) {
  return `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
}

// ── Similarity transform (Procrustes) ──

function fitSimilarityTransform(sourcePoints, targetPoints) {
  const n = Math.min(sourcePoints.length, targetPoints.length);
  if (n < 3) throw new Error("Alignment requires at least 3 mapped points");

  const muX = new THREE.Vector3();
  const muY = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    muX.add(sourcePoints[i]);
    muY.add(targetPoints[i]);
  }
  muX.multiplyScalar(1 / n);
  muY.multiplyScalar(1 / n);

  let varX = 0;
  const H = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < n; i++) {
    const x = sourcePoints[i].clone().sub(muX);
    const y = targetPoints[i].clone().sub(muY);
    varX += x.lengthSq();
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++)
        H[r][c] += y.getComponent(r) * x.getComponent(c);
  }
  varX /= n;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r][c] /= n;
  if (varX < 1e-12) throw new Error("Alignment failed: source variance too small");

  // Use Three.js Matrix4 for SVD-like alignment via polar decomposition
  const Hm = new THREE.Matrix3();
  Hm.set(H[0][0], H[0][1], H[0][2], H[1][0], H[1][1], H[1][2], H[2][0], H[2][1], H[2][2]);

  // Simple iterative approach: use the cross-covariance directly
  // For robustness, compute R via quaternion from correlation matrix
  const m4 = new THREE.Matrix4();
  m4.set(H[0][0], H[0][1], H[0][2], 0, H[1][0], H[1][1], H[1][2], 0, H[2][0], H[2][1], H[2][2], 0, 0, 0, 0, 1);

  // Extract rotation using polar decomposition (approximate)
  // R = H * (H^T * H)^{-1/2}
  const HtH = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        HtH[i][j] += H[k][i] * H[k][j];

  // Use eigendecomposition of H^T*H for SVD
  const { U, S, V } = svd3x3_simple(H);
  const Vt = transpose3(V);
  let R = mul3(U, Vt);
  const detR = det3(R);
  if (detR < 0) {
    const Sfix = [[1,0,0],[0,1,0],[0,0,-1]];
    R = mul3(mul3(U, Sfix), Vt);
  }

  const traceDS = S[0] + S[1] + (detR < 0 ? -S[2] : S[2]);
  const scale = traceDS / varX;

  const rmuX = matVec3(R, [muX.x, muX.y, muX.z]);
  const t = new THREE.Vector3(muY.x - scale * rmuX[0], muY.y - scale * rmuX[1], muY.z - scale * rmuX[2]);

  return { scale, R, t };
}

// ── Linear algebra helpers ──

function transpose3(A) {
  return [[A[0][0],A[1][0],A[2][0]], [A[0][1],A[1][1],A[2][1]], [A[0][2],A[1][2],A[2][2]]];
}

function mul3(A, B) {
  const out = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      out[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j];
  return out;
}

function det3(m) {
  return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}

function matVec3(A, v) {
  return [A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2], A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2], A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]];
}

function normalize3(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}

function cross3(a, b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

function dot3(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

function jacobiEigenSymmetric3(A) {
  const D = [[A[0][0],A[0][1],A[0][2]], [A[1][0],A[1][1],A[1][2]], [A[2][0],A[2][1],A[2][2]]];
  const V = [[1,0,0],[0,1,0],[0,0,1]];

  function rotate(p, q) {
    const apq = D[p][q];
    if (Math.abs(apq) < 1e-12) return;
    const phi = 0.5 * Math.atan2(2*apq, D[q][q] - D[p][p]);
    const c = Math.cos(phi), s = Math.sin(phi);
    for (let i = 0; i < 3; i++) {
      if (i === p || i === q) continue;
      const dip = D[i][p], diq = D[i][q];
      D[i][p] = c*dip - s*diq; D[p][i] = D[i][p];
      D[i][q] = s*dip + c*diq; D[q][i] = D[i][q];
    }
    const app = D[p][p], aqq = D[q][q];
    D[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
    D[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;
    D[p][q] = 0; D[q][p] = 0;
    for (let i = 0; i < 3; i++) {
      const vip = V[i][p], viq = V[i][q];
      V[i][p] = c*vip - s*viq;
      V[i][q] = s*vip + c*viq;
    }
  }

  for (let iter = 0; iter < 32; iter++) {
    let p=0, q=1, mx = Math.abs(D[0][1]);
    if (Math.abs(D[0][2]) > mx) { mx = Math.abs(D[0][2]); p=0; q=2; }
    if (Math.abs(D[1][2]) > mx) { mx = Math.abs(D[1][2]); p=1; q=2; }
    if (mx < 1e-10) break;
    rotate(p, q);
  }
  const ev = [D[0][0], D[1][1], D[2][2]];
  const evec = [[V[0][0],V[1][0],V[2][0]], [V[0][1],V[1][1],V[2][1]], [V[0][2],V[1][2],V[2][2]]];
  const idx = [0,1,2].sort((i,j) => ev[j]-ev[i]);
  return { eigenvalues: idx.map(i=>ev[i]), eigenvectors: idx.map(i=>evec[i]) };
}

function svd3x3_simple(M) {
  const Mt = transpose3(M);
  const A = mul3(Mt, M);
  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric3(A);
  const svals = eigenvalues.map(l => Math.sqrt(Math.max(0, l)));
  const V = [[eigenvectors[0][0],eigenvectors[1][0],eigenvectors[2][0]],
             [eigenvectors[0][1],eigenvectors[1][1],eigenvectors[2][1]],
             [eigenvectors[0][2],eigenvectors[1][2],eigenvectors[2][2]]];
  const Ucols = [];
  for (let i = 0; i < 3; i++) {
    const vi = [V[0][i], V[1][i], V[2][i]];
    const mvi = matVec3(M, vi);
    if (svals[i] > 1e-10) Ucols.push([mvi[0]/svals[i], mvi[1]/svals[i], mvi[2]/svals[i]]);
    else Ucols.push([0,0,0]);
  }
  // Orthonormalize
  const c0 = normalize3(Ucols[0]);
  let c1 = [Ucols[1][0]-dot3(Ucols[1],c0)*c0[0], Ucols[1][1]-dot3(Ucols[1],c0)*c0[1], Ucols[1][2]-dot3(Ucols[1],c0)*c0[2]];
  c1 = normalize3(c1);
  let c2 = cross3(c0, c1);
  c2 = normalize3(c2);
  const U = [[c0[0],c1[0],c2[0]], [c0[1],c1[1],c2[1]], [c0[2],c1[2],c2[2]]];
  return { U, S: svals, V };
}

// ── Main application ──

function start() {
  const viz = buildScene(elements.viewer);
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const clickState = { down: null };

  let current = {
    opencap: { data: null, skeleton: null, markers: null, markerMap: null, selectedMarker: null },
    smpl: { data: null, mesh: null, geom: null, material: null, positions: null, index: null, pick: null, mappedGroup: null },
    mapping: {},
    selectedSmpl: null,
    rawSceneData: null, // original scene JSON (unscaled body/marker positions)
  };

  // Body-to-parent mapping from scene edges, and body→segment mapping for scaling
  const BODY_SEGMENT = {
    "torso": "torso", "head": "torso",
    "humerus_r": "humerus", "humerus_l": "humerus",
    "ulna_r": "ulna", "ulna_l": "ulna",
    "radius_r": "ulna", "radius_l": "ulna",
    "hand_r": "hand", "hand_l": "hand",
    "femur_r": "femur", "femur_l": "femur",
    "tibia_r": "tibia", "tibia_l": "tibia",
    "talus_r": "foot", "talus_l": "foot",
    "calcn_r": "foot", "calcn_l": "foot",
    "toes_r": "foot", "toes_l": "foot",
  };

  // Which marker sites belong to which body (for rescaling)
  // SAM4Dcap 43-marker *_study naming convention
  const SITE_BODY = {
    // Pelvis
    "r.ASIS_study": "pelvis", "L.ASIS_study": "pelvis",
    "r.PSIS_study": "pelvis", "L.PSIS_study": "pelvis",
    // Knee (on femur in OpenSim)
    "r_knee_study": "femur_r", "r_mknee_study": "femur_r",
    "L_knee_study": "femur_l", "L_mknee_study": "femur_l",
    // Ankle (on tibia)
    "r_ankle_study": "tibia_r", "r_mankle_study": "tibia_r",
    "L_ankle_study": "tibia_l", "L_mankle_study": "tibia_l",
    // Foot (on calcn)
    "r_toe_study": "calcn_r", "r_5meta_study": "calcn_r", "r_calc_study": "calcn_r",
    "L_toe_study": "calcn_l", "L_5meta_study": "calcn_l", "L_calc_study": "calcn_l",
    // Shoulder & C7 (on torso)
    "r_shoulder_study": "torso", "L_shoulder_study": "torso",
    "C7_study": "torso",
    // Elbow (on humerus)
    "r_lelbow_study": "humerus_r", "r_melbow_study": "humerus_r",
    "L_lelbow_study": "humerus_l", "L_melbow_study": "humerus_l",
    // Wrist (on radius)
    "r_lwrist_study": "radius_r", "r_mwrist_study": "radius_r",
    "L_lwrist_study": "radius_l", "L_mwrist_study": "radius_l",
    // Thigh cluster (on femur)
    "r_thigh1_study": "femur_r", "r_thigh2_study": "femur_r", "r_thigh3_study": "femur_r",
    "L_thigh1_study": "femur_l", "L_thigh2_study": "femur_l", "L_thigh3_study": "femur_l",
    // Shank cluster (on tibia)
    "r_sh1_study": "tibia_r", "r_sh2_study": "tibia_r", "r_sh3_study": "tibia_r",
    "L_sh1_study": "tibia_l", "L_sh2_study": "tibia_l", "L_sh3_study": "tibia_l",
  };

  function getSegmentScales() {
    return {
      global: Number(elements.scaleGlobal?.value || 1),
      torso: Number(elements.scaleTorso?.value || 1),
      clavicle: Number(elements.scaleClavicle?.value || 1),
      humerus: Number(elements.scaleHumerus?.value || 1),
      ulna: Number(elements.scaleUlna?.value || 1),
      hand: Number(elements.scaleHand?.value || 1),
      femur: Number(elements.scaleFemur?.value || 1),
      tibia: Number(elements.scaleTibia?.value || 1),
      foot: Number(elements.scaleFoot?.value || 1),
    };
  }

  function buildParentMap(edges) {
    // edges: [[parent, child], ...] → child→parent
    const pm = {};
    for (const [p, c] of edges) pm[c] = p;
    return pm;
  }

  function rescaleScene() {
    if (!current.rawSceneData) return;
    const data = current.rawSceneData;
    const scales = getSegmentScales();
    const parentMap = buildParentMap(data.edges || []);

    // Compute adjusted body positions
    const rawBodies = data.bodies || {};
    const bodyOrder = ["pelvis"]; // BFS from root
    const visited = new Set(["pelvis"]);
    for (let i = 0; i < bodyOrder.length; i++) {
      for (const [p, c] of (data.edges || [])) {
        if (p === bodyOrder[i] && !visited.has(c)) { bodyOrder.push(c); visited.add(c); }
      }
    }

    const adjBodies = {};
    for (const name of bodyOrder) {
      const raw = rawBodies[name]?.translation;
      if (!raw) continue;
      const parent = parentMap[name];
      // Clavicle: torso→humerus offset uses clavicle scale, not humerus
      const isClavicle = parent === "torso" && (name === "humerus_r" || name === "humerus_l");
      const seg = BODY_SEGMENT[name];
      const segScale = isClavicle ? (scales.clavicle || 1) : (seg ? (scales[seg] || 1) : 1);

      if (!parent || !adjBodies[parent]) {
        adjBodies[name] = [raw[0] * scales.global, raw[1] * scales.global, raw[2] * scales.global];
      } else {
        const pp = adjBodies[parent];
        const rawP = rawBodies[parent]?.translation || [0,0,0];
        const offset = [raw[0] - rawP[0], raw[1] - rawP[1], raw[2] - rawP[2]];
        adjBodies[name] = [
          pp[0] + offset[0] * segScale * scales.global,
          pp[1] + offset[1] * segScale * scales.global,
          pp[2] + offset[2] * segScale * scales.global,
        ];
      }
    }

    // Update skeleton lines
    if (current.opencap.skeleton?.group) {
      const lineSegs = current.opencap.skeleton.group.children[0];
      if (lineSegs) {
        const pos = lineSegs.geometry.attributes.position;
        const edges = data.edges || [];
        for (let i = 0; i < edges.length; i++) {
          const [a, b] = edges[i];
          const pa = adjBodies[a] || [0,0,0];
          const pb = adjBodies[b] || [0,0,0];
          pos.setXYZ(i*2, pa[0], pa[1], pa[2]);
          pos.setXYZ(i*2+1, pb[0], pb[1], pb[2]);
        }
        pos.needsUpdate = true;
      }
    }

    // Update marker positions
    // Use osim_bodies (height-scaled OpenSim body positions) for offset computation
    // since markers are placed relative to OpenSim anatomy, not MHR skeleton joints
    const osimBodies = data.osim_bodies || {};
    if (current.opencap.markerMap) {
      for (const m of (data.markers || [])) {
        const entry = current.opencap.markerMap.get(m.name);
        if (!entry) continue;
        const bodyName = m.body || SITE_BODY[m.name];
        // Use pre-computed offset if available, else compute from osim_bodies
        const osimBody = bodyName && osimBodies[bodyName];
        const adjBody = bodyName && adjBodies[bodyName];
        if (adjBody && osimBody) {
          const siteOffset = m.offset || [
            m.position[0] - osimBody[0], m.position[1] - osimBody[1], m.position[2] - osimBody[2]
          ];
          const seg = BODY_SEGMENT[bodyName];
          const segScale = seg ? (scales[seg] || 1) : 1;
          entry.mesh.position.set(
            adjBody[0] + siteOffset[0] * segScale * scales.global,
            adjBody[1] + siteOffset[1] * segScale * scales.global,
            adjBody[2] + siteOffset[2] * segScale * scales.global,
          );
        } else {
          entry.mesh.position.set(
            m.position[0] * scales.global,
            m.position[1] * scales.global,
            m.position[2] * scales.global,
          );
        }
        entry.position.copy(entry.mesh.position);
      }
    }

    // Update slider labels
    for (const key of ["Global","Torso","Clavicle","Humerus","Ulna","Hand","Femur","Tibia","Foot"]) {
      const el = elements[`scale${key}Val`];
      const sl = elements[`scale${key}`];
      if (el && sl) el.textContent = Number(sl.value).toFixed(2);
    }
  }

  function updateSelectedMarkerUi() {
    elements.selectedMarker.textContent = current.opencap.selectedMarker || "(none)";
  }

  function updateSelectedSmplUi() {
    if (!current.selectedSmpl) { elements.selectedSmpl.textContent = "(none)"; return; }
    const s = current.selectedSmpl;
    elements.selectedSmpl.textContent = `v${s.vertex} f(${s.face.join(",")}) bary(${s.barycentric.map(x => x.toFixed(3)).join(",")}) @ ${fmtVec3(s.point)}`;
  }

  function updateLabelVisibility() {
    if (!current.opencap.markerMap) return;
    const showAll = Boolean(elements.showOpenCapLabels.checked);
    for (const [name, entry] of current.opencap.markerMap.entries()) {
      entry.label.visible = showAll || name === current.opencap.selectedMarker;
    }
  }

  function setSelectedMarker(name) {
    const next = name || null;
    if (current.opencap.selectedMarker === next) return;
    if (current.opencap.markerMap && current.opencap.selectedMarker) {
      const prev = current.opencap.markerMap.get(current.opencap.selectedMarker);
      if (prev) prev.material.emissive.setHex(0x000000);
    }
    current.opencap.selectedMarker = next;
    if (current.opencap.markerMap && next) {
      const entry = current.opencap.markerMap.get(next);
      if (entry) entry.material.emissive.setHex(0x66c2ff);
    }
    updateSelectedMarkerUi();
    updateLabelVisibility();
    renderMarkerList();
  }

  function clearAll() { current.mapping = {}; rebuildMappedPoints(); renderMarkerList(); }
  function clearCurrent() {
    const name = current.opencap.selectedMarker;
    if (!name) return;
    delete current.mapping[name];
    rebuildMappedPoints();
    renderMarkerList();
  }

  function markerNames() {
    if (!current.opencap.markerMap) return [];
    return [...current.opencap.markerMap.keys()].sort();
  }

  function renderMarkerList() {
    elements.markerList.replaceChildren();
    const names = markerNames();
    const allMapped = names.filter(n => current.mapping[n]);
    elements.markerCountTitle.textContent = `Markers (${allMapped.length}/${names.length} mapped)`;

    // Group markers by segment
    const groupOrder = ["Pelvis","Knee","Ankle","Foot","Shoulder","Spine",
      "Elbow","Wrist","Thigh Cluster","Shank Cluster",
      // Legacy groups (kept for backwards compatibility)
      "Waist","Thigh","Shank","Torso","Scapula","Neck","Head","Upper Arm","Forearm","Other"];
    const groups = new Map();
    for (const name of names) {
      const entry = current.opencap.markerMap?.get(name);
      const g = entry?.segGroup || "Other";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(name);
    }

    for (const gName of groupOrder) {
      const gNames = groups.get(gName);
      if (!gNames || !gNames.length) continue;

      const gMapped = gNames.filter(n => current.mapping[n]).length;
      const colorHex = GROUP_COLORS[gName] || 0xd6d6d6;
      const colorStr = `#${colorHex.toString(16).padStart(6, "0")}`;

      const header = document.createElement("div");
      header.className = "group-header";
      header.dataset.group = gName;
      header.innerHTML = `<span class="group-dot" style="background:${colorStr}"></span>`
        + `<span class="group-name">${gName}</span>`
        + `<span class="group-count">${gMapped}/${gNames.length}</span>`;
      header.addEventListener("click", () => {
        const container = header.nextElementSibling;
        if (container) container.style.display = container.style.display === "none" ? "" : "none";
        header.classList.toggle("collapsed");
      });
      elements.markerList.appendChild(header);

      const container = document.createElement("div");
      container.className = "group-items";

      for (const name of gNames) {
        const item = document.createElement("div");
        item.className = "marker-item";
        item.dataset.name = name;
        item.dataset.group = gName;

        const code = document.createElement("code");
        code.textContent = name;

        const map = document.createElement("span");
        const m = current.mapping[name];
        if (m) { map.className = "map"; map.textContent = `v${m.vertex}`; }
        else { map.className = "map missing"; map.textContent = "\u2014"; }

        if (name === current.opencap.selectedMarker) {
          item.style.outline = "1px solid rgba(102,194,255,0.55)";
          item.style.background = "rgba(102,194,255,0.08)";
        }

        item.appendChild(code);
        item.appendChild(map);
        item.addEventListener("click", () => setSelectedMarker(name));
        container.appendChild(item);
      }

      elements.markerList.appendChild(container);
    }
    applyMarkerFilter();
  }

  function applyMarkerFilter() {
    const q = elements.markerFilter.value.trim().toLowerCase();
    for (const item of elements.markerList.querySelectorAll(".marker-item")) {
      const name = (item.dataset.name || "").toLowerCase();
      const group = (item.dataset.group || "").toLowerCase();
      item.style.display = !q || name.includes(q) || group.includes(q) ? "" : "none";
    }
    // Hide group headers if all their items are hidden
    for (const header of elements.markerList.querySelectorAll(".group-header")) {
      const container = header.nextElementSibling;
      if (!container) continue;
      const visible = container.querySelectorAll('.marker-item:not([style*="display: none"])');
      header.style.display = !q || visible.length > 0 ? "" : "none";
    }
  }

  function mhrVertexPosition(i) {
    const p = current.smpl.positions;
    return new THREE.Vector3(p[i*3], p[i*3+1], p[i*3+2]);
  }

  function findFirstFaceContainingVertex(vertexId) {
    const idx = current.smpl.index;
    if (!idx) return null;
    const v = Number(vertexId);
    for (let i = 0; i+2 < idx.length; i += 3) {
      if (idx[i] === v || idx[i+1] === v || idx[i+2] === v) return [idx[i], idx[i+1], idx[i+2]];
    }
    return null;
  }

  function mhrBaryPointLocal(face, bary) {
    const a = mhrVertexPosition(face[0]);
    const b = mhrVertexPosition(face[1]);
    const c = mhrVertexPosition(face[2]);
    return a.multiplyScalar(bary[0]).add(b.multiplyScalar(bary[1])).add(c.multiplyScalar(bary[2]));
  }

  function rebuildMappedPoints() {
    if (!current.smpl.mappedGroup) return;
    current.smpl.mappedGroup.clear();
    current.smpl.mappedGroup.visible = Boolean(elements.showMappedPoints.checked);
    if (!current.smpl.positions) return;

    const sphere = new THREE.SphereGeometry(1, 10, 8);
    const r = 0.012;
    for (const [markerName, m] of Object.entries(current.mapping)) {
      if (!m?.face || !m?.barycentric) continue;
      const local = mhrBaryPointLocal(m.face, m.barycentric);
      const material = new THREE.MeshStandardMaterial({ color: 0xff3333, roughness: 0.55, metalness: 0.05 });
      const dot = new THREE.Mesh(sphere, material);
      dot.scale.setScalar(r);
      dot.position.copy(local);
      dot.name = `mapped:${markerName}`;
      current.smpl.mappedGroup.add(dot);
    }
  }

  function setMhrPick(pick) {
    current.selectedSmpl = pick;
    updateSelectedSmplUi();
    if (current.smpl.pick) {
      current.smpl.pick.visible = Boolean(pick);
      if (pick) current.smpl.pick.position.copy(pick.pointLocal);
    }
  }

  function nextUnmappedMarker(afterName) {
    const names = markerNames();
    if (!names.length) return null;
    const start = afterName ? Math.max(0, names.indexOf(afterName)) : 0;
    for (let i = start+1; i < names.length; i++) if (!current.mapping[names[i]]) return names[i];
    for (let i = 0; i <= start; i++) if (!current.mapping[names[i]]) return names[i];
    return null;
  }

  function getMirrorName(name) {
    // Use explicit lr_pairs from mesh data if available (SAM4Dcap naming)
    const pairs = current.smpl.data?.lr_pairs;
    if (pairs) {
      for (const [r, l] of pairs) {
        if (name === r) return l;
        if (name === l) return r;
      }
      return null;  // midline marker, no mirror
    }
    // Fallback: first-character swap (old convention)
    const names = markerNames();
    const first = name[0];
    let mirror;
    if (first === 'r') mirror = 'l' + name.slice(1);
    else if (first === 'l') mirror = 'r' + name.slice(1);
    else if (first === 'R') mirror = 'L' + name.slice(1);
    else if (first === 'L') mirror = 'R' + name.slice(1);
    else return null;
    return names.includes(mirror) ? mirror : null;
  }

  function findClosestMeshPoint(localPoint) {
    const pos = current.smpl.positions;
    if (!pos) return null;
    const nVerts = pos.length / 3;
    let bestDist = Infinity;
    let bestVert = 0;
    for (let i = 0; i < nVerts; i++) {
      const dx = pos[i * 3] - localPoint.x;
      const dy = pos[i * 3 + 1] - localPoint.y;
      const dz = pos[i * 3 + 2] - localPoint.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestDist) { bestDist = d2; bestVert = i; }
    }
    const face = findFirstFaceContainingVertex(bestVert);
    if (!face) return null;
    const pt = mhrVertexPosition(bestVert);
    const pa = mhrVertexPosition(face[0]);
    const pb = mhrVertexPosition(face[1]);
    const pc = mhrVertexPosition(face[2]);
    const bary = barycentricCoords(pa, pb, pc, pt);
    return { vertex: bestVert, face, barycentric: bary };
  }

  function autoSiteRemaining() {
    if (!current.opencap.markerMap || !current.smpl.mesh) return;
    const names = markerNames();
    let count = 0;
    for (const name of names) {
      if (current.mapping[name]) continue;
      const entry = current.opencap.markerMap.get(name);
      if (!entry) continue;
      // Transform marker position to MHR mesh local space
      const worldPos = entry.mesh.getWorldPosition(new THREE.Vector3());
      const localPos = current.smpl.mesh.worldToLocal(worldPos.clone());
      const pick = findClosestMeshPoint(localPos);
      if (pick) {
        current.mapping[name] = pick;
        count++;
      }
    }
    rebuildMappedPoints();
    renderMarkerList();
    setMeta(`Auto-sited ${count} markers to nearest mesh points`);
  }

  async function loadAtlasFromFile() {
    // Load auto-generated atlas JSON (marker_name → vertex_index)
    // Looks for mhr_marker_atlas.json in the conversion directory
    const atlasUrl = "../../../src/core/conversion/mhr_marker_atlas.json";
    try {
      const atlas = await loadJson(atlasUrl, "Atlas JSON");
      if (!current.smpl.positions) {
        setMeta("Load mesh first, then load atlas");
        return;
      }
      let count = 0;
      for (const [markerName, vertexId] of Object.entries(atlas)) {
        if (typeof vertexId !== "number") continue;
        const face = findFirstFaceContainingVertex(vertexId);
        if (!face) continue;
        const pt = mhrVertexPosition(vertexId);
        const pa = mhrVertexPosition(face[0]);
        const pb = mhrVertexPosition(face[1]);
        const pc = mhrVertexPosition(face[2]);
        const bary = barycentricCoords(pa, pb, pc, pt);
        current.mapping[markerName] = { vertex: vertexId, face, barycentric: bary };
        count++;
      }
      rebuildMappedPoints();
      renderMarkerList();
      setMeta(`Loaded atlas: ${count} markers from auto-generated atlas`);
    } catch (e) {
      setMeta(`Atlas load failed: ${e}. Run: uv run python scripts/build_mhr_atlas.py --auto --mesh data/output/Max/`);
    }
  }

  function assignMappingForSelectedMarker(pick) {
    const markerName = current.opencap.selectedMarker;
    if (!markerName) return;
    current.mapping[markerName] = { vertex: pick.vertex, face: pick.face, barycentric: pick.barycentric };

    // Auto-symmetry: mirror to L↔R counterpart
    if (elements.autoSymmetry?.checked) {
      const mirror = getMirrorName(markerName);
      if (mirror && !current.mapping[mirror]) {
        const localPt = mhrBaryPointLocal(pick.face, pick.barycentric);
        // Mirror across X=0 in mesh local space (X is L/R axis in MHR)
        const mirrorPt = new THREE.Vector3(-localPt.x, localPt.y, localPt.z);
        const mirrorPick = findClosestMeshPoint(mirrorPt);
        if (mirrorPick) {
          current.mapping[mirror] = mirrorPick;
        }
      }
    }

    rebuildMappedPoints();
    renderMarkerList();
    if (elements.autoAdvance.checked) {
      const next = nextUnmappedMarker(markerName);
      if (next) setSelectedMarker(next);
    }
  }

  function pickFromPointer(event) {
    const rect = viz.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return { marker: null, smpl: null };
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viz.camera);

    let markerName = null;
    if (current.opencap.markers?.group?.visible) {
      const hits = raycaster.intersectObjects(current.opencap.markers.group.children, true);
      if (hits.length) {
        let obj = hits[0].object;
        while (obj && obj.parent && !obj.userData?.markerName && !String(obj.name||"").startsWith("marker:")) obj = obj.parent;
        markerName = obj?.userData?.markerName || (String(obj?.name||"").startsWith("marker:") ? obj.name.slice("marker:".length) : null);
      }
    }

    let smplHit = null;
    if (!markerName && current.smpl.mesh && current.smpl.mesh.visible) {
      const hits = raycaster.intersectObject(current.smpl.mesh, true);
      if (hits.length) smplHit = hits[0];
    }
    return { marker: markerName, smpl: smplHit };
  }

  function buildPickFromMhrHit(hit) {
    if (!hit?.face || !current.smpl.index) return null;
    const { a, b, c } = hit.face;
    const face = [a, b, c];
    const localPoint = current.smpl.mesh.worldToLocal(hit.point.clone());
    const pa = mhrVertexPosition(a), pb = mhrVertexPosition(b), pc = mhrVertexPosition(c);
    const bary = barycentricCoords(pa, pb, pc, localPoint);
    let vertex = a;
    if (bary[1] >= bary[0] && bary[1] >= bary[2]) vertex = b;
    if (bary[2] >= bary[0] && bary[2] >= bary[1]) vertex = c;
    return { vertex, face, barycentric: bary, point: hit.point.clone(), pointLocal: localPoint };
  }

  function tick() {
    requestAnimationFrame(tick);
    viz.renderer.render(viz.scene, viz.camera);
    viz.labelRenderer.render(viz.scene, viz.camera);
  }
  requestAnimationFrame(tick);

  function clearAllScene() {
    viz.openCap.clear();
    viz.smpl.clear();
    elements.markerList.replaceChildren();
    current = {
      opencap: { data: null, skeleton: null, markers: null, markerMap: null, selectedMarker: null },
      smpl: { data: null, mesh: null, geom: null, material: null, positions: null, index: null, pick: null, mappedGroup: null },
      mapping: {},
      selectedSmpl: null,
      rawSceneData: null,
    };
    updateSelectedMarkerUi();
    updateSelectedSmplUi();
  }

  function setMhrYawPreset() {
    const deg = Number(elements.smplYawPreset.value || "0");
    viz.smpl.rotation.set(0, THREE.MathUtils.degToRad(deg), 0);
  }

  function getLockedScaleOrNull() {
    if (!elements.lockScaleToBsm?.checked) return null;
    const s = Number(current.smpl.data?.meta?.scale_to_bsm_c7_ground);
    return (Number.isFinite(s) && s > 0) ? s : null;
  }

  function resetMhrTransform() {
    viz.smpl.position.set(0, 0, 0);
    viz.smpl.rotation.set(0, 0, 0);
    const locked = getLockedScaleOrNull();
    if (locked) viz.smpl.scale.setScalar(locked);
    else viz.smpl.scale.set(1, 1, 1);
    setMhrYawPreset();
  }

  function roughAlignBBox() {
    if (!current.opencap.markerMap || !current.smpl.positions) return;
    const openPoints = [...current.opencap.markerMap.values()].map(e => e.position.clone());
    const boxO = computeBounds(openPoints);

    const boxS = new THREE.Box3();
    const pos = current.smpl.positions;
    for (let i = 0; i < pos.length; i += 3) boxS.expandByPoint(new THREE.Vector3(pos[i], pos[i+1], pos[i+2]));

    const locked = getLockedScaleOrNull();
    const hO = boxO.max.y - boxO.min.y;
    const hS = boxS.max.y - boxS.min.y;
    const scale = locked || (hS > 1e-6 ? hO / hS : 1);

    const centerO = boxO.getCenter(new THREE.Vector3());
    const centerS = boxS.getCenter(new THREE.Vector3());
    viz.smpl.scale.setScalar(scale);
    viz.smpl.position.copy(centerO.clone().sub(centerS.multiplyScalar(scale)));
  }

  function fitAlignFromMapping() {
    if (!current.opencap.markerMap || !current.smpl.positions) throw new Error("Data not loaded yet");
    const src = [], dst = [];
    for (const [markerName, m] of Object.entries(current.mapping)) {
      const marker = current.opencap.markerMap.get(markerName);
      if (!marker || !m?.face || !m?.barycentric) continue;
      src.push(mhrBaryPointLocal(m.face, m.barycentric));
      dst.push(marker.position.clone());
    }
    if (src.length < 3) throw new Error("Alignment requires at least 3 valid mappings");

    const { scale, R, t } = fitSimilarityTransform(src, dst);
    const m4 = new THREE.Matrix4();
    m4.set(R[0][0],R[0][1],R[0][2],0, R[1][0],R[1][1],R[1][2],0, R[2][0],R[2][1],R[2][2],0, 0,0,0,1);
    viz.smpl.quaternion.setFromRotationMatrix(m4);
    viz.smpl.scale.setScalar(scale);
    viz.smpl.position.copy(t);
  }

  function fitCamera() {
    const points = [];
    if (current.opencap.markerMap) for (const e of current.opencap.markerMap.values()) points.push(e.mesh.getWorldPosition(new THREE.Vector3()));
    if (current.smpl.positions && current.smpl.mesh?.visible) {
      const b = current.smpl.mesh.geometry.boundingBox || new THREE.Box3().setFromObject(current.smpl.mesh);
      points.push(b.min.clone()); points.push(b.max.clone());
    }
    if (points.length) fitCameraToBox(viz.camera, viz.controls, computeBounds(points));
  }

  function applyLayout() {
    const mode = String(elements.layoutMode.value || "overlay");
    const d = Number(elements.splitDistance.value || "1.2");
    if (mode === "split") {
      viz.openCapLayout.position.set(-d/2, 0, 0);
      viz.smplLayout.position.set(d/2, 0, 0);
    } else {
      viz.openCapLayout.position.set(0, 0, 0);
      viz.smplLayout.position.set(0, 0, 0);
    }
  }

  function buildExportObject() {
    const smplRot = new THREE.Euler().setFromQuaternion(viz.smpl.quaternion, "XYZ");
    return {
      meta: {
        format: "opensim_to_mhr_manual_v1",
        created_at: new Date().toISOString(),
        opensim_scene: elements.opencapScenePath.value.trim(),
        mhr_mesh: elements.smplMeshPath.value.trim(),
      },
      mhr_transform: {
        position: [viz.smpl.position.x, viz.smpl.position.y, viz.smpl.position.z],
        rotation_xyz_rad: [smplRot.x, smplRot.y, smplRot.z],
        scale: [viz.smpl.scale.x, viz.smpl.scale.y, viz.smpl.scale.z],
      },
      mapping: current.mapping,
    };
  }

  function exportJson() {
    downloadText("mhr_marker_mapping.json", JSON.stringify(buildExportObject(), null, 2), "application/json");
  }

  function exportCsv() {
    const names = markerNames();
    const lines = ["marker_name,vertex,face_a,face_b,face_c,bary_u,bary_v,bary_w"];
    for (const name of names) {
      const m = current.mapping[name];
      if (!m) { lines.push(`${name},,,,,,,`); continue; }
      const face = m.face || ["","",""];
      const b = m.barycentric || ["","",""];
      lines.push(`${name},${m.vertex},${face[0]},${face[1]},${face[2]},${b[0]},${b[1]},${b[2]}`);
    }
    downloadText("mhr_marker_mapping.csv", lines.join("\n"), "text/csv");
  }

  function importJsonFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const obj = JSON.parse(String(reader.result || ""));
        current.mapping = obj?.mapping || {};
        const t = obj?.mhr_transform || obj?.smpl_transform;
        if (t?.position?.length === 3) viz.smpl.position.set(t.position[0], t.position[1], t.position[2]);
        if (t?.rotation_xyz_rad?.length === 3) viz.smpl.rotation.set(t.rotation_xyz_rad[0], t.rotation_xyz_rad[1], t.rotation_xyz_rad[2], "XYZ");
        if (t?.scale?.length === 3) viz.smpl.scale.set(t.scale[0], t.scale[1], t.scale[2]);
        rebuildMappedPoints();
        renderMarkerList();
        setMeta(`Imported mapping: ${Object.keys(current.mapping).length} markers`);
      } catch (e) { setMeta(`Import failed: ${e}`); }
    };
    reader.readAsText(file);
  }

  async function loadAll() {
    const ocUrl = elements.opencapScenePath.value.trim();
    const meshUrl = elements.smplMeshPath.value.trim();
    if (!ocUrl || !meshUrl) { setMeta("Enter both scene JSON and mesh JSON paths, then click Load"); return; }

    setMeta("Loading...");
    clearAllScene();

    const [ocData, meshData] = await Promise.all([
      loadJson(ocUrl, "OpenSim scene JSON"),
      loadJson(meshUrl, "MHR mesh JSON"),
    ]);

    current.opencap.data = ocData;
    current.smpl.data = meshData;
    current.rawSceneData = ocData;

    // Scale lock
    const scaleToBsm = Number(meshData?.meta?.scale_to_bsm_c7_ground);
    if (Number.isFinite(scaleToBsm) && scaleToBsm > 0) {
      elements.lockScaleToBsm.disabled = false;
    } else {
      elements.lockScaleToBsm.checked = false;
      elements.lockScaleToBsm.disabled = true;
    }

    // OpenSim bodies + skeleton
    const bodies = new Map();
    for (const [name, entry] of Object.entries(ocData.bodies || {})) {
      if (!entry?.translation) continue;
      bodies.set(name, new THREE.Vector3(entry.translation[0], entry.translation[1], entry.translation[2]));
    }
    current.opencap.skeleton = buildOpenCapSkeleton(viz.openCap, bodies, ocData.edges || []);
    current.opencap.skeleton.group.visible = Boolean(elements.showOpenCapSkeleton.checked);

    // OpenSim markers
    const markerRadius = Number(elements.markerSize.value || "0.02");
    current.opencap.markers = buildOpenCapMarkers(viz.openCap, ocData.markers || [], markerRadius, elements.showOpenCapLabels.checked);
    current.opencap.markerMap = current.opencap.markers.markerMap;
    current.opencap.markers.group.visible = Boolean(elements.showOpenCapMarkers.checked);
    updateLabelVisibility();

    // MHR mesh
    resetMhrTransform();
    const mhr = buildMhrMesh(viz.smpl, meshData);
    current.smpl.mesh = mhr.mesh;
    current.smpl.geom = mhr.geometry;
    current.smpl.material = mhr.material;
    current.smpl.positions = mhr.positions;
    current.smpl.index = mhr.index;
    current.smpl.mesh.visible = Boolean(elements.showSmplMesh.checked);
    current.smpl.material.wireframe = Boolean(elements.showSmplWireframe.checked);

    // MHR pick marker + mapped dots
    const pickSphere = new THREE.Mesh(
      new THREE.SphereGeometry(1, 12, 10),
      new THREE.MeshStandardMaterial({ color: 0xffbf66 })
    );
    pickSphere.scale.setScalar(0.016);
    pickSphere.visible = false;
    pickSphere.name = "mhrPick";
    viz.smpl.add(pickSphere);
    current.smpl.pick = pickSphere;

    const mappedGroup = new THREE.Group();
    mappedGroup.name = "mhrMapped";
    viz.smpl.add(mappedGroup);
    current.smpl.mappedGroup = mappedGroup;
    rebuildMappedPoints();

    viz.grid.visible = Boolean(elements.showGrid.checked);
    viz.axes.visible = Boolean(elements.showAxes.checked);
    applyLayout();

    // Meta
    const nMarkers = (ocData.markers || []).length;
    const nVerts = (meshData.vertices || []).length / 3;
    const nFaces = (meshData.faces || meshData.triangles || []).length / 3;
    setMeta(`OpenSim: ${nMarkers} markers, ${bodies.size} bodies, ${(ocData.edges||[]).length} edges\nMHR: ${nVerts} vertices, ${nFaces} faces`);

    rescaleScene();
    renderMarkerList();
    fitCamera();
  }

  // Segment scale sliders
  for (const key of ["Global","Torso","Clavicle","Humerus","Ulna","Hand","Femur","Tibia","Foot"]) {
    const sl = elements[`scale${key}`];
    if (sl) sl.addEventListener("input", rescaleScene);
  }

  // Event handlers
  viz.renderer.domElement.addEventListener("pointerdown", (event) => {
    clickState.down = { x: event.clientX, y: event.clientY, id: event.pointerId };
  }, { capture: true });

  viz.renderer.domElement.addEventListener("pointerup", (event) => {
    const down = clickState.down;
    clickState.down = null;
    if (!down || down.id !== event.pointerId) return;
    if ((event.clientX-down.x)**2 + (event.clientY-down.y)**2 > 25) return;

    const picked = pickFromPointer(event);
    if (picked.marker) { setSelectedMarker(picked.marker); return; }
    if (picked.smpl) {
      const pick = buildPickFromMhrHit(picked.smpl);
      if (!pick) return;
      setMhrPick(pick);
      assignMappingForSelectedMarker(pick);
    }
  }, { capture: true });

  elements.load.addEventListener("click", () => loadAll().catch(e => setMeta(String(e))));
  elements.exportJson.addEventListener("click", exportJson);
  elements.exportCsv.addEventListener("click", exportCsv);
  elements.importJson.addEventListener("change", () => {
    const f = elements.importJson.files?.[0];
    if (f) importJsonFile(f);
    elements.importJson.value = "";
  });

  elements.markerFilter.addEventListener("input", applyMarkerFilter);
  elements.clearCurrent.addEventListener("click", clearCurrent);
  elements.clearAll.addEventListener("click", clearAll);
  elements.autoSite.addEventListener("click", autoSiteRemaining);
  elements.loadAtlas?.addEventListener("click", () => loadAtlasFromFile().catch(e => setMeta(String(e))));

  elements.layoutMode.addEventListener("change", () => { applyLayout(); fitCamera(); });
  elements.splitDistance.addEventListener("input", applyLayout);
  elements.smplYawPreset.addEventListener("change", setMhrYawPreset);
  elements.resetSmpl.addEventListener("click", () => { resetMhrTransform(); rebuildMappedPoints(); });
  elements.bboxAlign.addEventListener("click", () => { try { setMhrYawPreset(); roughAlignBBox(); fitCamera(); } catch(e) { setMeta(String(e)); } });
  elements.fitAlign.addEventListener("click", () => { try { fitAlignFromMapping(); fitCamera(); } catch(e) { setMeta(String(e)); } });
  elements.fitCamera.addEventListener("click", fitCamera);

  elements.showOpenCapSkeleton.addEventListener("change", () => { if (current.opencap.skeleton?.group) current.opencap.skeleton.group.visible = Boolean(elements.showOpenCapSkeleton.checked); });
  elements.showOpenCapMarkers.addEventListener("change", () => { if (current.opencap.markers?.group) current.opencap.markers.group.visible = Boolean(elements.showOpenCapMarkers.checked); });
  elements.showOpenCapLabels.addEventListener("change", updateLabelVisibility);
  elements.showSmplMesh.addEventListener("change", () => { if (current.smpl.mesh) current.smpl.mesh.visible = Boolean(elements.showSmplMesh.checked); });
  elements.showSmplWireframe.addEventListener("change", () => { if (current.smpl.material) current.smpl.material.wireframe = Boolean(elements.showSmplWireframe.checked); });
  elements.showMappedPoints.addEventListener("change", () => { if (current.smpl.mappedGroup) current.smpl.mappedGroup.visible = Boolean(elements.showMappedPoints.checked); });
  elements.showGrid.addEventListener("change", () => { viz.grid.visible = Boolean(elements.showGrid.checked); });
  elements.showAxes.addEventListener("change", () => { viz.axes.visible = Boolean(elements.showAxes.checked); });
  elements.markerSize.addEventListener("input", () => {
    if (!current.opencap.markerMap) return;
    const r = Number(elements.markerSize.value || "0.02");
    for (const entry of current.opencap.markerMap.values()) {
      entry.mesh.scale.setScalar(r);
      entry.label.position.set(0, r*1.2, 0);
    }
  });
  elements.lockScaleToBsm?.addEventListener("change", () => {
    const locked = getLockedScaleOrNull();
    if (locked) viz.smpl.scale.setScalar(locked);
  });

  applyLayout();
}

start();
