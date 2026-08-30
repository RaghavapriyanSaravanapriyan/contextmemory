import * as THREE from "three";
import type { GraphData, GraphNode, Hit } from "../lib/types";

export interface GoldenSignalPulse {
  edgeId: string;
  curve: THREE.QuadraticBezierCurve3;
  progress: number;
  speed: number;
  headMesh: THREE.Mesh;
  tailMesh: THREE.Line;
  light?: THREE.PointLight;
}

export class BrainGraph {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private animationFrameId: number | null = null;

  // Node & Edge objects
  private nodeMeshes: Map<string, THREE.Group> = new Map();
  private edgeLines: Map<string, THREE.Line> = new Map();
  private edgeCurves: Map<string, THREE.QuadraticBezierCurve3> = new Map();
  private nodePositions: Map<string, THREE.Vector3> = new Map();
  private signalPulses: GoldenSignalPulse[] = [];

  // Particle Atmosphere & Golden Dust
  private dustParticles!: THREE.Points;
  private ambientPulseTimer = 0;

  // Raycasting & Interaction
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2(-999, -999);
  private hoveredNodeId: string | null = null;

  // Camera Orbit & Transition
  private isDragging = false;
  private previousMousePosition = { x: 0, y: 0 };
  private cameraTargetPos = new THREE.Vector3(0, 0, 180);
  private cameraLookAt = new THREE.Vector3(0, 0, 0);

  // Callbacks
  public onNodeSelect?: (node: GraphNode | null) => void;
  public onNodeHover?: (nodeId: string | null) => void;

  private currentData: GraphData | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // 1. Scene & Renderer setup
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x020204, 0.003);

    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 2000);
    this.camera.position.copy(this.cameraTargetPos);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;
    this.container.appendChild(this.renderer.domElement);

    // 2. Ambient & Gold Accent Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const goldPointLight = new THREE.PointLight(0xffd700, 2.5, 400);
    goldPointLight.position.set(40, 60, 90);
    this.scene.add(goldPointLight);

    // 3. Add Ambient Dust Cloud
    this.createAtmosphere();

    // 4. Register Event Listeners
    this.attachEvents();

    // 5. Start Render Loop
    this.animate();
  }

  private createAtmosphere(): void {
    const particleCount = 750;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const goldColor = new THREE.Color(0xffd700);
    const whiteColor = new THREE.Color(0xffffff);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 450;
      positions[i + 1] = (Math.random() - 0.5) * 450;
      positions[i + 2] = (Math.random() - 0.5) * 450;

      const isGold = Math.random() > 0.6;
      const c = isGold ? goldColor : whiteColor;
      colors[i] = c.r;
      colors[i + 1] = c.g;
      colors[i + 2] = c.b;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 1.4,
      transparent: true,
      opacity: 0.35,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
    });

    this.dustParticles = new THREE.Points(geometry, material);
    this.scene.add(this.dustParticles);
  }

  public setData(data: GraphData): void {
    this.currentData = data;
    this.computePositions(data.nodes);
    this.renderGraph(data);
  }

  private computePositions(nodes: GraphNode[]): void {
    const count = nodes.length;
    const radius = Math.max(40, count * 7);

    nodes.forEach((node, i) => {
      if (!this.nodePositions.has(node.id)) {
        const phi = Math.acos(-1 + (2 * i) / count);
        const theta = Math.sqrt(count * Math.PI) * phi;
        const r = radius * (0.6 + 0.4 * Math.random());

        const x = r * Math.cos(theta) * Math.sin(phi);
        const y = r * Math.sin(theta) * Math.sin(phi);
        const z = r * Math.cos(phi);

        this.nodePositions.set(node.id, new THREE.Vector3(x, y, z));
      }
    });
  }

  private renderGraph(data: GraphData): void {
    this.nodeMeshes.forEach((mesh) => this.scene.remove(mesh));
    this.nodeMeshes.clear();
    this.edgeLines.forEach((line) => this.scene.remove(line));
    this.edgeLines.clear();
    this.edgeCurves.clear();

    // Render Nodes
    data.nodes.forEach((node) => {
      const pos = this.nodePositions.get(node.id) || new THREE.Vector3();
      const nodeGroup = this.createNodeGroup(node);
      nodeGroup.position.copy(pos);
      this.scene.add(nodeGroup);
      this.nodeMeshes.set(node.id, nodeGroup);
    });

    // Render Curved Edges
    data.edges.forEach((edge) => {
      const fromPos = this.nodePositions.get(edge.from);
      const toPos = this.nodePositions.get(edge.to);
      if (fromPos && toPos) {
        const midPoint = new THREE.Vector3()
          .addVectors(fromPos, toPos)
          .multiplyScalar(0.5);
        midPoint.add(
          new THREE.Vector3(
            (Math.random() - 0.5) * 12,
            (Math.random() - 0.5) * 12,
            (Math.random() - 0.5) * 12
          )
        );

        const curve = new THREE.QuadraticBezierCurve3(fromPos, midPoint, toPos);
        const points = curve.getPoints(32);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const material = new THREE.LineBasicMaterial({
          color: 0x444444,
          transparent: true,
          opacity: edge.derived ? 0.12 : 0.25,
          linewidth: 1,
        });

        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        this.edgeLines.set(edge.id, line);
        this.edgeCurves.set(edge.id, curve);
      }
    });
  }

  private createNodeGroup(node: GraphNode): THREE.Group {
    const group = new THREE.Group();
    const radius = 2.5 + (node.salience || 0.5) * 3.5;

    // Golden Core Sphere
    const coreGeo = new THREE.SphereGeometry(radius, 24, 24);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.15,
      metalness: 0.85,
      emissive: 0x111115,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    coreMesh.userData = { nodeId: node.id };
    group.add(coreMesh);

    // Subtle Outer Ring
    const ringGeo = new THREE.RingGeometry(radius * 1.3, radius * 1.6, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffd700,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI / 4;
    group.add(ringMesh);

    // Text Label Sprite
    const labelSprite = this.createTextSprite(node.subject || node.kind);
    labelSprite.position.set(0, radius + 3.8, 0);
    group.add(labelSprite);

    return group;
  }

  private createTextSprite(text: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.font = "bold 24px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
      ctx.textAlign = "center";
      ctx.shadowColor = "rgba(255, 215, 0, 0.6)";
      ctx.shadowBlur = 10;
      ctx.fillText(text, 128, 40);
    }
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(16, 4, 1);
    return sprite;
  }

  /** Spawn rapid golden electrical stroke pulse along a bezier curve */
  private spawnGoldenStroke(edgeId: string, curve: THREE.QuadraticBezierCurve3, fast = true): void {
    // 1. Glowing Gold Head Sphere
    const headGeo = new THREE.SphereGeometry(1.8, 16, 16);
    const headMat = new THREE.MeshBasicMaterial({
      color: 0xfff0aa,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
    });
    const headMesh = new THREE.Mesh(headGeo, headMat);

    // 2. Rapid Golden Stroke Tail Line
    const tailPoints = [curve.getPoint(0), curve.getPoint(0.05)];
    const tailGeo = new THREE.BufferGeometry().setFromPoints(tailPoints);
    const tailMat = new THREE.LineBasicMaterial({
      color: 0xffd700,
      transparent: true,
      opacity: 0.9,
      linewidth: 3,
      blending: THREE.AdditiveBlending,
    });
    const tailMesh = new THREE.Line(tailGeo, tailMat);

    // 3. Dynamic Golden PointLight
    const strokeLight = new THREE.PointLight(0xffd700, 2.0, 35);

    this.scene.add(headMesh);
    this.scene.add(tailMesh);
    this.scene.add(strokeLight);

    this.signalPulses.push({
      edgeId,
      curve,
      progress: 0,
      speed: fast ? 0.045 + Math.random() * 0.025 : 0.025,
      headMesh,
      tailMesh,
      light: strokeLight,
    });
  }

  /** Trigger Real-Time Synaptic Golden Stroke Flow across evidence nodes */
  public triggerSynapticFlow(evidence: Hit[]): void {
    if (!evidence || evidence.length === 0) return;

    evidence.forEach((hit) => {
      this.currentData?.edges.forEach((edge) => {
        if (edge.from === hit.id || edge.to === hit.id) {
          const curve = this.edgeCurves.get(edge.id);
          if (curve) {
            // Launch 3 rapid golden strokes with slight staggering
            this.spawnGoldenStroke(edge.id, curve, true);
            setTimeout(() => this.spawnGoldenStroke(edge.id, curve, true), 80);
            setTimeout(() => this.spawnGoldenStroke(edge.id, curve, true), 160);
          }
        }
      });
    });
  }

  public focusNode(nodeId: string): void {
    const pos = this.nodePositions.get(nodeId);
    if (pos) {
      this.cameraTargetPos.set(pos.x, pos.y, pos.z + 60);
      this.cameraLookAt.copy(pos);
    }
  }

  public resetCamera(): void {
    this.cameraTargetPos.set(0, 0, 180);
    this.cameraLookAt.set(0, 0, 0);
  }

  private animate = (): void => {
    this.animationFrameId = requestAnimationFrame(this.animate);

    // Rotate ambient dust cloud slowly
    if (this.dustParticles) {
      this.dustParticles.rotation.y += 0.0004;
      this.dustParticles.rotation.x += 0.0002;
    }

    // Continuous real-time ambient micro-pulses
    this.ambientPulseTimer++;
    if (this.ambientPulseTimer > 45 && this.edgeCurves.size > 0) {
      this.ambientPulseTimer = 0;
      const keys = Array.from(this.edgeCurves.keys());
      const randomKey = keys[Math.floor(Math.random() * keys.length)];
      const curve = this.edgeCurves.get(randomKey);
      if (curve) {
        this.spawnGoldenStroke(randomKey, curve, false);
      }
    }

    // Update real-time golden stroke pulses along bezier curves
    for (let i = this.signalPulses.length - 1; i >= 0; i--) {
      const pulse = this.signalPulses[i];
      pulse.progress += pulse.speed;

      if (pulse.progress >= 1) {
        this.scene.remove(pulse.headMesh);
        this.scene.remove(pulse.tailMesh);
        if (pulse.light) this.scene.remove(pulse.light);
        this.signalPulses.splice(i, 1);
      } else {
        const headPos = pulse.curve.getPoint(pulse.progress);
        const tailProgress = Math.max(0, pulse.progress - 0.15);
        const tailPos = pulse.curve.getPoint(tailProgress);

        pulse.headMesh.position.copy(headPos);
        if (pulse.light) pulse.light.position.copy(headPos);

        // Update stroke trail line geometry
        const trailPoints = [tailPos, headPos];
        pulse.tailMesh.geometry.dispose();
        pulse.tailMesh.geometry = new THREE.BufferGeometry().setFromPoints(trailPoints);
      }
    }

    // Smooth Camera Interpolation
    this.camera.position.lerp(this.cameraTargetPos, 0.05);
    this.camera.lookAt(this.cameraLookAt);

    this.renderer.render(this.scene, this.camera);
  };

  private attachEvents(): void {
    const el = this.container;

    el.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    el.addEventListener("mousemove", (e) => {
      const rect = el.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (this.isDragging) {
        const deltaX = e.clientX - this.previousMousePosition.x;
        const deltaY = e.clientY - this.previousMousePosition.y;

        this.cameraTargetPos.x -= deltaX * 0.2;
        this.cameraTargetPos.y += deltaY * 0.2;

        this.previousMousePosition = { x: e.clientX, y: e.clientY };
      }

      this.checkHover();
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    el.addEventListener("click", () => {
      this.checkClick();
    });

    el.addEventListener("wheel", (e) => {
      this.cameraTargetPos.z += e.deltaY * 0.1;
      this.cameraTargetPos.z = Math.max(30, Math.min(400, this.cameraTargetPos.z));
    });

    window.addEventListener("resize", this.onWindowResize);
  }

  private checkHover(): void {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const meshes: THREE.Object3D[] = [];
    this.nodeMeshes.forEach((group) => {
      group.traverse((child) => {
        if (child.userData?.nodeId) meshes.push(child);
      });
    });

    const intersects = this.raycaster.intersectObjects(meshes);
    if (intersects.length > 0) {
      const hoveredId = intersects[0].object.userData.nodeId;
      if (this.hoveredNodeId !== hoveredId) {
        this.hoveredNodeId = hoveredId;
        this.onNodeHover?.(hoveredId);
      }
    } else if (this.hoveredNodeId !== null) {
      this.hoveredNodeId = null;
      this.onNodeHover?.(null);
    }
  }

  private checkClick(): void {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const meshes: THREE.Object3D[] = [];
    this.nodeMeshes.forEach((group) => {
      group.traverse((child) => {
        if (child.userData?.nodeId) meshes.push(child);
      });
    });

    const intersects = this.raycaster.intersectObjects(meshes);
    if (intersects.length > 0) {
      const clickedId = intersects[0].object.userData.nodeId;
      this.focusNode(clickedId);
      const node = this.currentData?.nodes.find((n) => n.id === clickedId) || null;
      this.onNodeSelect?.(node);
    } else {
      this.onNodeSelect?.(null);
    }
  }

  private onWindowResize = (): void => {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  public destroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    window.removeEventListener("resize", this.onWindowResize);
    this.renderer.dispose();
    if (this.container.contains(this.renderer.domElement)) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
