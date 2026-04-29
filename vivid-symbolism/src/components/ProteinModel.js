// 3D viewer for the CTCF zinc-finger / DNA complex (PDB 5T0U).
//
// Loaded as a GLB asset via @react-three/drei's useGLTF; rendered in a
// Canvas with a hemisphere light, soft directional rim, and an
// OrbitControls so the user can spin the structure manually.
//
// Mounted defensively: a WebGL availability check runs first, and an
// error boundary wraps the Canvas so any context-creation failure
// (remote desktop, hardware acceleration off, exhausted GPU contexts,
// strict browser policies) falls back to rendering nothing rather than
// crashing the page.

import React, { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment } from "@react-three/drei";

import { hasWebGL } from "@/lib/webgl";

const MODEL_PATH = "/model/ctcf_a_guiding_hand_5t0u.glb";

useGLTF.preload(MODEL_PATH);

function ModelMesh({ scale = 1, position = [0, 0, 0] }) {
  const ref = useRef();
  const { scene } = useGLTF(MODEL_PATH);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.18;
  });
  return (
    <group ref={ref} scale={scale} position={position}>
      <primitive object={scene} />
    </group>
  );
}

class CanvasErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error, info) {
    if (typeof console !== "undefined") {
      // eslint-disable-next-line no-console
      console.warn("ProteinModel canvas failed:", error?.message ?? error);
    }
  }
  render() {
    if (this.state.hasError) return null;
    return this.props.children;
  }
}

export default function ProteinModel({ className = "" }) {
  // useState(null) for "haven't checked yet" so we don't render the canvas
  // during SSR or the first paint, which avoids a no-op throw.
  const [supported, setSupported] = useState(null);

  useEffect(() => {
    setSupported(hasWebGL());
  }, []);

  if (!supported) {
    // No WebGL: render a transparent placeholder so the layout stays the
    // same and the foreground text is unaffected.
    return <div className={`pointer-events-none w-full ${className}`} />;
  }

  return (
    <div className={`relative w-full ${className}`}>
      <CanvasErrorBoundary>
        <Canvas
          camera={{ position: [0, 0, 18], fov: 35 }}
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true, powerPreference: "default", failIfMajorPerformanceCaveat: false }}
          onCreated={({ gl }) => {
            gl.domElement.addEventListener("webglcontextlost", (e) => {
              e.preventDefault();
            });
          }}
          style={{ background: "transparent" }}
        >
          <hemisphereLight intensity={0.9} groundColor="#222" />
          <directionalLight position={[6, 8, 6]} intensity={1.0} />
          <directionalLight position={[-6, -4, -6]} intensity={0.4} />
          <Suspense fallback={null}>
            <Environment preset="city" />
            <ModelMesh scale={0.08} position={[0, 3, 0]} />
          </Suspense>
          <OrbitControls
            enablePan={false}
            enableZoom={false}
            autoRotate={false}
            rotateSpeed={0.5}
          />
        </Canvas>
      </CanvasErrorBoundary>
    </div>
  );
}
