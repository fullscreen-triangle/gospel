// 3D viewer for the CTCF zinc-finger / DNA complex (PDB 5T0U).
// Loaded as a GLB asset via @react-three/drei's useGLTF; rendered in a
// Canvas with a hemisphere light, soft directional rim, and an autorotating
// OrbitControls so the structure is always presented in motion.

import { Suspense, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Environment } from "@react-three/drei";

const MODEL_PATH = "/model/ctcf_a_guiding_hand_5t0u.glb";

// Preload so the canvas doesn't flash while fetching.
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

export default function ProteinModel({ className = "" }) {
  return (
    <div className={`relative w-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 18], fov: 35 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <hemisphereLight intensity={0.9} groundColor="#222" />
        <directionalLight position={[6, 8, 6]} intensity={1.0} />
        <directionalLight position={[-6, -4, -6]} intensity={0.4} />
        <Suspense fallback={null}>
          <Environment preset="city" />
          <ModelMesh scale={0.32} position={[0, 0.5, 0]} />
        </Suspense>
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          autoRotate={false}
          rotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}

