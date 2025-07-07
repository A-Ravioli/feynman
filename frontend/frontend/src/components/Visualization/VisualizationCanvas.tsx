import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, Stats, Environment, PerspectiveCamera } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';

import { SimulationData, SimulationState } from '../../types';

interface VisualizationCanvasProps {
  simulationData: SimulationData | null;
  simulationState: SimulationState;
  onTimeChange: (time: number, step: number) => void;
}

interface ParticleProps {
  position: [number, number, number];
  color: string;
  size: number;
  velocity?: [number, number, number];
  trail?: [number, number, number][];
}

const Particle: React.FC<ParticleProps> = ({ position, color, size, velocity, trail }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      // Add subtle pulsing animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <group>
      {/* Main particle */}
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial 
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          transparent
          opacity={0.8}
        />
      </mesh>
      
      {/* Glow effect */}
      <mesh position={position}>
        <sphereGeometry args={[size * 1.5, 16, 16]} />
        <meshBasicMaterial 
          color={color}
          transparent
          opacity={0.2}
        />
      </mesh>
      
      {/* Trail */}
      {trail && trail.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              array={new Float32Array(trail.flat())}
              count={trail.length}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={color} opacity={0.5} transparent />
        </line>
      )}
      
      {/* Velocity vector */}
      {velocity && (
        <arrowHelper
          args={[
            new THREE.Vector3(velocity[0], velocity[1], velocity[2]).normalize(),
            new THREE.Vector3(position[0], position[1], position[2]),
            new THREE.Vector3(velocity[0], velocity[1], velocity[2]).length() * 0.1,
            color,
            size * 0.5,
            size * 0.3
          ]}
        />
      )}
    </group>
  );
};

const Scene: React.FC<{ simulationData: SimulationData; currentStep: number }> = ({ 
  simulationData, 
  currentStep 
}) => {
  const particles: ParticleProps[] = [];
  
  // Process entities for visualization
  Object.entries(simulationData.entities).forEach(([name, entity]) => {
    if (entity.time_series.type === 'object') {
      const positions = entity.time_series.positions;
      const velocities = entity.time_series.velocities;
      
      if (positions && positions[currentStep]) {
        const position = positions[currentStep];
        const velocity = velocities?.[currentStep];
        const color = entity.initial_properties.color || '#3b82f6';
        const size = entity.initial_properties.size || 0.5;
        
        particles.push({
          position: [position[0], position[1], position[2]],
          color,
          size,
          velocity: velocity ? [velocity[0], velocity[1], velocity[2]] : undefined,
        });
      }
    } else if (entity.time_series.type === 'atom') {
      // Quantum particle visualization
      const expectedPos = entity.time_series.expected_position;
      
      if (expectedPos && expectedPos[currentStep]) {
        const position = expectedPos[currentStep];
        const color = '#8b5cf6'; // Purple for quantum
        const size = 0.3;
        
        particles.push({
          position: [position[0], position[1], position[2]],
          color,
          size,
        });
      }
    }
  });

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <pointLight position={[-10, -10, -5]} intensity={0.5} />
      
      {/* Environment */}
      <Environment preset="night" />
      
      {/* Grid */}
      <Grid 
        args={[20, 20]} 
        position={[0, -5, 0]}
        cellColor="#374151"
        sectionColor="#4b5563"
      />
      
      {/* Particles */}
      {particles.map((particle, index) => (
        <Particle key={index} {...particle} />
      ))}
      
      {/* Coordinate axes */}
      <axesHelper args={[2]} />
    </>
  );
};

const VisualizationCanvas: React.FC<VisualizationCanvasProps> = ({ 
  simulationData, 
  simulationState, 
  onTimeChange 
}) => {
  const [showStats, setShowStats] = useState(false);
  
  if (!simulationData) {
    return (
      <div className="h-full flex items-center justify-center bg-dark-bg">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-purple-400/20 to-blue-500/20 rounded-2xl flex items-center justify-center">
            <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
          </div>
          <h2 className="text-2xl font-bold mb-2 text-glow">3D Visualization</h2>
          <p className="text-gray-400">
            No simulation data available. Upload a physics file to see the visualization.
          </p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="h-full relative bg-dark-bg">
      {/* Controls overlay */}
      <div className="absolute top-4 right-4 z-10 space-y-2">
        <button
          onClick={() => setShowStats(!showStats)}
          className="btn-ghost"
        >
          Stats
        </button>
      </div>
      
      {/* 3D Canvas */}
      <Canvas
        shadows
        camera={{ position: [10, 10, 10], fov: 50 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
      >
        {showStats && <Stats />}
        
        <PerspectiveCamera makeDefault position={[10, 10, 10]} />
        
        <OrbitControls 
          enableDamping
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={50}
        />
        
        <Scene 
          simulationData={simulationData} 
          currentStep={simulationState.currentStep} 
        />
      </Canvas>
      
      {/* Info overlay */}
      <div className="absolute bottom-4 left-4 z-10 glass-effect p-4 rounded-xl border border-gray-700">
        <div className="text-sm space-y-1">
          <div className="text-gray-400">Current Time:</div>
          <div className="text-white font-mono">{simulationState.currentTime.toFixed(3)}s</div>
          <div className="text-gray-400">Step:</div>
          <div className="text-white font-mono">
            {simulationState.currentStep}/{simulationData.time_points.length - 1}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizationCanvas;