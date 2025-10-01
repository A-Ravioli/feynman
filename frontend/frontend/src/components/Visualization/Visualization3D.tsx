import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, Text, Line, Box } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Vector3, Color, BufferGeometry, Float32BufferAttribute } from 'three';
import type { SimulationData, SimulationState, EntityData } from '../../types';

interface Visualization3DProps {
  simulationData: SimulationData;
  simulationState: SimulationState;
  className?: string;
}

interface ParticleProps {
  entity: EntityData;
  position: [number, number, number];
  velocity: [number, number, number];
  isQuantum?: boolean;
  wavefunction?: number[];
  size?: number;
}

interface QuantumCloudProps {
  wavefunction: number[];
  probabilityDensity: number[];
  gridSize: number;
  opacity?: number;
}

const Particle: React.FC<ParticleProps> = ({ 
  entity, 
  position, 
  velocity, 
  isQuantum = false, 
  size = 0.1 
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const trailRef = useRef<THREE.Line>(null);
  const [trail, setTrail] = useState<Vector3[]>([]);
  
  useEffect(() => {
    // Update trail positions
    const newPos = new Vector3(...position);
    setTrail(prev => [...prev.slice(-20), newPos]);
  }, [position]);

  useFrame((state) => {
    if (meshRef.current) {
      // Smooth position transitions
      meshRef.current.position.lerp(new Vector3(...position), 0.1);
      
      // Glow effect for quantum particles
      if (isQuantum) {
        const scale = 1 + 0.2 * Math.sin(state.clock.elapsedTime * 3);
        meshRef.current.scale.setScalar(scale);
      }
    }
  });

  const velocityMagnitude = Math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2);
  const color = isQuantum 
    ? new Color(0.5 + 0.5 * Math.sin(Date.now() * 0.001), 0.8, 1.0)
    : new Color().setHSL(Math.min(velocityMagnitude * 0.1, 1), 0.8, 0.6);

  return (
    <group>
      {/* Particle */}
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={isQuantum ? 0.5 : 0.2}
          transparent={isQuantum}
          opacity={isQuantum ? 0.8 : 1.0}
        />
      </mesh>

      {/* Velocity vector */}
      <Line
        points={[position, [position[0] + velocity[0] * 0.1, position[1] + velocity[1] * 0.1, position[2] + velocity[2] * 0.1]]}
        color="yellow"
        lineWidth={2}
        transparent
        opacity={0.6}
      />

      {/* Trail */}
      {trail.length > 1 && (
        <Line
          points={trail.map(v => [v.x, v.y, v.z])}
          color={color}
          lineWidth={1}
          transparent
          opacity={0.3}
        />
      )}

      {/* Entity label */}
      <Text
        position={[position[0], position[1] + size + 0.2, position[2]]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {`Particle`}
      </Text>
    </group>
  );
};

const QuantumCloud: React.FC<QuantumCloudProps> = ({ 
  wavefunction, 
  probabilityDensity, 
  gridSize,
  opacity = 0.3 
}) => {
  const cloudRef = useRef<THREE.Points>(null);
  
  useEffect(() => {
    if (!cloudRef.current || !probabilityDensity) return;
    
    const geometry = new BufferGeometry();
    const positions = [];
    const colors = [];
    const sizes = [];
    
    const step = 2.0 / gridSize;
    const offset = -1.0;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let k = 0; k < gridSize; k++) {
          const idx = i * gridSize * gridSize + j * gridSize + k;
          const prob = probabilityDensity[idx] || 0;
          
          if (prob > 0.001) { // Only render visible points
            const x = offset + i * step;
            const y = offset + j * step;
            const z = offset + k * step;
            
            positions.push(x, y, z);
            
            // Color based on probability density
            const intensity = Math.min(prob * 10, 1);
            colors.push(0.5 + intensity * 0.5, 0.8, 1.0);
            
            sizes.push(prob * 50);
          }
        }
      }
    }
    
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new Float32BufferAttribute(sizes, 1));
    
    cloudRef.current.geometry = geometry;
  }, [probabilityDensity, gridSize]);

  return (
    <points ref={cloudRef}>
      <pointsMaterial
        size={0.05}
        transparent
        opacity={opacity}
        vertexColors
        sizeAttenuation
      />
    </points>
  );
};

const Scene: React.FC<{
  simulationData: SimulationData;
  simulationState: SimulationState;
}> = ({ simulationData, simulationState }) => {
  const { scene } = useThree();
  const currentStep = simulationState.currentStep;
  
  // Get current simulation frame data
  const getCurrentFrameData = () => {
    const entitiesArray = Object.values(simulationData.entities || {});
    if (entitiesArray.length === 0) {
      return { positions: [], velocities: [], entities: [] };
    }
    
    const positions = entitiesArray.map(entity => {
      if (entity.time_series.type === 'object') {
        const pos = entity.time_series.positions?.[currentStep] || [0, 0, 0];
        return pos as [number, number, number];
      } else {
        const pos = entity.time_series.expected_position?.[currentStep] || [0, 0, 0];
        return pos as [number, number, number];
      }
    });
    
    const velocities = entitiesArray.map(entity => {
      if (entity.time_series.type === 'object') {
        const vel = entity.time_series.velocities?.[currentStep] || [0, 0, 0];
        return vel as [number, number, number];
      } else {
        const vel = entity.time_series.expected_momentum?.[currentStep] || [0, 0, 0];
        return vel as [number, number, number];
      }
    });
    
    return { positions, velocities, entities: entitiesArray };
  };

  const { positions, velocities, entities } = getCurrentFrameData();

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#4299e1" />

      {/* Grid */}
      <Grid
        args={[10, 10]}
        position={[0, -2, 0]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#374151"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#4b5563"
        fadeDistance={20}
        fadeStrength={1}
      />

      {/* Coordinate system */}
      <Line points={[[0, 0, 0], [2, 0, 0]]} color="red" lineWidth={3} />
      <Line points={[[0, 0, 0], [0, 2, 0]]} color="green" lineWidth={3} />
      <Line points={[[0, 0, 0], [0, 0, 2]]} color="blue" lineWidth={3} />

      {/* Axis labels */}
      <Text position={[2.2, 0, 0]} fontSize={0.2} color="red">X</Text>
      <Text position={[0, 2.2, 0]} fontSize={0.2} color="green">Y</Text>
      <Text position={[0, 0, 2.2]} fontSize={0.2} color="blue">Z</Text>

      {/* Particles */}
      {entities.map((entity, index) => (
        <Particle
          key={index}
          entity={entity}
          position={positions[index] || [0, 0, 0]}
          velocity={velocities[index] || [0, 0, 0]}
          isQuantum={simulationData.simulation_parameters?.model_type === 'quantum'}
          size={entity.initial_properties?.mass ? Math.max(0.05, Math.min(0.3, entity.initial_properties.mass * 1e30)) : 0.1}
        />
      ))}

      {/* Quantum wavefunction cloud */}
      {simulationData.simulation_parameters?.model_type === 'quantum' && 
       simulationData.quantum_data?.wavefunction_flat && 
       simulationData.quantum_data?.probability_density_flat && (
        <QuantumCloud
          wavefunction={simulationData.quantum_data.wavefunction_flat.flat()}
          probabilityDensity={simulationData.quantum_data.probability_density_flat.flat()}
          gridSize={Math.round(Math.pow(simulationData.quantum_data.wavefunction_flat.flat().length, 1/3))}
          opacity={0.2}
        />
      )}

      {/* Bounding box for simulation space */}
      <Box args={[4, 4, 4]} position={[0, 0, 0]}>
        <meshBasicMaterial wireframe color="#374151" opacity={0.1} transparent />
      </Box>
    </>
  );
};

const Visualization3D: React.FC<Visualization3DProps> = ({ 
  simulationData, 
  simulationState,
  className = '' 
}) => {
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([5, 5, 5]);
  const [autoRotate, setAutoRotate] = useState(false);

  // Camera controls
  const resetCamera = () => {
    setCameraPosition([5, 5, 5]);
  };

  const presetViews = [
    { name: 'Isometric', position: [5, 5, 5] as [number, number, number] },
    { name: 'Front', position: [0, 0, 8] as [number, number, number] },
    { name: 'Side', position: [8, 0, 0] as [number, number, number] },
    { name: 'Top', position: [0, 8, 0] as [number, number, number] },
  ];

  return (
    <div className={`relative w-full h-full ${className}`}>
      {/* Camera Controls */}
      <div className="absolute top-4 left-4 z-10 space-y-2">
        <div className="glass-effect border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-white mb-2">Camera</h3>
          <div className="space-y-1">
            {presetViews.map((view) => (
              <button
                key={view.name}
                onClick={() => setCameraPosition(view.position)}
                className="w-full px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
              >
                {view.name}
              </button>
            ))}
          </div>
          <div className="mt-2 flex items-center space-x-2">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              className="rounded"
            />
            <label className="text-xs text-gray-300">Auto Rotate</label>
          </div>
        </div>
      </div>

      {/* Simulation Info */}
      <div className="absolute top-4 right-4 z-10">
        <div className="glass-effect border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-white mb-2">Simulation</h3>
          <div className="space-y-1 text-xs text-gray-300">
            <div>Type: {simulationData.simulation_parameters?.model_type}</div>
            <div>Entities: {Object.keys(simulationData.entities || {}).length}</div>
            <div>Step: {simulationState.currentStep}</div>
            <div>Time: {simulationState.currentTime.toFixed(3)}s</div>
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: cameraPosition, fov: 60 }}
        className="bg-gray-900"
      >
        <Scene simulationData={simulationData} simulationState={simulationState} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={autoRotate}
          autoRotateSpeed={0.5}
          minDistance={2}
          maxDistance={20}
        />
      </Canvas>

      {/* Loading state */}
      {!simulationData.entities || Object.keys(simulationData.entities).length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center"
          >
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white text-lg">Loading simulation data...</p>
            <p className="text-gray-400 text-sm mt-2">Preparing 3D visualization</p>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default Visualization3D;