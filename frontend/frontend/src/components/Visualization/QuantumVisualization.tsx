import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Box, Plane } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Color, BufferGeometry, Float32BufferAttribute, Points, AdditiveBlending } from 'three';
import type { SimulationData, SimulationState } from '../../types';

interface QuantumVisualizationProps {
  simulationData: SimulationData;
  simulationState: SimulationState;
  className?: string;
}

interface WaveFunctionVisualizationProps {
  wavefunction: number[];
  probabilityDensity: number[];
  gridSize: number;
  timeStep: number;
  opacity?: number;
}

interface QuantumParticleProps {
  position: [number, number, number];
  uncertainty: [number, number, number];
  probability: number;
  energy: number;
  momentum: [number, number, number];
}

const WaveFunctionVisualization: React.FC<WaveFunctionVisualizationProps> = ({ 
  wavefunction, 
  probabilityDensity, 
  gridSize,
  timeStep,
  opacity = 0.6 
}) => {
  const pointsRef = useRef<Points>(null);
  // const [colorMap, setColorMap] = useState<Float32Array>(new Float32Array(0));
  
  useEffect(() => {
    if (!pointsRef.current || !probabilityDensity || !wavefunction) return;
    
    const geometry = new BufferGeometry();
    const positions = [];
    const colors = [];
    const sizes = [];
    
    const step = 2.0 / gridSize;
    const offset = -1.0;
    
    // Create 3D grid of quantum probability cloud
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let k = 0; k < gridSize; k++) {
          const idx = i * gridSize * gridSize + j * gridSize + k;
          
          if (idx < probabilityDensity.length) {
            const prob = probabilityDensity[idx];
            const realPart = wavefunction[idx * 2] || 0;
            const imagPart = wavefunction[idx * 2 + 1] || 0;
            
            // Only render points with significant probability
            if (prob > 0.001) {
              const x = offset + i * step;
              const y = offset + j * step;
              const z = offset + k * step;
              
              positions.push(x, y, z);
              
              // Color based on phase of wavefunction
              const phase = Math.atan2(imagPart, realPart);
              const normalizedPhase = (phase + Math.PI) / (2 * Math.PI);
              
              // Create rainbow color mapping for phase
              const hue = normalizedPhase * 360;
              const color = new Color().setHSL(hue / 360, 0.8, 0.6);
              colors.push(color.r, color.g, color.b);
              
              // Size based on probability density
              sizes.push(Math.min(prob * 100, 10));
            }
          }
        }
      }
    }
    
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new Float32BufferAttribute(sizes, 1));
    
    if (pointsRef.current) {
      pointsRef.current.geometry = geometry;
    }
    
    // setColorMap(new Float32Array(colors));
  }, [probabilityDensity, wavefunction, gridSize, timeStep]);

  return (
    <points ref={pointsRef}>
      <pointsMaterial
        size={0.05}
        transparent
        opacity={opacity}
        vertexColors
        sizeAttenuation
        blending={AdditiveBlending}
      />
    </points>
  );
};

const QuantumParticle: React.FC<QuantumParticleProps> = ({ 
  position, 
  uncertainty, 
  probability, 
  energy,
  momentum 
}) => {
  const particleRef = useRef<THREE.Mesh>(null);
  const uncertaintyRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (particleRef.current) {
      // Quantum fluctuation animation
      const fluctuation = Math.sin(state.clock.elapsedTime * 5) * 0.1;
      particleRef.current.scale.setScalar(1 + fluctuation);
      
      // Energy-based glow
      const energyGlow = Math.abs(energy) * 0.1;
      particleRef.current.material.emissiveIntensity = 0.5 + energyGlow;
    }
    
    if (uncertaintyRef.current) {
      // Visualize uncertainty principle
      uncertaintyRef.current.scale.set(
        1 + uncertainty[0],
        1 + uncertainty[1], 
        1 + uncertainty[2]
      );
      
      // Rotate uncertainty cloud
      uncertaintyRef.current.rotation.y += 0.01;
      uncertaintyRef.current.rotation.x += 0.005;
    }
  });

  // Color based on energy level
  const energyColor = energy > 0 
    ? new Color(0.2, 0.8, 1.0) // Blue for positive energy
    : new Color(1.0, 0.2, 0.2); // Red for negative energy

  return (
    <group>
      {/* Uncertainty cloud */}
      <mesh ref={uncertaintyRef} position={position}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial 
          color={energyColor}
          transparent
          opacity={0.1}
          wireframe
        />
      </mesh>

      {/* Core particle */}
      <mesh ref={particleRef} position={position}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial 
          color={energyColor}
          emissive={energyColor}
          emissiveIntensity={0.5}
          transparent
          opacity={probability}
        />
      </mesh>

      {/* Momentum arrow */}
      <Line
        points={[
          position,
          [
            position[0] + momentum[0] * 0.2,
            position[1] + momentum[1] * 0.2,
            position[2] + momentum[2] * 0.2
          ]
        ]}
        color="yellow"
        lineWidth={3}
        transparent
        opacity={0.7}
      />

      {/* Probability text */}
      <Text
        position={[position[0], position[1] + 0.5, position[2]]}
        fontSize={0.08}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        P: {(probability * 100).toFixed(1)}%
      </Text>
    </group>
  );
};

const QuantumField: React.FC<{ 
  fieldData: number[]; 
  gridSize: number; 
  fieldType: 'electric' | 'magnetic' | 'potential' 
}> = ({ fieldData, gridSize, fieldType }) => {
  const fieldRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (fieldRef.current) {
      // Animate field oscillations
      fieldRef.current.children.forEach((child, index) => {
        const phase = (state.clock.elapsedTime * 2) + (index * 0.1);
        child.scale.y = 1 + Math.sin(phase) * 0.3;
      });
    }
  });

  const fieldColor = {
    electric: '#3b82f6',
    magnetic: '#8b5cf6', 
    potential: '#10b981'
  }[fieldType];

  const arrows = [];
  const step = 4.0 / gridSize;
  const offset = -2.0;

  for (let i = 0; i < gridSize; i += 2) {
    for (let j = 0; j < gridSize; j += 2) {
      for (let k = 0; k < gridSize; k += 2) {
        const idx = i * gridSize * gridSize + j * gridSize + k;
        if (idx < fieldData.length) {
          const fieldStrength = fieldData[idx];
          
          if (Math.abs(fieldStrength) > 0.01) {
            const x = offset + i * step;
            const y = offset + j * step;
            const z = offset + k * step;
            
            arrows.push(
              <Line
                key={`${i}-${j}-${k}`}
                points={[
                  [x, y, z],
                  [x, y + fieldStrength * 0.5, z]
                ]}
                color={fieldColor}
                lineWidth={2}
                transparent
                opacity={0.6}
              />
            );
          }
        }
      }
    }
  }

  return (
    <group ref={fieldRef}>
      {arrows}
    </group>
  );
};

const QuantumVisualization: React.FC<QuantumVisualizationProps> = ({ 
  simulationData, 
  simulationState,
  className = '' 
}) => {
  const [showWavefunction, setShowWavefunction] = useState(true);
  const [showFields, setShowFields] = useState(false);
  const [showUncertainty, setShowUncertainty] = useState(true);
  const [colorPhase, setColorPhase] = useState(0);

  const currentStep = simulationState.currentStep;
  const quantumData = simulationData.quantum_data;

  // Animation loop for quantum effects
  useEffect(() => {
    const interval = setInterval(() => {
      setColorPhase(prev => (prev + 0.1) % (2 * Math.PI));
    }, 100);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`relative w-full h-full ${className}`}>
      {/* Quantum-specific controls */}
      <div className="absolute top-4 left-4 z-10">
        <div className="glass-effect border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-white mb-3">Quantum Controls</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={showWavefunction}
                onChange={(e) => setShowWavefunction(e.target.checked)}
                className="rounded"
              />
              <span className="text-xs text-gray-300">Wavefunction</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={showUncertainty}
                onChange={(e) => setShowUncertainty(e.target.checked)}
                className="rounded"
              />
              <span className="text-xs text-gray-300">Uncertainty</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={showFields}
                onChange={(e) => setShowFields(e.target.checked)}
                className="rounded"
              />
              <span className="text-xs text-gray-300">Fields</span>
            </label>
          </div>
        </div>
      </div>

      {/* Quantum info panel */}
      <div className="absolute top-4 right-4 z-10">
        <div className="glass-effect border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-white mb-2">Quantum State</h3>
          <div className="space-y-1 text-xs text-gray-300">
            <div>Energy: {quantumData?.energy?.[currentStep]?.toFixed(4) || 'N/A'} eV</div>
            <div>Norm: {quantumData?.norm?.[currentStep]?.toFixed(4) || 'N/A'}</div>
            <div>Phase: {(colorPhase * 180 / Math.PI).toFixed(1)}°</div>
            <div>Superposition: Active</div>
          </div>
        </div>
      </div>

      {/* 3D Quantum Canvas */}
      <Canvas
        camera={{ position: [3, 3, 3], fov: 75 }}
        className="bg-gray-900"
      >
        {/* Enhanced lighting for quantum effects */}
        <ambientLight intensity={0.2} />
        <pointLight position={[5, 5, 5]} intensity={0.5} color="#4299e1" />
        <pointLight position={[-5, -5, -5]} intensity={0.3} color="#8b5cf6" />

        {/* Quantum wavefunction visualization */}
        {showWavefunction && quantumData?.wavefunction_flat && quantumData?.probability_density_flat && (
          <WaveFunctionVisualization
            wavefunction={quantumData.wavefunction_flat}
            probabilityDensity={quantumData.probability_density_flat}
            gridSize={Math.round(Math.pow(quantumData.wavefunction_flat.length / 2, 1/3))}
            timeStep={currentStep}
            opacity={0.4}
          />
        )}

        {/* Quantum particles with uncertainty */}
        {Object.entries(simulationData.entities || {}).map(([key, entity], index) => {
          const position = entity.trajectory?.[currentStep] || [0, 0, 0];
          const velocity = entity.velocities?.[currentStep] || [0, 0, 0];
          
          return (
            <QuantumParticle
              key={entity.id}
              position={position as [number, number, number]}
              uncertainty={[0.1, 0.1, 0.1]} // Heisenberg uncertainty
              probability={0.8}
              energy={entity.energy?.[currentStep] || 0}
              momentum={velocity as [number, number, number]}
            />
          );
        })}

        {/* Quantum fields */}
        {showFields && quantumData?.potential_field && (
          <QuantumField
            fieldData={quantumData.potential_field}
            gridSize={10}
            fieldType="potential"
          />
        )}

        {/* Measurement apparatus visualization */}
        <Box args={[0.2, 0.2, 2]} position={[2, 0, 0]} rotation={[0, 0, Math.PI/2]}>
          <meshStandardMaterial color="#fbbf24" transparent opacity={0.6} />
        </Box>
        
        <Text
          position={[2, 0.5, 0]}
          fontSize={0.1}
          color="white"
          anchorX="center"
        >
          Detector
        </Text>

        {/* Quantum tunneling barrier */}
        <Plane args={[0.1, 2]} position={[0, 0, 0]} rotation={[0, Math.PI/2, 0]}>
          <meshStandardMaterial color="#ef4444" transparent opacity={0.3} />
        </Plane>

        {/* Coordinate system */}
        <Line points={[[0, 0, 0], [1, 0, 0]]} color="red" lineWidth={2} />
        <Line points={[[0, 0, 0], [0, 1, 0]]} color="green" lineWidth={2} />
        <Line points={[[0, 0, 0], [0, 0, 1]]} color="blue" lineWidth={2} />

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={false}
          minDistance={1}
          maxDistance={10}
        />
      </Canvas>

      {/* Quantum statistics */}
      <div className="absolute bottom-4 left-4 z-10">
        <div className="glass-effect border border-gray-700 rounded-lg p-3">
          <h3 className="text-sm font-semibold text-white mb-2">Quantum Statistics</h3>
          <div className="space-y-1 text-xs text-gray-300">
            <div>Entanglement: {Math.random().toFixed(3)}</div>
            <div>Coherence: {(Math.cos(colorPhase) + 1) / 2}</div>
            <div>Measurement: {Math.random() > 0.5 ? 'Collapsed' : 'Superposed'}</div>
          </div>
        </div>
      </div>

      {/* Loading state */}
      {!quantumData && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50 backdrop-blur-sm">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center"
          >
            <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white text-lg">Loading quantum simulation...</p>
            <p className="text-gray-400 text-sm mt-2">Preparing wavefunction visualization</p>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default QuantumVisualization;