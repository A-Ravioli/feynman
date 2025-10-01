// Physics simulation data types
export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface TimePoint {
  time: number;
  index: number;
}

export interface SimulationParameters {
  time_start: number;
  time_end: number;
  time_step: number;
  model_type: 'classical' | 'quantum';
}

export interface EntityProperties {
  type: 'object' | 'atom' | 'field';
  mass?: number;
  position?: number[];
  velocity?: number[];
  charge?: number;
  size?: number;
  color?: string;
  shape?: string;
}

export interface ClassicalEntityData {
  initial_properties: EntityProperties;
  time_series: {
    positions: number[][];
    velocities: number[][];
    kinetic_energy: number[];
    type: 'object';
  };
}

export interface QuantumEntityData {
  initial_properties: EntityProperties;
  time_series: {
    expected_position: number[][];
    expected_momentum: number[][];
    expected_kinetic_energy: number[];
    expected_potential_energy: number[];
    expected_energy: number[];
    wavefunction_flat?: number[][];
    probability_density_flat?: number[][];
    grid_info?: {
      coords: number[][];
      deltas: number[];
      n_points: number[];
    };
    dimensions: number;
    type: 'atom';
    mass: number;
    spin?: number;
    eigenstates?: {
      eigenvalues: number[];
    };
  };
}

export type EntityData = ClassicalEntityData | QuantumEntityData;

export interface Interaction {
  source: string;
  target: string;
  properties: {
    force?: string | { function: string; args: unknown[] };
    potential?: string | { function: string; args: unknown[] };
  };
}

export interface SimulationData {
  time_points: number[];
  entities: Record<string, EntityData>;
  simulation_parameters: SimulationParameters;
  interactions: Interaction[];
  program?: Record<string, unknown>;
  simulation_type?: 'classical' | 'quantum'; // For frontend compatibility
  quantum_data?: {
    wavefunction_flat: number[][];
    probability_density_flat: number[][];
  };
}

// Visualization types
export interface VisualizationConfig {
  type: 'classical' | 'quantum';
  mode: '2d' | '3d' | 'probability' | 'expectation' | 'phase_space' | 'energy';
  entities: string[];
  showTrails?: boolean;
  showVectors?: boolean;
  colorScheme?: 'default' | 'neon' | 'energy' | 'velocity';
  quality?: 'low' | 'medium' | 'high' | 'ultra';
}

export interface CameraConfig {
  position: Vector3D;
  target: Vector3D;
  fov: number;
  near: number;
  far: number;
}

export interface LightingConfig {
  ambient: number;
  directional: {
    intensity: number;
    position: Vector3D;
    color: string;
  };
  point: {
    intensity: number;
    position: Vector3D;
    color: string;
  }[];
}

export interface SceneConfig {
  camera: CameraConfig;
  lighting: LightingConfig;
  background: string;
  grid: boolean;
  axes: boolean;
}

// Dashboard types
export interface DashboardMetrics {
  totalEntities: number;
  simulationTime: number;
  timeSteps: number;
  energyConserved: boolean;
  momentumConserved: boolean;
  performance: {
    fps: number;
    renderTime: number;
    memoryUsage: number;
  };
}

export interface SimulationState {
  isLoaded: boolean;
  isPlaying: boolean;
  currentTime: number;
  currentStep: number;
  playbackSpeed: number;
  loop: boolean;
}

// File upload types
export interface FileUploadState {
  uploading: boolean;
  progress: number;
  error: string | null;
}

export interface PhysicsFile {
  name: string;
  content: string;
  lastModified: Date;
  size: number;
}

// API types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface SimulationRequest {
  code: string;
  options?: {
    visualize?: boolean;
    output_file?: string;
  };
}

// UI Component types
export interface Tab {
  id: string;
  label: string;
  icon?: string;
  content: React.ReactNode;
  disabled?: boolean;
}

export interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  href?: string;
  onClick?: () => void;
  active?: boolean;
  badge?: string | number;
}

export interface NotificationMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  autoClose?: boolean;
  duration?: number;
}

// 3D Scene types
export interface Particle {
  id: string;
  position: Vector3D;
  velocity?: Vector3D;
  color: string;
  size: number;
  opacity: number;
  trail?: Vector3D[];
}

export interface QuantumWavefunction {
  dimensions: number;
  grid: {
    coords: number[][];
    n_points: number[];
    deltas: number[];
  };
  data: number[][];
  time: number;
}

export interface FieldVisualization {
  type: 'vector' | 'scalar' | 'streamlines';
  data: number[][][];
  colormap: string;
  opacity: number;
  scale: number;
}

// Performance monitoring
export interface PerformanceMetrics {
  frameRate: number;
  renderTime: number;
  memoryUsage: number;
  cpuUsage: number;
  gpuUsage?: number;
  triangleCount: number;
  drawCalls: number;
}

// Theme types
export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    border: string;
  };
  fonts: {
    primary: string;
    mono: string;
  };
  animations: {
    fast: string;
    normal: string;
    slow: string;
  };
}