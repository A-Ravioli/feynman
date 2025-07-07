import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Square, 
  RotateCcw, 
  Upload, 
  Settings, 
  Activity,
  Zap,
  Cpu,
  MonitorSpeaker,
  FileText,
  BarChart3,
  Globe,
  Atom,
  Waves
} from 'lucide-react';

import Dashboard from './components/Dashboard/Dashboard';
import Sidebar from './components/Sidebar/Sidebar';
import Visualization3D from './components/Visualization/Visualization3D';
import QuantumVisualization from './components/Visualization/QuantumVisualization';
import SimulationControls from './components/Controls/SimulationControls';
import FileUpload from './components/FileUpload/FileUpload';
import NotificationCenter from './components/Notifications/NotificationCenter';
import PerformanceMonitor from './components/Performance/PerformanceMonitor';

import { SimulationData, SimulationState, NavigationItem, NotificationMessage } from './types';
import { simulationAPI } from './services/api';

interface AppState {
  simulationData: SimulationData | null;
  simulationState: SimulationState;
  activeTab: string;
  notifications: NotificationMessage[];
  sidebarCollapsed: boolean;
  showPerformance: boolean;
}

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>({
    simulationData: null,
    simulationState: {
      isLoaded: false,
      isPlaying: false,
      currentTime: 0,
      currentStep: 0,
      playbackSpeed: 1,
      loop: false,
    },
    activeTab: 'dashboard',
    notifications: [],
    sidebarCollapsed: false,
    showPerformance: false,
  });

  // Navigation items for the sidebar
  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: 'BarChart3',
      active: appState.activeTab === 'dashboard',
    },
    {
      id: 'visualization',
      label: '3D Visualization',
      icon: 'Globe',
      active: appState.activeTab === 'visualization',
    },
    {
      id: 'quantum',
      label: 'Quantum States',
      icon: 'Atom',
      active: appState.activeTab === 'quantum',
    },
    {
      id: 'fields',
      label: 'Field Visualization',
      icon: 'Waves',
      active: appState.activeTab === 'fields',
    },
    {
      id: 'upload',
      label: 'Upload Simulation',
      icon: 'Upload',
      active: appState.activeTab === 'upload',
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: 'Settings',
      active: appState.activeTab === 'settings',
    },
  ];

  // Load simulation data on component mount
  useEffect(() => {
    loadDefaultSimulation();
  }, []);

  const loadDefaultSimulation = async () => {
    try {
      // Try to load existing results
      const data = await simulationAPI.getResults();
      if (data) {
        setAppState(prev => ({
          ...prev,
          simulationData: data,
          simulationState: {
            ...prev.simulationState,
            isLoaded: true,
          },
        }));
        addNotification('success', 'Simulation Loaded', 'Successfully loaded simulation data');
      }
    } catch (error) {
      console.error('Failed to load simulation:', error);
      addNotification('info', 'No Data', 'No simulation data found. Upload a .phys file to get started.');
    }
  };

  const handleTabChange = (tabId: string) => {
    setAppState(prev => ({
      ...prev,
      activeTab: tabId,
    }));
  };

  const handleSimulationUpload = async (file: File) => {
    try {
      addNotification('info', 'Processing', 'Uploading and processing simulation file...');
      
      const result = await simulationAPI.uploadAndSimulate(file);
      
      if (result.success && result.data) {
        setAppState(prev => ({
          ...prev,
          simulationData: result.data,
          simulationState: {
            ...prev.simulationState,
            isLoaded: true,
            currentTime: 0,
            currentStep: 0,
          },
          activeTab: result.data.simulation_type === 'quantum' ? 'quantum' : 'visualization',
        }));
        
        const simType = result.data.simulation_type === 'quantum' ? 'quantum' : 'classical';
        addNotification('success', 'Simulation Complete', `${simType} physics simulation completed successfully!`);
      } else {
        throw new Error(result.error || 'Simulation failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      addNotification('error', 'Simulation Error', error instanceof Error ? error.message : 'Unknown error occurred');
    }
  };

  const handlePlaybackControl = (action: 'play' | 'pause' | 'stop' | 'reset') => {
    setAppState(prev => ({
      ...prev,
      simulationState: {
        ...prev.simulationState,
        isPlaying: action === 'play',
        currentTime: action === 'reset' ? 0 : prev.simulationState.currentTime,
        currentStep: action === 'reset' ? 0 : prev.simulationState.currentStep,
      },
    }));
  };

  const addNotification = (type: NotificationMessage['type'], title: string, message: string) => {
    const notification: NotificationMessage = {
      id: Date.now().toString(),
      type,
      title,
      message,
      timestamp: new Date(),
      autoClose: type !== 'error',
      duration: 5000,
    };

    setAppState(prev => ({
      ...prev,
      notifications: [notification, ...prev.notifications.slice(0, 4)],
    }));
  };

  const removeNotification = (id: string) => {
    setAppState(prev => ({
      ...prev,
      notifications: prev.notifications.filter(n => n.id !== id),
    }));
  };

  const toggleSidebar = () => {
    setAppState(prev => ({
      ...prev,
      sidebarCollapsed: !prev.sidebarCollapsed,
    }));
  };

  const renderMainContent = () => {
    switch (appState.activeTab) {
      case 'dashboard':
        return (
          <Dashboard 
            simulationData={appState.simulationData}
            simulationState={appState.simulationState}
          />
        );
      
      case 'visualization':
        return (
          <Visualization3D
            simulationData={appState.simulationData || {
              simulation_type: 'classical',
              entities: [],
              time_points: [],
              simulation_parameters: { model_type: 'classical' }
            }}
            simulationState={appState.simulationState}
            className="h-full"
          />
        );
      
      case 'quantum':
        return appState.simulationData?.simulation_type === 'quantum' ? (
          <QuantumVisualization
            simulationData={appState.simulationData}
            simulationState={appState.simulationState}
            className="h-full"
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Atom className="w-16 h-16 mx-auto mb-4 text-blue-400 animate-spin" />
              <h3 className="text-xl font-semibold mb-2">Quantum Visualization</h3>
              <p className="text-gray-400">Load a quantum simulation to see advanced quantum state visualization</p>
            </div>
          </div>
        );
      
      case 'fields':
        return (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Waves className="w-16 h-16 mx-auto mb-4 text-purple-400 animate-pulse" />
              <h3 className="text-xl font-semibold mb-2">Field Visualization</h3>
              <p className="text-gray-400">Electromagnetic and potential field visualization coming soon...</p>
            </div>
          </div>
        );
      
      case 'upload':
        return (
          <FileUpload 
            onFileUpload={handleSimulationUpload}
            onNotification={addNotification}
          />
        );
      
      case 'settings':
        return (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Settings className="w-16 h-16 mx-auto mb-4 text-green-400 animate-spin" />
              <h3 className="text-xl font-semibold mb-2">Settings</h3>
              <p className="text-gray-400">Configuration panel coming soon...</p>
            </div>
          </div>
        );
      
      default:
        return <Dashboard simulationData={appState.simulationData} simulationState={appState.simulationState} />;
    }
  };

  return (
    <div className="h-screen bg-dark-bg overflow-hidden flex cyber-grid">
      {/* Background overlay */}
      <div className="fixed inset-0 bg-gradient-to-br from-blue-900/10 via-purple-900/10 to-pink-900/10 pointer-events-none" />
      
      {/* Sidebar */}
      <Sidebar
        items={navigationItems}
        collapsed={appState.sidebarCollapsed}
        onItemClick={handleTabChange}
        onToggle={toggleSidebar}
      />

      {/* Main content area */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ${
        appState.sidebarCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        {/* Header */}
        <header className="h-16 glass-effect border-b border-gray-700 flex items-center justify-between px-6 z-10">
          <div className="flex items-center space-x-4">
            <motion.h1 
              className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              Feynman Physics Lab
            </motion.h1>
            {appState.simulationData && (
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <Activity className="w-4 h-4" />
                <span>
                  {appState.simulationData.simulation_parameters.model_type === 'quantum' ? 'Quantum' : 'Classical'} Simulation
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center space-x-4">
            {/* Performance toggle */}
            <button
              onClick={() => setAppState(prev => ({ ...prev, showPerformance: !prev.showPerformance }))}
              className={`p-2 rounded-lg transition-colors ${
                appState.showPerformance 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
              }`}
            >
              <Cpu className="w-5 h-5" />
            </button>

            {/* Status indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                appState.simulationState.isLoaded ? 'bg-green-400 animate-pulse' : 'bg-gray-500'
              }`} />
              <span className="text-sm text-gray-400">
                {appState.simulationState.isLoaded ? 'Ready' : 'No Data'}
              </span>
            </div>
          </div>
        </header>

        {/* Simulation controls */}
        {appState.simulationData && (
          <SimulationControls
            simulationData={appState.simulationData}
            simulationState={appState.simulationState}
            onPlaybackControl={handlePlaybackControl}
            onTimeChange={(time, step) => 
              setAppState(prev => ({
                ...prev,
                simulationState: {
                  ...prev.simulationState,
                  currentTime: time,
                  currentStep: step,
                },
              }))
            }
            onSpeedChange={(speed) =>
              setAppState(prev => ({
                ...prev,
                simulationState: {
                  ...prev.simulationState,
                  playbackSpeed: speed,
                },
              }))
            }
          />
        )}

        {/* Main content */}
        <main className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={appState.activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {renderMainContent()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {/* Performance monitor overlay */}
      <AnimatePresence>
        {appState.showPerformance && (
          <PerformanceMonitor
            onClose={() => setAppState(prev => ({ ...prev, showPerformance: false }))}
          />
        )}
      </AnimatePresence>

      {/* Notification center */}
      <NotificationCenter
        notifications={appState.notifications}
        onRemove={removeNotification}
      />
    </div>
  );
};

export default App;
