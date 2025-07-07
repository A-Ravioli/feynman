import React from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Square, RotateCcw, SkipBack, SkipForward } from 'lucide-react';
import { SimulationData, SimulationState } from '../../types';

interface SimulationControlsProps {
  simulationData: SimulationData;
  simulationState: SimulationState;
  onPlaybackControl: (action: 'play' | 'pause' | 'stop' | 'reset') => void;
  onTimeChange: (time: number, step: number) => void;
  onSpeedChange: (speed: number) => void;
}

const SimulationControls: React.FC<SimulationControlsProps> = ({
  simulationData,
  simulationState,
  onPlaybackControl,
  onTimeChange,
  onSpeedChange,
}) => {
  const maxStep = simulationData.time_points.length - 1;
  const progress = maxStep > 0 ? (simulationState.currentStep / maxStep) * 100 : 0;

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const step = parseInt(event.target.value);
    const time = simulationData.time_points[step];
    onTimeChange(time, step);
  };

  const handleStepForward = () => {
    const nextStep = Math.min(simulationState.currentStep + 1, maxStep);
    const time = simulationData.time_points[nextStep];
    onTimeChange(time, nextStep);
  };

  const handleStepBackward = () => {
    const prevStep = Math.max(simulationState.currentStep - 1, 0);
    const time = simulationData.time_points[prevStep];
    onTimeChange(time, prevStep);
  };

  const speedOptions = [0.25, 0.5, 1, 2, 4, 8];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-16 glass-effect border-b border-gray-700 px-6 flex items-center justify-between"
    >
      {/* Playback Controls */}
      <div className="flex items-center space-x-2">
        <button
          onClick={() => onPlaybackControl('reset')}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-all duration-200"
        >
          <RotateCcw className="w-5 h-5" />
        </button>

        <button
          onClick={handleStepBackward}
          disabled={simulationState.currentStep === 0}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <SkipBack className="w-5 h-5" />
        </button>

        <button
          onClick={() => onPlaybackControl(simulationState.isPlaying ? 'pause' : 'play')}
          className="p-3 rounded-lg bg-blue-500 hover:bg-blue-600 text-white transition-all duration-200 neon-glow"
        >
          {simulationState.isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6" />
          )}
        </button>

        <button
          onClick={handleStepForward}
          disabled={simulationState.currentStep === maxStep}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <SkipForward className="w-5 h-5" />
        </button>

        <button
          onClick={() => onPlaybackControl('stop')}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-all duration-200"
        >
          <Square className="w-5 h-5" />
        </button>
      </div>

      {/* Progress Slider */}
      <div className="flex-1 mx-8">
        <div className="relative">
          <input
            type="range"
            min="0"
            max={maxStep}
            value={simulationState.currentStep}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            style={{
              background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${progress}%, #374151 ${progress}%, #374151 100%)`
            }}
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>{simulationData.time_points[0]?.toFixed(2)}s</span>
            <span>{simulationState.currentTime.toFixed(3)}s</span>
            <span>{simulationData.time_points[maxStep]?.toFixed(2)}s</span>
          </div>
        </div>
      </div>

      {/* Speed Control */}
      <div className="flex items-center space-x-3">
        <span className="text-sm text-gray-400">Speed:</span>
        <select
          value={simulationState.playbackSpeed}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
          className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg px-3 py-1 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        >
          {speedOptions.map(speed => (
            <option key={speed} value={speed}>
              {speed}x
            </option>
          ))}
        </select>
      </div>

      {/* Time Display */}
      <div className="text-sm text-gray-300 font-mono bg-gray-700 px-3 py-1 rounded-lg">
        {simulationState.currentStep}/{maxStep}
      </div>
    </motion.div>
  );
};

export default SimulationControls;