import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { X, Cpu, HardDrive, Activity, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

interface PerformanceMonitorProps {
  onClose: () => void;
}

interface PerformanceData {
  timestamp: number;
  fps: number;
  memory: number;
  cpu: number;
  gpu: number;
}

const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({ onClose }) => {
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState({
    fps: 60,
    memory: 45,
    cpu: 25,
    gpu: 30,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const newData: PerformanceData = {
        timestamp: now,
        fps: 60 - Math.random() * 5,
        memory: 40 + Math.random() * 20,
        cpu: 20 + Math.random() * 30,
        gpu: 25 + Math.random() * 25,
      };

      setCurrentMetrics({
        fps: newData.fps,
        memory: newData.memory,
        cpu: newData.cpu,
        gpu: newData.gpu,
      });

      setPerformanceData(prev => {
        const updated = [...prev, newData];
        return updated.slice(-50); // Keep last 50 data points
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatValue = (value: number, unit: string) => {
    return `${value.toFixed(1)}${unit}`;
  };

  const getColor = (value: number, type: 'fps' | 'memory' | 'cpu' | 'gpu') => {
    switch (type) {
      case 'fps':
        return value > 50 ? 'text-green-400' : value > 30 ? 'text-yellow-400' : 'text-red-400';
      case 'memory':
      case 'cpu':
      case 'gpu':
        return value < 50 ? 'text-green-400' : value < 75 ? 'text-yellow-400' : 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const chartData = performanceData.map((data, index) => ({
    time: index,
    fps: data.fps,
    memory: data.memory,
    cpu: data.cpu,
    gpu: data.gpu,
  }));

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="fixed top-20 right-4 w-96 h-80 glass-effect border border-gray-700 rounded-xl z-50 overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-green-400" />
          <h3 className="font-semibold text-white">Performance Monitor</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-gray-700 transition-colors"
        >
          <X className="w-4 h-4 text-gray-400" />
        </button>
      </div>

      {/* Metrics */}
      <div className="p-4 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Zap className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-gray-400">FPS</span>
            </div>
            <div className={`text-lg font-mono font-semibold ${getColor(currentMetrics.fps, 'fps')}`}>
              {formatValue(currentMetrics.fps, '')}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <HardDrive className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-gray-400">Memory</span>
            </div>
            <div className={`text-lg font-mono font-semibold ${getColor(currentMetrics.memory, 'memory')}`}>
              {formatValue(currentMetrics.memory, '%')}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Cpu className="w-4 h-4 text-green-400" />
              <span className="text-sm text-gray-400">CPU</span>
            </div>
            <div className={`text-lg font-mono font-semibold ${getColor(currentMetrics.cpu, 'cpu')}`}>
              {formatValue(currentMetrics.cpu, '%')}
            </div>
          </div>

          <div className="bg-gray-800/50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Activity className="w-4 h-4 text-orange-400" />
              <span className="text-sm text-gray-400">GPU</span>
            </div>
            <div className={`text-lg font-mono font-semibold ${getColor(currentMetrics.gpu, 'gpu')}`}>
              {formatValue(currentMetrics.gpu, '%')}
            </div>
          </div>
        </div>

        {/* Mini Chart */}
        <div className="h-24 bg-gray-800/50 rounded-lg p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis hide />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb',
                  fontSize: '12px'
                }}
              />
              <Line
                type="monotone"
                dataKey="fps"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="cpu"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="memory"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </motion.div>
  );
};

export default PerformanceMonitor;