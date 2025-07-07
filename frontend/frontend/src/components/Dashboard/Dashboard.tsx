import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Zap, 
  Clock, 
  Layers, 
  TrendingUp, 
  Target,
  Atom,
  Waves,
  BarChart3,
  PieChart,
  LineChart
} from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart as RechartsPieChart,
  Cell,
  BarChart as RechartsBarChart,
  Bar
} from 'recharts';

import { SimulationData, SimulationState } from '../../types';

interface DashboardProps {
  simulationData: SimulationData | null;
  simulationState: SimulationState;
}

interface MetricCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  color: string;
  trend?: number;
  animate?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  value, 
  subtitle, 
  icon, 
  color, 
  trend, 
  animate = true 
}) => (
  <motion.div
    initial={animate ? { opacity: 0, y: 20 } : {}}
    animate={animate ? { opacity: 1, y: 0 } : {}}
    className="card-glow p-6 relative overflow-hidden"
  >
    {/* Background pattern */}
    <div className="absolute inset-0 opacity-5">
      <div className="w-full h-full bg-cyber-grid" />
    </div>
    
    <div className="relative z-10">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${color}`}>
          {icon}
        </div>
        {trend !== undefined && (
          <div className={`flex items-center space-x-1 text-sm ${
            trend > 0 ? 'text-green-400' : trend < 0 ? 'text-red-400' : 'text-gray-400'
          }`}>
            <TrendingUp className={`w-4 h-4 ${trend < 0 ? 'rotate-180' : ''}`} />
            <span>{Math.abs(trend)}%</span>
          </div>
        )}
      </div>
      
      <div className="space-y-1">
        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
          {title}
        </h3>
        <p className="text-3xl font-bold text-white font-mono text-glow">
          {value}
        </p>
        <p className="text-sm text-gray-500">
          {subtitle}
        </p>
      </div>
    </div>
  </motion.div>
);

const Dashboard: React.FC<DashboardProps> = ({ simulationData, simulationState }) => {
  const metrics = useMemo(() => {
    if (!simulationData) {
      return {
        entityCount: 0,
        timeSteps: 0,
        simulationType: 'None',
        totalEnergy: 0,
        energyData: [],
        entityDistribution: [],
        performanceData: [],
      };
    }

    const entities = Object.values(simulationData.entities);
    const entityCount = entities.length;
    const timeSteps = simulationData.time_points.length;
    const simulationType = simulationData.simulation_parameters.model_type;

    // Calculate energy metrics for classical simulations
    let totalEnergy = 0;
    let energyData: Array<{ time: number; kinetic: number; potential: number; total: number }> = [];
    
    if (simulationType === 'classical') {
      simulationData.time_points.forEach((time, index) => {
        let kineticSum = 0;
        let potentialSum = 0;

        entities.forEach(entity => {
          if (entity.time_series.type === 'object') {
            const ke = entity.time_series.kinetic_energy?.[index] || 0;
            kineticSum += ke;
          }
        });

        const total = kineticSum + potentialSum;
        energyData.push({
          time: parseFloat(time.toFixed(3)),
          kinetic: kineticSum,
          potential: potentialSum,
          total: total,
        });

        if (index === simulationData.time_points.length - 1) {
          totalEnergy = total;
        }
      });
    } else if (simulationType === 'quantum') {
      // Quantum energy data
      simulationData.time_points.forEach((time, index) => {
        let totalEnergySum = 0;
        let kineticSum = 0;
        let potentialSum = 0;

        entities.forEach(entity => {
          if (entity.time_series.type === 'atom') {
            const ke = entity.time_series.expected_kinetic_energy?.[index] || 0;
            const pe = entity.time_series.expected_potential_energy?.[index] || 0;
            const te = entity.time_series.expected_energy?.[index] || 0;
            
            kineticSum += ke;
            potentialSum += pe;
            totalEnergySum += te;
          }
        });

        energyData.push({
          time: parseFloat(time.toFixed(3)),
          kinetic: kineticSum,
          potential: potentialSum,
          total: totalEnergySum,
        });

        if (index === simulationData.time_points.length - 1) {
          totalEnergy = totalEnergySum;
        }
      });
    }

    // Entity distribution
    const entityTypes = entities.reduce((acc, entity) => {
      const type = entity.time_series.type;
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const entityDistribution = Object.entries(entityTypes).map(([type, count]) => ({
      name: type.charAt(0).toUpperCase() + type.slice(1),
      value: count,
      color: type === 'object' ? '#3b82f6' : type === 'atom' ? '#8b5cf6' : '#10b981'
    }));

    // Performance data (mock for now)
    const performanceData = simulationData.time_points.slice(0, 10).map((time, index) => ({
      time: parseFloat(time.toFixed(3)),
      fps: 60 - Math.random() * 5,
      memory: 45 + Math.random() * 10,
      cpu: 30 + Math.random() * 20,
    }));

    return {
      entityCount,
      timeSteps,
      simulationType,
      totalEnergy,
      energyData: energyData.slice(0, 50), // Limit data points for performance
      entityDistribution,
      performanceData,
    };
  }, [simulationData]);

  const formatEnergy = (energy: number): string => {
    if (Math.abs(energy) < 1e-15) return '0 J';
    if (Math.abs(energy) < 1e-12) return `${(energy * 1e15).toFixed(2)} fJ`;
    if (Math.abs(energy) < 1e-9) return `${(energy * 1e12).toFixed(2)} pJ`;
    if (Math.abs(energy) < 1e-6) return `${(energy * 1e9).toFixed(2)} nJ`;
    if (Math.abs(energy) < 1e-3) return `${(energy * 1e6).toFixed(2)} μJ`;
    if (Math.abs(energy) < 1) return `${(energy * 1e3).toFixed(2)} mJ`;
    return `${energy.toFixed(3)} J`;
  };

  if (!simulationData) {
    return (
      <div className="h-full flex items-center justify-center">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-400/20 to-purple-500/20 rounded-2xl flex items-center justify-center">
            <BarChart3 className="w-12 h-12 text-blue-400" />
          </div>
          <h2 className="text-2xl font-bold mb-2 text-glow">Welcome to Feynman Physics Lab</h2>
          <p className="text-gray-400 mb-8 max-w-md mx-auto">
            Upload a physics simulation file or load an example to get started with advanced visualization and analysis.
          </p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="btn-primary"
          >
            Get Started
          </motion.button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-glow mb-2">Simulation Dashboard</h1>
          <p className="text-gray-400">
            Real-time analysis and metrics for your physics simulation
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`px-4 py-2 rounded-lg border ${
            simulationState.isPlaying 
              ? 'border-green-400 bg-green-400/10 text-green-400' 
              : 'border-gray-600 bg-gray-600/10 text-gray-400'
          }`}>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                simulationState.isPlaying ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
              }`} />
              <span className="text-sm font-medium">
                {simulationState.isPlaying ? 'Playing' : 'Paused'}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Entities"
          value={metrics.entityCount.toString()}
          subtitle="Total objects in simulation"
          icon={<Layers className="w-6 h-6 text-white" />}
          color="from-blue-500 to-cyan-500"
        />
        
        <MetricCard
          title="Time Steps"
          value={metrics.timeSteps.toLocaleString()}
          subtitle="Total simulation frames"
          icon={<Clock className="w-6 h-6 text-white" />}
          color="from-purple-500 to-pink-500"
        />
        
        <MetricCard
          title="Simulation Type"
          value={metrics.simulationType.charAt(0).toUpperCase() + metrics.simulationType.slice(1)}
          subtitle="Physics model in use"
          icon={metrics.simulationType === 'quantum' ? 
            <Atom className="w-6 h-6 text-white" /> : 
            <Waves className="w-6 h-6 text-white" />
          }
          color="from-green-500 to-teal-500"
        />
        
        <MetricCard
          title="Total Energy"
          value={formatEnergy(metrics.totalEnergy)}
          subtitle="System energy conservation"
          icon={<Zap className="w-6 h-6 text-white" />}
          color="from-orange-500 to-red-500"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Energy Evolution Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-glow">Energy Evolution</h3>
            <LineChart className="w-5 h-5 text-blue-400" />
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics.energyData}>
                <defs>
                  <linearGradient id="kineticGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05}/>
                  </linearGradient>
                  <linearGradient id="potentialGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05}/>
                  </linearGradient>
                  <linearGradient id="totalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9ca3af"
                  fontSize={12}
                  tickFormatter={(value) => formatEnergy(value)}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                  formatter={(value: number) => formatEnergy(value)}
                />
                <Area
                  type="monotone"
                  dataKey="kinetic"
                  stackId="1"
                  stroke="#3b82f6"
                  fill="url(#kineticGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="potential"
                  stackId="1"
                  stroke="#8b5cf6"
                  fill="url(#potentialGradient)"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="total"
                  stroke="#10b981"
                  strokeWidth={3}
                  strokeDasharray="5 5"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Entity Distribution */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-glow">Entity Distribution</h3>
            <PieChart className="w-5 h-5 text-purple-400" />
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={metrics.entityDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {metrics.entityDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-4 space-y-2">
            {metrics.entityDistribution.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-300">{item.name}</span>
                </div>
                <span className="text-sm text-gray-400">{item.value}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Performance Monitor */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-glow">Performance Metrics</h3>
            <Activity className="w-5 h-5 text-green-400" />
          </div>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsLineChart data={metrics.performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="fps"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="FPS"
                />
                <Line
                  type="monotone"
                  dataKey="memory"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="Memory %"
                />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="CPU %"
                />
              </RechartsLineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Current State Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-glow">Current State</h3>
            <Target className="w-5 h-5 text-cyan-400" />
          </div>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Current Time</span>
              <span className="text-lg font-mono text-white">
                {simulationState.currentTime.toFixed(3)}s
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Progress</span>
              <span className="text-lg font-mono text-white">
                {simulationState.currentStep}/{metrics.timeSteps - 1}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Speed</span>
              <span className="text-lg font-mono text-white">
                {simulationState.playbackSpeed}x
              </span>
            </div>
            
            <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
              <div 
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ 
                  width: `${(simulationState.currentStep / Math.max(metrics.timeSteps - 1, 1)) * 100}%` 
                }}
              />
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;