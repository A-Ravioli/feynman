import axios from 'axios';
import { SimulationData, APIResponse, SimulationRequest } from '../types';

// Create axios instance with default config
const api = axios.create({
  baseURL: 'http://localhost:8001', // Python backend URL
  timeout: 30000, // 30 seconds timeout for simulations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      // Redirect to login if needed
    }
    return Promise.reject(error);
  }
);

class SimulationAPI {
  // Run a new simulation from PhysicsLang code
  async runSimulation(code: string, options?: { visualize?: boolean }): Promise<APIResponse<SimulationData>> {
    try {
      const request: SimulationRequest = {
        code,
        options: options || { visualize: false },
      };

      const response = await api.post('/simulate', request);
      
      return {
        success: true,
        data: response.data,
        message: 'Simulation completed successfully',
      };
    } catch (error) {
      console.error('Simulation error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Simulation failed',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Simulation failed',
      };
    }
  }

  // Get existing simulation results
  async getResults(): Promise<SimulationData | null> {
    try {
      const response = await api.get('/results');
      return response.data;
    } catch (error) {
      console.error('Failed to get results:', error);
      return null;
    }
  }

  // Upload a .phys file and run simulation
  async uploadAndSimulate(file: File): Promise<APIResponse<SimulationData>> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/upload-simulate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return {
        success: true,
        data: response.data,
        message: 'File uploaded and simulation completed',
      };
    } catch (error) {
      console.error('Upload error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Upload and simulation failed',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Upload failed',
      };
    }
  }

  // Get system status and health
  async getStatus(): Promise<{ status: string; version: string; uptime: number }> {
    try {
      const response = await api.get('/status');
      return response.data;
    } catch (error) {
      console.error('Status check failed:', error);
      return {
        status: 'offline',
        version: 'unknown',
        uptime: 0,
      };
    }
  }

  // Get available example files
  async getExamples(): Promise<string[]> {
    try {
      const response = await api.get('/examples');
      return response.data.examples || [];
    } catch (error) {
      console.error('Failed to get examples:', error);
      return [];
    }
  }

  // Load a specific example
  async loadExample(filename: string): Promise<APIResponse<SimulationData>> {
    try {
      const response = await api.post(`/examples/${filename}`);
      
      return {
        success: true,
        data: response.data,
        message: `Example '${filename}' loaded successfully`,
      };
    } catch (error) {
      console.error('Failed to load example:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: `Failed to load example '${filename}'`,
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Failed to load example',
      };
    }
  }

  // Validate PhysicsLang code without running simulation
  async validateCode(code: string): Promise<APIResponse<{ valid: boolean; errors: string[]; warnings: string[] }>> {
    try {
      const response = await api.post('/validate', { code });
      
      return {
        success: true,
        data: response.data,
        message: 'Code validation completed',
      };
    } catch (error) {
      console.error('Validation error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Code validation failed',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Validation failed',
      };
    }
  }

  // Get simulation performance metrics
  async getPerformanceMetrics(): Promise<{
    cpu_usage: number;
    memory_usage: number;
    simulation_time: number;
    entity_count: number;
  }> {
    try {
      const response = await api.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to get metrics:', error);
      return {
        cpu_usage: 0,
        memory_usage: 0,
        simulation_time: 0,
        entity_count: 0,
      };
    }
  }

  // Real-time updates via Server-Sent Events
  createEventSource(endpoint: string): EventSource | null {
    try {
      const baseURL = api.defaults.baseURL || 'http://localhost:8000';
      return new EventSource(`${baseURL}${endpoint}`);
    } catch (error) {
      console.error('Failed to create event source:', error);
      return null;
    }
  }

  // WebSocket connection for real-time simulation updates
  createWebSocket(endpoint: string): WebSocket | null {
    try {
      const baseURL = api.defaults.baseURL || 'http://localhost:8000';
      const wsURL = baseURL.replace('http', 'ws');
      return new WebSocket(`${wsURL}${endpoint}`);
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      return null;
    }
  }

  // File operations
  async saveSimulation(name: string, data: SimulationData): Promise<APIResponse<{ id: string }>> {
    try {
      const response = await api.post('/simulations', { name, data });
      
      return {
        success: true,
        data: response.data,
        message: `Simulation '${name}' saved successfully`,
      };
    } catch (error) {
      console.error('Save error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Failed to save simulation',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Save failed',
      };
    }
  }

  async loadSimulation(id: string): Promise<APIResponse<SimulationData>> {
    try {
      const response = await api.get(`/simulations/${id}`);
      
      return {
        success: true,
        data: response.data,
        message: 'Simulation loaded successfully',
      };
    } catch (error) {
      console.error('Load error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Failed to load simulation',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Load failed',
      };
    }
  }

  async listSimulations(): Promise<{ id: string; name: string; created: string; type: string }[]> {
    try {
      const response = await api.get('/simulations');
      return response.data.simulations || [];
    } catch (error) {
      console.error('Failed to list simulations:', error);
      return [];
    }
  }

  async deleteSimulation(id: string): Promise<APIResponse<void>> {
    try {
      await api.delete(`/simulations/${id}`);
      
      return {
        success: true,
        message: 'Simulation deleted successfully',
      };
    } catch (error) {
      console.error('Delete error:', error);
      
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.error || error.message,
          message: 'Failed to delete simulation',
        };
      }
      
      return {
        success: false,
        error: 'Unknown error occurred',
        message: 'Delete failed',
      };
    }
  }
}

// Create and export singleton instance
export const simulationAPI = new SimulationAPI();

// Export types and utilities
export { api };
export type { APIResponse, SimulationRequest };