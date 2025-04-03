import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import base64
import tempfile
import os

class Visualizer:
    """Visualizer for physics simulation results"""
    
    def __init__(self):
        plt.style.use('dark_background')  # Use a nice dark theme for plots
    
    def visualize(self, entity_name: str, entity_data: Dict[str, Any], 
                 time_points: np.ndarray, target_name: Optional[str] = None, 
                 target_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create visualizations for entity data"""
        entity_type = entity_data.get("type", "object")
        
        if entity_type == "object":
            return self._visualize_classical_object(
                entity_name, entity_data, time_points, target_name, target_data
            )
        elif entity_type == "atom":
            return self._visualize_quantum_particle(
                entity_name, entity_data, time_points, target_name, target_data
            )
        elif entity_type == "field":
            return self._visualize_field(
                entity_name, entity_data, time_points, target_name, target_data
            )
        else:
            return {"error": f"Unknown entity type: {entity_type}"}
    
    def _visualize_classical_object(self, name: str, data: Dict[str, Any], 
                                   time_points: np.ndarray, target_name: Optional[str] = None, 
                                   target_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Visualize a classical object's trajectory"""
        positions = data.get("positions", [])
        if len(positions) == 0:
            return {"error": "No position data available"}
        
        # Create a figure with trajectory (2D projection)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # If positions are 3D, use first two dimensions
        if positions.shape[1] >= 2:
            ax.plot(positions[:, 0], positions[:, 1], 'w-', alpha=0.7, label=name)
            ax.scatter(positions[0, 0], positions[0, 1], color='green', s=50, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], color='red', s=50, label='End')
            
            # If target is also an object, plot its trajectory too
            if target_data and target_data.get("type") == "object":
                target_positions = target_data.get("positions", [])
                if len(target_positions) > 0 and target_positions.shape[1] >= 2:
                    ax.plot(target_positions[:, 0], target_positions[:, 1], 'y-', alpha=0.7, label=target_name)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Trajectory of {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save the trajectory plot
            trajectory_img = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Also create energy plot
            fig, ax = plt.subplots(figsize=(10, 6))
            kinetic_energy = data.get("kinetic_energy", [])
            if len(kinetic_energy) > 0:
                ax.plot(time_points, kinetic_energy, 'r-', label='Kinetic Energy')
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy')
                ax.set_title(f'Energy of {name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            energy_img = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Create animation frames (position over time)
            if len(positions) > 1:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                def update(frame):
                    ax.clear()
                    
                    # Show full trajectory as a faint line
                    ax.plot(positions[:, 0], positions[:, 1], 'w-', alpha=0.2)
                    
                    # Show current position
                    ax.scatter(positions[frame, 0], positions[frame, 1], color='cyan', s=100)
                    
                    # If we have target data, show it too
                    if target_data and target_data.get("type") == "object":
                        target_positions = target_data.get("positions", [])
                        if len(target_positions) > frame and target_positions.shape[1] >= 2:
                            ax.scatter(target_positions[frame, 0], target_positions[frame, 1], 
                                      color='yellow', s=100, alpha=0.7)
                    
                    # Set consistent axis limits based on full trajectory
                    ax.set_xlim(np.min(positions[:, 0]) - 0.1, np.max(positions[:, 0]) + 0.1)
                    ax.set_ylim(np.min(positions[:, 1]) - 0.1, np.max(positions[:, 1]) + 0.1)
                    
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.set_title(f'{name} at t={time_points[frame]:.2f}')
                    ax.grid(True, alpha=0.3)
                
                # Create animation (just 30 frames for performance)
                num_frames = min(30, len(positions))
                frame_indices = np.linspace(0, len(positions)-1, num_frames, dtype=int)
                ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=100)
                
                # Save animation as GIF to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                    ani.save(tmpfile.name, writer='pillow', fps=10)
                    
                    # Read and encode the file
                    with open(tmpfile.name, 'rb') as f:
                        animation_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Clean up the temp file
                    os.unlink(tmpfile.name)
                
                plt.close(fig)
                
                return {
                    "trajectory_plot": trajectory_img,
                    "energy_plot": energy_img,
                    "animation": animation_data,
                    "animation_type": "gif"
                }
            
            return {
                "trajectory_plot": trajectory_img,
                "energy_plot": energy_img
            }
        
        return {"error": "Insufficient position data dimensions"}
    
    def _visualize_quantum_particle(self, name: str, data: Dict[str, Any], 
                                  time_points: np.ndarray, target_name: Optional[str] = None, 
                                  target_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Visualize a quantum particle's wavefunction and probability density"""
        wavefunction = data.get("wavefunction", [])
        prob_density = data.get("probability_density", [])
        grid = data.get("grid", [])
        expected_position = data.get("expected_position", [])
        
        if len(wavefunction) == 0 or len(grid) == 0:
            return {"error": "No wavefunction data available"}
        
        # Create wavefunction plot at initial and final times
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Initial wavefunction
        axs[0].plot(grid, np.real(wavefunction[0]), 'b-', label='Re(ψ)')
        axs[0].plot(grid, np.imag(wavefunction[0]), 'r-', label='Im(ψ)')
        axs[0].plot(grid, prob_density[0], 'g-', label='|ψ|²')
        axs[0].set_xlabel('Position')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(f'Initial wavefunction of {name} at t={time_points[0]:.2f}')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Final wavefunction
        axs[1].plot(grid, np.real(wavefunction[-1]), 'b-', label='Re(ψ)')
        axs[1].plot(grid, np.imag(wavefunction[-1]), 'r-', label='Im(ψ)')
        axs[1].plot(grid, prob_density[-1], 'g-', label='|ψ|²')
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_title(f'Final wavefunction of {name} at t={time_points[-1]:.2f}')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        wavefunction_img = self._figure_to_base64(fig)
        plt.close(fig)
        
        # Create a probability density evolution plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # We'll create a 2D heatmap of probability density vs. position and time
        # For performance, we'll downsample to at most 100 time points
        time_indices = np.linspace(0, len(time_points)-1, min(100, len(time_points)), dtype=int)
        grid_indices = np.linspace(0, len(grid)-1, min(200, len(grid)), dtype=int)
        
        prob_density_subset = prob_density[time_indices][:, grid_indices]
        time_subset = time_points[time_indices]
        grid_subset = grid[grid_indices]
        
        # Create a mesh grid for the plot
        T, X = np.meshgrid(time_subset, grid_subset, indexing='ij')
        
        # Create the heatmap
        c = ax.pcolormesh(T, X, prob_density_subset, cmap='viridis', shading='auto')
        fig.colorbar(c, ax=ax, label='Probability Density |ψ|²')
        
        # Plot the expected position if available
        if len(expected_position) > 0:
            ax.plot(time_points, expected_position, 'r-', linewidth=2, label='<x>')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title(f'Probability Density Evolution of {name}')
        
        # Add a legend if we plotted expected position
        if len(expected_position) > 0:
            ax.legend()
        
        probability_evolution_img = self._figure_to_base64(fig)
        plt.close(fig)
        
        # Create animation of wavefunction evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update(frame):
            ax.clear()
            ax.plot(grid, np.real(wavefunction[frame]), 'b-', alpha=0.7, label='Re(ψ)')
            ax.plot(grid, np.imag(wavefunction[frame]), 'r-', alpha=0.7, label='Im(ψ)')
            ax.plot(grid, prob_density[frame], 'g-', label='|ψ|²')
            
            # Add expected position marker
            if len(expected_position) > frame:
                ax.axvline(x=expected_position[frame], color='yellow', linestyle='--', label='<x>')
            
            # Add potential if target is a field with a specified region
            if target_data and target_data.get("type") == "field":
                if "region" in target_data:
                    for region in target_data["region"]:
                        ax.axvspan(region[0], region[1], alpha=0.2, color='gray')
            
            ax.set_ylim(-1, 1)  # Consistent y limits
            ax.set_xlabel('Position')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'{name} at t={time_points[frame]:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Create animation (just 30 frames for performance)
        num_frames = min(30, len(wavefunction))
        frame_indices = np.linspace(0, len(wavefunction)-1, num_frames, dtype=int)
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=100)
        
        # Save animation as GIF to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            ani.save(tmpfile.name, writer='pillow', fps=10)
            
            # Read and encode the file
            with open(tmpfile.name, 'rb') as f:
                animation_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up the temp file
            os.unlink(tmpfile.name)
        
        plt.close(fig)
        
        return {
            "wavefunction_plot": wavefunction_img,
            "probability_evolution": probability_evolution_img,
            "animation": animation_data,
            "animation_type": "gif"
        }
    
    def _visualize_field(self, name: str, data: Dict[str, Any], 
                        time_points: np.ndarray, target_name: Optional[str] = None, 
                        target_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Visualize a field (e.g., detector screen)"""
        # For now, let's just return a simple message
        # Future: implement field visualization based on field type
        return {
            "message": f"Field visualization for {name} is not fully implemented yet"
        }
    
    def _figure_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str 