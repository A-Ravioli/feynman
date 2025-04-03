import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from typing import Dict, List, Any, Optional, Tuple, Union
from io import BytesIO
import base64
import tempfile
import os

class Visualizer3D:
    """Advanced 3D visualizer for physics simulation results"""
    
    def __init__(self):
        plt.style.use('dark_background')  # Use a nice dark theme for plots
        
        # Default color palette for objects
        self.color_palette = {
            'object': 'dodgerblue',
            'atom': 'magenta',
            'field': 'gold',
            'default': 'white'
        }
        
        # Default transparencies
        self.alpha = {
            'solid': 0.9,
            'field': 0.3,
            'trajectory': 0.5,
            'highlight': 1.0
        }
    
    def visualize_scene(self, entities: Dict[str, Dict[str, Any]], 
                        time_points: np.ndarray,
                        interactions: Optional[List[Dict[str, Any]]] = None,
                        time_index: int = -1,
                        camera_position: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """Create a 3D visualization of the entire physics scene at a specific time index"""
        # Create a figure with 3D axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw all entities
        min_coords = [float('inf'), float('inf'), float('inf')]
        max_coords = [float('-inf'), float('-inf'), float('-inf')]
        
        has_entities = False
        
        for entity_name, entity_data in entities.items():
            has_entities = True
            entity_type = entity_data.get("type", "object")
            
            # Get position at the specified time index
            if entity_type == "object":
                positions = entity_data.get("positions", [])
                if len(positions) > 0:
                    # Update min/max coordinates for better view
                    for dim in range(min(3, positions.shape[1])):
                        min_coords[dim] = min(min_coords[dim], np.min(positions[:, dim]))
                        max_coords[dim] = max(max_coords[dim], np.max(positions[:, dim]))
                    
                    # Draw trajectory
                    self._draw_trajectory(ax, positions, entity_name)
                    
                    # Draw object at current position
                    if 0 <= time_index < len(positions):
                        current_pos = positions[time_index]
                        obj_props = entity_data.get("properties", {})
                        self._draw_object(ax, current_pos, entity_name, obj_props)
            
            elif entity_type == "atom":
                # For quantum particles, we'll show expected position
                expected_position = entity_data.get("expected_position", [])
                if len(expected_position) > 0:
                    # For atoms/quantum objects, create a representation based on probability density
                    if 0 <= time_index < len(expected_position):
                        position = [expected_position[time_index], 0, 0]  # Default to 1D along x-axis
                        
                        # If we have 2D or 3D data, use it
                        if isinstance(expected_position, np.ndarray) and expected_position.ndim > 1:
                            dim = min(3, expected_position.shape[1])
                            position = expected_position[time_index][:dim]
                            while len(position) < 3:
                                position = np.append(position, 0)  # Pad with zeros if needed
                        
                        # Update min/max coordinates
                        for dim in range(3):
                            min_coords[dim] = min(min_coords[dim], position[dim] - 2)
                            max_coords[dim] = max(max_coords[dim], position[dim] + 2)
                        
                        # Draw quantum particle
                        self._draw_quantum_particle(ax, position, entity_name)
            
            elif entity_type == "field":
                # Draw field as a semi-transparent region
                field_props = entity_data.get("properties", {})
                field_region = field_props.get("region", [])
                
                if field_region:
                    field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                    field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                    
                    # Update min/max coordinates
                    for dim in range(3):
                        min_coords[dim] = min(min_coords[dim], field_min[dim])
                        max_coords[dim] = max(max_coords[dim], field_max[dim])
                    
                    # Draw the field
                    self._draw_field(ax, field_min, field_max, entity_name, field_props)
        
        # If no entities were valid, set default bounds
        if not has_entities or min_coords[0] == float('inf') or max_coords[0] == float('-inf'):
            min_coords = [-10, -10, -10]
            max_coords = [10, 10, 10]
        
        # Check for any remaining infinities in coordinates and replace with sensible defaults
        for i in range(3):
            if min_coords[i] == float('inf'):
                min_coords[i] = -10
            if max_coords[i] == float('-inf'):
                max_coords[i] = 10
            # Ensure min and max are different
            if min_coords[i] == max_coords[i]:
                min_coords[i] -= 5
                max_coords[i] += 5
        
        # Draw interactions if provided
        if interactions:
            for interaction in interactions:
                source = interaction.get("source", "")
                target = interaction.get("target", "")
                
                if source in entities and target in entities:
                    source_pos = self._get_entity_position(entities[source], time_index)
                    target_pos = self._get_entity_position(entities[target], time_index)
                    
                    if source_pos is not None and target_pos is not None:
                        self._draw_interaction(ax, source_pos, target_pos, interaction)
        
        # Set equal aspect ratio and labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Visualization at t={time_points[time_index]:.2f}')
        
        # Set axis limits with some padding
        padding = 0.1 * np.array([
            max_coords[0] - min_coords[0],
            max_coords[1] - min_coords[1],
            max_coords[2] - min_coords[2]
        ])
        padding = np.where(padding == 0, 1.0, padding)  # Avoid zero padding
        
        ax.set_xlim(min_coords[0] - padding[0], max_coords[0] + padding[0])
        ax.set_ylim(min_coords[1] - padding[1], max_coords[1] + padding[1])
        ax.set_zlim(min_coords[2] - padding[2], max_coords[2] + padding[2])
        
        # Set custom view if provided, otherwise use a good default
        if camera_position:
            ax.view_init(elev=camera_position[0], azim=camera_position[1], roll=camera_position[2])
        else:
            ax.view_init(elev=20, azim=30)
        
        # Convert to base64 image
        scene_img = self._figure_to_base64(fig)
        plt.close(fig)
        
        return {
            "scene_3d": scene_img,
            "time": time_points[time_index]
        }
    
    def create_animation(self, entities: Dict[str, Dict[str, Any]], 
                         time_points: np.ndarray,
                         interactions: Optional[List[Dict[str, Any]]] = None,
                         camera_position: Optional[Tuple[float, float, float]] = None,
                         num_frames: int = 30) -> str:
        """Create a 3D animation of the physics scene"""
        # Create a figure with 3D axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate overall scene bounds for consistent view
        min_coords = [float('inf'), float('inf'), float('inf')]
        max_coords = [float('-inf'), float('-inf'), float('-inf')]
        
        has_entities = False
        
        for entity_name, entity_data in entities.items():
            has_entities = True
            entity_type = entity_data.get("type", "object")
            
            if entity_type == "object":
                positions = entity_data.get("positions", [])
                if len(positions) > 0:
                    for dim in range(min(3, positions.shape[1])):
                        min_coords[dim] = min(min_coords[dim], np.min(positions[:, dim]))
                        max_coords[dim] = max(max_coords[dim], np.max(positions[:, dim]))
            
            elif entity_type == "atom":
                expected_position = entity_data.get("expected_position", [])
                if len(expected_position) > 0:
                    # If multi-dimensional, handle accordingly
                    if isinstance(expected_position, np.ndarray) and expected_position.ndim > 1:
                        dim = min(3, expected_position.shape[1])
                        for d in range(dim):
                            min_coords[d] = min(min_coords[d], np.min(expected_position[:, d]) - 2)
                            max_coords[d] = max(max_coords[d], np.max(expected_position[:, d]) + 2)
                    else:
                        # Assume 1D along x-axis
                        min_coords[0] = min(min_coords[0], np.min(expected_position) - 2)
                        max_coords[0] = max(max_coords[0], np.max(expected_position) + 2)
                        # Set some default y and z bounds
                        min_coords[1] = min(min_coords[1], -5)
                        max_coords[1] = max(max_coords[1], 5)
                        min_coords[2] = min(min_coords[2], -5)
                        max_coords[2] = max(max_coords[2], 5)
            
            elif entity_type == "field":
                field_props = entity_data.get("properties", {})
                field_region = field_props.get("region", [])
                
                if field_region:
                    field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                    field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                    
                    for dim in range(3):
                        min_coords[dim] = min(min_coords[dim], field_min[dim])
                        max_coords[dim] = max(max_coords[dim], field_max[dim])
        
        # If no entities were valid, set default bounds
        if not has_entities or min_coords[0] == float('inf') or max_coords[0] == float('-inf'):
            min_coords = [-10, -10, -10]
            max_coords = [10, 10, 10]
        
        # Check for any remaining infinities in coordinates and replace with sensible defaults
        for dim in range(3):
            if min_coords[dim] == float('inf'):
                min_coords[dim] = -10
            if max_coords[dim] == float('-inf'):
                max_coords[dim] = 10
            # If min == max, add some space
            if min_coords[dim] == max_coords[dim]:
                min_coords[dim] -= 5
                max_coords[dim] += 5
        
        # Set padding
        padding = 0.2 * np.array([
            max_coords[0] - min_coords[0],
            max_coords[1] - min_coords[1],
            max_coords[2] - min_coords[2]
        ])
        
        # Animation function
        time_indices = np.linspace(0, len(time_points)-1, num_frames, dtype=int)
        
        def update(frame):
            ax.clear()
            time_index = time_indices[frame]
            
            # Draw all entities at current time
            for entity_name, entity_data in entities.items():
                entity_type = entity_data.get("type", "object")
                
                if entity_type == "object":
                    positions = entity_data.get("positions", [])
                    if len(positions) > 0:
                        # Draw trajectory up to current point
                        trajectory = positions[:time_index+1]
                        if len(trajectory) > 0:
                            self._draw_trajectory(ax, trajectory, entity_name, fade=True)
                        
                        # Draw object at current position
                        if 0 <= time_index < len(positions):
                            current_pos = positions[time_index]
                            obj_props = entity_data.get("properties", {})
                            self._draw_object(ax, current_pos, entity_name, obj_props)
                
                elif entity_type == "atom":
                    expected_position = entity_data.get("expected_position", [])
                    if len(expected_position) > 0 and 0 <= time_index < len(expected_position):
                        # For quantum particles
                        if isinstance(expected_position, np.ndarray) and expected_position.ndim > 1:
                            dim = min(3, expected_position.shape[1])
                            position = expected_position[time_index][:dim]
                            while len(position) < 3:
                                position = np.append(position, 0)
                        else:
                            position = [expected_position[time_index], 0, 0]
                        
                        self._draw_quantum_particle(ax, position, entity_name)
                
                elif entity_type == "field":
                    # Fields are typically static, but could change over time in more advanced simulations
                    field_props = entity_data.get("properties", {})
                    field_region = field_props.get("region", [])
                    
                    if field_region:
                        field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                        field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                        self._draw_field(ax, field_min, field_max, entity_name, field_props)
            
            # Draw interactions if provided
            if interactions:
                for interaction in interactions:
                    source = interaction.get("source", "")
                    target = interaction.get("target", "")
                    
                    if source in entities and target in entities:
                        source_pos = self._get_entity_position(entities[source], time_index)
                        target_pos = self._get_entity_position(entities[target], time_index)
                        
                        if source_pos is not None and target_pos is not None:
                            self._draw_interaction(ax, source_pos, target_pos, interaction)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Visualization at t={time_points[time_index]:.2f}')
            
            # Set consistent axis limits
            ax.set_xlim(min_coords[0] - padding[0], max_coords[0] + padding[0])
            ax.set_ylim(min_coords[1] - padding[1], max_coords[1] + padding[1])
            ax.set_zlim(min_coords[2] - padding[2], max_coords[2] + padding[2])
            
            # Set consistent view
            if camera_position:
                ax.view_init(elev=camera_position[0], azim=camera_position[1], roll=camera_position[2])
            else:
                ax.view_init(elev=20, azim=30)
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(time_indices), interval=100)
        
        # Save animation as GIF to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            ani.save(tmpfile.name, writer='pillow', fps=10, dpi=120)
            
            # Read and encode the file
            with open(tmpfile.name, 'rb') as f:
                animation_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up the temp file
            os.unlink(tmpfile.name)
        
        plt.close(fig)
        
        return animation_data
    
    def _draw_object(self, ax, position, name, properties=None):
        """Draw a classical object with the appropriate shape"""
        if properties is None:
            properties = {}
        
        # Determine shape and size
        shape = properties.get("shape", "sphere")
        size = float(properties.get("size", 1.0))
        color = properties.get("color", self.color_palette.get("object"))
        
        if shape == "sphere":
            self._draw_sphere(ax, position, radius=size, color=color, label=name)
        elif shape == "cube" or shape == "block":
            self._draw_cube(ax, position, size=size, color=color, label=name)
        elif shape == "rectangle":
            # For rectangle, use width, height, depth if available
            width = float(properties.get("width", size))
            height = float(properties.get("height", size))
            depth = float(properties.get("depth", size))
            self._draw_rectangle(ax, position, width, height, depth, color=color, label=name)
        else:
            # Default to sphere if shape not recognized
            self._draw_sphere(ax, position, radius=size, color=color, label=name)
    
    def _draw_quantum_particle(self, ax, position, name):
        """Draw a quantum particle (atom)"""
        # For quantum particles, draw a semi-transparent sphere with wave-like surface
        color = self.color_palette.get("atom")
        
        # Base sphere
        self._draw_sphere(ax, position, radius=1.0, color=color, 
                           alpha=self.alpha["highlight"], label=name)
        
        # Add a wave-like effect around it (simplified representation of wavefunction)
        theta = np.linspace(0, 2 * np.pi, 20)
        phi = np.linspace(0, np.pi, 20)
        
        # Create wave effect by modulating the radius
        for i, t in enumerate(np.linspace(0, 1, 3)):  # Draw multiple shells
            # Modulate radius with sine wave to create quantum-like appearance
            radius = 1.2 + 0.2 * i + 0.1 * np.sin(4 * phi[:, np.newaxis] + 6 * theta + t * np.pi)
            
            # Map to cartesian coordinates
            x = position[0] + radius * np.sin(phi[:, np.newaxis]) * np.cos(theta)
            y = position[1] + radius * np.sin(phi[:, np.newaxis]) * np.sin(theta)
            z = position[2] + radius * np.cos(phi[:, np.newaxis]) * np.ones_like(theta)
            
            # Create a lighter color for the outer shells
            shell_color = color
            shell_alpha = max(0.1, self.alpha["field"] * (1 - i * 0.2))
            
            # Plot the shell as a surface
            ax.plot_surface(x, y, z, color=shell_color, alpha=shell_alpha, 
                            linewidth=0, antialiased=True, shade=True)
    
    def _draw_field(self, ax, min_coords, max_coords, name, properties=None):
        """Draw a field as a semi-transparent region"""
        if properties is None:
            properties = {}
        
        field_type = properties.get("type", "generic")
        color = properties.get("color", self.color_palette.get("field"))
        alpha = float(properties.get("alpha", self.alpha["field"]))
        
        # Create the rectangular prism representing the field
        self._draw_rectangle(ax, (min_coords + max_coords) / 2,  # Center point
                             width=max_coords[0] - min_coords[0],
                             height=max_coords[1] - min_coords[1],
                             depth=max_coords[2] - min_coords[2],
                             color=color, alpha=alpha, wire=True, label=name)
        
        # For special field types, add additional visual elements
        if field_type == "barrier":
            # Add a more solid surface for barriers
            vertices = self._get_cube_vertices(
                (min_coords + max_coords) / 2,  # Center
                max_coords[0] - min_coords[0],
                max_coords[1] - min_coords[1],
                max_coords[2] - min_coords[2]
            )
            
            # Create faces for a barrier (more solid than generic field)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right
            ]
            
            collection = Poly3DCollection(faces, alpha=alpha*1.5, linewidth=1, edgecolor=color)
            collection.set_facecolor(color)
            ax.add_collection3d(collection)
    
    def _draw_interaction(self, ax, source_pos, target_pos, interaction):
        """Draw an interaction between entities"""
        interaction_type = interaction.get("properties", {}).get("type", "generic")
        
        if interaction_type == "gravity" or interaction_type == "force":
            # Draw a line with arrow between objects
            ax.quiver(
                source_pos[0], source_pos[1], source_pos[2],
                target_pos[0] - source_pos[0],
                target_pos[1] - source_pos[1],
                target_pos[2] - source_pos[2],
                color="cyan", alpha=0.7, arrow_length_ratio=0.1
            )
        
        elif interaction_type == "potential":
            # Draw a dashed line for potential interactions
            ax.plot(
                [source_pos[0], target_pos[0]],
                [source_pos[1], target_pos[1]],
                [source_pos[2], target_pos[2]],
                'y--', alpha=0.5, linewidth=1.5
            )
        
        else:
            # Generic interaction
            ax.plot(
                [source_pos[0], target_pos[0]],
                [source_pos[1], target_pos[1]],
                [source_pos[2], target_pos[2]],
                'w-', alpha=0.3, linewidth=1
            )
    
    def _draw_trajectory(self, ax, positions, name, fade=False):
        """Draw the trajectory of an object"""
        if len(positions) < 2:
            return
        
        color = self.color_palette.get("object")
        alpha = self.alpha["trajectory"]
        
        # Extract coordinates
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2] if positions.shape[1] > 2 else np.zeros(len(positions))
        
        if fade:
            # Create a gradient effect to show time progression
            colors = np.zeros((len(x), 4))
            for i in range(len(x)):
                # Fade from dark to bright
                intensity = i / max(1, len(x) - 1)
                rgba = to_rgba(color, alpha=alpha * (0.3 + 0.7 * intensity))
                colors[i] = rgba
            
            # Plot segments with changing colors
            for i in range(len(x) - 1):
                ax.plot(
                    x[i:i+2], y[i:i+2], z[i:i+2],
                    color=colors[i], linewidth=1.5
                )
        else:
            # Plot whole trajectory with consistent color
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=1.5, label=f"{name} trajectory")
    
    def _draw_sphere(self, ax, center, radius=1.0, color="blue", alpha=0.9, label=None):
        """Draw a sphere"""
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot sphere
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)
        
        # Add label if provided
        if label:
            ax.text(center[0], center[1], center[2] + radius, label, 
                    color='white', fontsize=9, ha='center')
    
    def _draw_cube(self, ax, center, size=1.0, color="red", alpha=0.9, wire=False, label=None):
        """Draw a cube centered at the given position"""
        # For a cube, height = width = depth = size
        self._draw_rectangle(ax, center, size, size, size, color, alpha, wire, label)
    
    def _draw_rectangle(self, ax, center, width, height, depth, color="green", alpha=0.9, wire=False, label=None):
        """Draw a rectangular prism"""
        # Get vertices
        vertices = self._get_cube_vertices(center, width, height, depth)
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right
        ]
        
        # Create collection
        collection = Poly3DCollection(faces, alpha=alpha, linewidth=1, edgecolor='black' if not wire else color)
        
        if wire:
            # For wireframe, set facecolor to transparent
            collection.set_facecolor((0, 0, 0, 0))
            collection.set_edgecolor(color)
        else:
            collection.set_facecolor(color)
        
        ax.add_collection3d(collection)
        
        # Add label if provided
        if label:
            ax.text(center[0], center[1], center[2] + height/2, label, 
                    color='white', fontsize=9, ha='center')
    
    def _get_cube_vertices(self, center, width, height, depth):
        """Calculate the vertices of a rectangular prism"""
        x, y, z = center
        w, h, d = width/2, height/2, depth/2
        
        vertices = [
            [x-w, y-h, z-d],  # 0: bottom left front
            [x+w, y-h, z-d],  # 1: bottom right front
            [x+w, y+h, z-d],  # 2: bottom right back
            [x-w, y+h, z-d],  # 3: bottom left back
            [x-w, y-h, z+d],  # 4: top left front
            [x+w, y-h, z+d],  # 5: top right front
            [x+w, y+h, z+d],  # 6: top right back
            [x-w, y+h, z+d]   # 7: top left back
        ]
        
        return vertices
    
    def _get_entity_position(self, entity_data, time_index):
        """Get the current position of an entity at the given time index"""
        entity_type = entity_data.get("type", "object")
        
        if entity_type == "object":
            positions = entity_data.get("positions", [])
            if len(positions) > 0 and 0 <= time_index < len(positions):
                pos = positions[time_index]
                # Ensure 3D position
                if len(pos) < 3:
                    pos = np.append(pos, [0] * (3 - len(pos)))
                return pos
        
        elif entity_type == "atom":
            expected_position = entity_data.get("expected_position", [])
            if len(expected_position) > 0 and 0 <= time_index < len(expected_position):
                if isinstance(expected_position, np.ndarray) and expected_position.ndim > 1:
                    # Multi-dimensional
                    pos = expected_position[time_index]
                    if len(pos) < 3:
                        pos = np.append(pos, [0] * (3 - len(pos)))
                    return pos
                else:
                    # 1D
                    return [expected_position[time_index], 0, 0]
        
        elif entity_type == "field":
            # Fields typically have a fixed region, take center
            field_props = entity_data.get("properties", {})
            field_region = field_props.get("region", [])
            
            if field_region:
                field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                return (field_min + field_max) / 2
        
        return None
    
    def _figure_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 encoded string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str 