import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import base64
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import tempfile
import os
import json

class EnhancedVisualizer:
    """Enhanced 3D visualizer with sidebar charts"""
    
    def __init__(self):
        # Default color palette for different entity types
        self.color_palette = {
            'object': '#1f77b4',  # blue
            'atom': '#d62728',    # red
            'field': '#ff7f0e',   # orange
            'default': '#7f7f7f'  # gray
        }
        
        # Default marker sizes
        self.marker_sizes = {
            'object': 10,
            'atom': 12,
            'field': 8
        }
        
        # Default opacity values
        self.opacities = {
            'solid': 0.9,
            'field': 0.3,
            'trajectory': 0.7
        }
    
    def create_visualization(self, 
                           entities: Dict[str, Dict[str, Any]], 
                           time_points: np.ndarray,
                           interactions: Optional[List[Dict[str, Any]]] = None,
                           selected_entity: Optional[str] = None) -> str:
        """
        Create an enhanced 3D visualization with sidebar charts
        
        Args:
            entities: Dictionary of entities data
            time_points: Array of time points
            interactions: List of interaction data
            selected_entity: Name of the entity to focus on
            
        Returns:
            HTML string containing the visualization
        """
        # Create a figure with subplots - 3D view and side charts
        fig = plt.figure(figsize=(15, 10))
        
        # Define grid for the layout
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
        
        # Add 3D subplot
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_energy = fig.add_subplot(gs[0, 1])
        ax_position = fig.add_subplot(gs[1, 1])
        
        # Calculate scene bounds
        min_coords, max_coords = self._calculate_scene_bounds(entities)
        
        # Add entities to the 3D scene
        has_entities = False
        for entity_name, entity_data in entities.items():
            entity_type = entity_data.get("type", "object")
            
            # Highlight the selected entity
            is_selected = (selected_entity == entity_name) if selected_entity else False
            
            if entity_type == "object":
                self._add_object_to_scene(ax_3d, entity_name, entity_data, is_selected)
                has_entities = True
            elif entity_type == "atom":
                self._add_atom_to_scene(ax_3d, entity_name, entity_data, is_selected)
                has_entities = True
            elif entity_type == "field":
                self._add_field_to_scene(ax_3d, entity_name, entity_data, is_selected, min_coords, max_coords)
                has_entities = True
        
        # Add interactions if provided
        if interactions:
            self._add_interactions_to_scene(ax_3d, entities, interactions)
        
        # Add sidebar charts for the selected entity
        if selected_entity and selected_entity in entities:
            entity_data = entities[selected_entity]
            self._add_energy_chart(ax_energy, entity_data, time_points)
            self._add_position_chart(ax_position, entity_data, time_points)
        
        # Set axes limits with padding
        padding = 0.1 * (max_coords - min_coords)
        ax_3d.set_xlim(min_coords[0] - padding[0], max_coords[0] + padding[0])
        ax_3d.set_ylim(min_coords[1] - padding[1], max_coords[1] + padding[1])
        ax_3d.set_zlim(min_coords[2] - padding[2], max_coords[2] + padding[2])
        
        # Set labels and title
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"3D Physics Simulation")
        
        # Set equal aspect ratio
        ax_3d.set_box_aspect([1, 1, 1])
        
        # Apply a better view angle
        ax_3d.view_init(elev=20, azim=30)
        
        # Add a legend
        ax_3d.legend()
        
        # Add main title
        fig.suptitle(f"Enhanced Physics Visualization", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # Create HTML with the image and interactive elements
        html = self._create_html_interface(img_base64, entities, selected_entity)
        
        return html
    
    def _create_html_interface(self, image_base64, entities, selected_entity):
        """Create an HTML interface with the visualization and controls"""
        # Create entity selection dropdown options
        entity_options = ""
        for name in entities.keys():
            selected = "selected" if name == selected_entity else ""
            entity_options += f'<option value="{name}" {selected}>{name}</option>'
        
        # Create the entity info section based on selected entity
        entity_info = ""
        if selected_entity and selected_entity in entities:
            entity = entities[selected_entity]
            entity_type = entity.get("type", "object")
            properties = entity.get("properties", {})
            
            # Display different properties based on entity type
            entity_info += f'<h3>{selected_entity} ({entity_type})</h3>'
            entity_info += '<div class="property-list">'
            
            if entity_type == "object":
                # Show object properties
                if "positions" in entity and len(entity["positions"]) > 0:
                    final_pos = entity["positions"][-1]
                    entity_info += f'<div class="property"><span>Position:</span> {self._format_vector(final_pos)}</div>'
                
                if "velocities" in entity and len(entity["velocities"]) > 0:
                    final_vel = entity["velocities"][-1]
                    entity_info += f'<div class="property"><span>Velocity:</span> {self._format_vector(final_vel)}</div>'
                
                if "mass" in properties:
                    entity_info += f'<div class="property"><span>Mass:</span> {properties["mass"]}</div>'
                
                if "kinetic_energy" in entity and len(entity["kinetic_energy"]) > 0:
                    final_ke = entity["kinetic_energy"][-1]
                    entity_info += f'<div class="property"><span>Kinetic Energy:</span> {final_ke:.2f}</div>'
            
            elif entity_type == "atom":
                # Show quantum particle properties
                if "expected_position" in entity and len(entity["expected_position"]) > 0:
                    final_pos = entity["expected_position"][-1]
                    entity_info += f'<div class="property"><span>Expected Position:</span> {self._format_vector(final_pos)}</div>'
                
                if "mass" in properties:
                    entity_info += f'<div class="property"><span>Mass:</span> {properties["mass"]}</div>'
            
            elif entity_type == "field":
                # Show field properties
                field_type = properties.get("type", "generic")
                entity_info += f'<div class="property"><span>Field Type:</span> {field_type}</div>'
                
                if "strength" in properties:
                    entity_info += f'<div class="property"><span>Strength:</span> {properties["strength"]}</div>'
                
                if "region" in properties:
                    region = properties["region"]
                    entity_info += f'<div class="property"><span>Region:</span> {self._format_region(region)}</div>'
            
            entity_info += '</div>'
        
        # Create the HTML interface with responsive design
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Physics Visualization</title>
            <style>
                /* Modern, responsive styling */
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #181818;
                    color: #e0e0e0;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    display: flex;
                    flex-direction: column;
                }}
                h1, h2, h3 {{
                    color: #f0f0f0;
                }}
                .main-title {{
                    text-align: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #444;
                }}
                .visualization-container {{
                    display: flex;
                    flex-direction: row;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .main-view {{
                    flex: 3;
                    min-width: 300px;
                    background-color: #222;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                }}
                .main-view img {{
                    width: 100%;
                    height: auto;
                    display: block;
                }}
                .sidebar {{
                    flex: 1;
                    min-width: 250px;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }}
                .control-panel, .entity-info {{
                    background-color: #2a2a2a;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                }}
                .control-panel h3, .entity-info h3 {{
                    margin-top: 0;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #444;
                }}
                select, button {{
                    width: 100%;
                    padding: 8px;
                    margin: 5px 0;
                    border-radius: 4px;
                    border: 1px solid #444;
                    background-color: #333;
                    color: white;
                }}
                button {{
                    background-color: #2a5885;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }}
                button:hover {{
                    background-color: #3a6d9e;
                }}
                input[type="range"] {{
                    width: 100%;
                    margin: 10px 0;
                }}
                .property-list {{
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }}
                .property {{
                    display: flex;
                    justify-content: space-between;
                }}
                .property span {{
                    font-weight: bold;
                    color: #aaa;
                }}
                @media (max-width: 768px) {{
                    .visualization-container {{
                        flex-direction: column;
                    }}
                    .main-view, .sidebar {{
                        width: 100%;
                    }}
                }}
                
                /* Tabs for different views */
                .tabs {{
                    display: flex;
                    margin-bottom: 10px;
                }}
                .tab {{
                    padding: 8px 15px;
                    background-color: #333;
                    border-radius: 4px 4px 0 0;
                    cursor: pointer;
                    margin-right: 2px;
                }}
                .tab.active {{
                    background-color: #2a5885;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="main-title">Enhanced Physics Visualization</h1>
                
                <div class="visualization-container">
                    <div class="main-view">
                        <div class="tabs">
                            <div class="tab active" onclick="switchView('static')">Static View</div>
                            <div class="tab" onclick="switchView('animation')">Animation</div>
                        </div>
                        <div id="static-view">
                            <img src="data:image/png;base64,{image_base64}" alt="3D Visualization">
                        </div>
                        <div id="animation-view" style="display: none;">
                            <p>Animation would appear here in a dynamic implementation.</p>
                        </div>
                    </div>
                    
                    <div class="sidebar">
                        <div class="control-panel">
                            <h3>Controls</h3>
                            <label for="entity-select">Select Entity:</label>
                            <select id="entity-select" onchange="updateEntity(this.value)">
                                {entity_options}
                            </select>
                            
                            <div>
                                <label for="time-slider">Simulation Time:</label>
                                <input type="range" id="time-slider" min="0" max="100" value="100">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>0.0</span>
                                    <span id="time-value">10.0</span>
                                </div>
                            </div>
                            
                            <div style="margin-top: 15px;">
                                <button onclick="resetView()">Reset View</button>
                            </div>
                        </div>
                        
                        <div class="entity-info">
                            <h3>Entity Information</h3>
                            <div id="entity-details">
                                {entity_info}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // In a dynamic implementation, these functions would update the view
                function updateEntity(entityName) {{
                    // This would reload the page with the new entity in a full implementation
                    console.log("Selected entity: " + entityName);
                    // window.location.href = window.location.pathname + "?entity=" + entityName;
                }}
                
                function switchView(viewType) {{
                    if (viewType === 'static') {{
                        document.getElementById('static-view').style.display = 'block';
                        document.getElementById('animation-view').style.display = 'none';
                        document.querySelectorAll('.tab')[0].classList.add('active');
                        document.querySelectorAll('.tab')[1].classList.remove('active');
                    }} else {{
                        document.getElementById('static-view').style.display = 'none';
                        document.getElementById('animation-view').style.display = 'block';
                        document.querySelectorAll('.tab')[0].classList.remove('active');
                        document.querySelectorAll('.tab')[1].classList.add('active');
                    }}
                }}
                
                function resetView() {{
                    // This would reset the view in a full implementation
                    console.log("Resetting view");
                }}
                
                // Update time value display when slider changes
                document.getElementById('time-slider').addEventListener('input', function() {{
                    const value = this.value / 10;
                    document.getElementById('time-value').textContent = value.toFixed(1);
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _format_vector(self, vector):
        """Format a vector for display"""
        if isinstance(vector, (list, np.ndarray)):
            if len(vector) == 1:
                return f"{vector[0]:.2f}"
            elif len(vector) == 2:
                return f"[{vector[0]:.2f}, {vector[1]:.2f}]"
            elif len(vector) == 3:
                return f"[{vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}]"
        return str(vector)
    
    def _format_region(self, region):
        """Format a region for display"""
        if isinstance(region, (list, tuple)) and len(region) >= 2:
            return f"{self._format_vector(region[0])} to {self._format_vector(region[1])}"
        return str(region)
    
    def _calculate_scene_bounds(self, entities):
        """Calculate min and max coordinates for the scene"""
        min_coords = np.array([float('inf'), float('inf'), float('inf')])
        max_coords = np.array([float('-inf'), float('-inf'), float('-inf')])
        has_entities = False
        
        for entity_name, entity_data in entities.items():
            entity_type = entity_data.get("type", "object")
            
            if entity_type == "object":
                positions = entity_data.get("positions", [])
                if len(positions) > 0:
                    positions_array = np.array(positions)
                    for dim in range(min(3, positions_array.shape[1])):
                        min_coords[dim] = min(min_coords[dim], np.min(positions_array[:, dim]))
                        max_coords[dim] = max(max_coords[dim], np.max(positions_array[:, dim]))
                    has_entities = True
            
            elif entity_type == "atom":
                expected_position = entity_data.get("expected_position", [])
                if len(expected_position) > 0:
                    # Handle 1D or multi-dimensional positions
                    ep_array = np.array(expected_position)
                    if ep_array.ndim > 1:
                        dim = min(3, ep_array.shape[1])
                        for d in range(dim):
                            min_coords[d] = min(min_coords[d], np.min(ep_array[:, d]) - 2)
                            max_coords[d] = max(max_coords[d], np.max(ep_array[:, d]) + 2)
                    else:
                        min_coords[0] = min(min_coords[0], np.min(ep_array) - 2)
                        max_coords[0] = max(max_coords[0], np.max(ep_array) + 2)
                        # Set default bounds for other dimensions
                        min_coords[1:] = np.minimum(min_coords[1:], [-5, -5])
                        max_coords[1:] = np.maximum(max_coords[1:], [5, 5])
                    has_entities = True
            
            elif entity_type == "field":
                field_props = entity_data.get("properties", {})
                field_region = field_props.get("region", [])
                
                if field_region:
                    field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                    field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                    
                    min_coords = np.minimum(min_coords, field_min)
                    max_coords = np.maximum(max_coords, field_max)
                    has_entities = True
        
        # Set default bounds if no entities were found
        if not has_entities or min_coords[0] == float('inf'):
            min_coords = np.array([-10, -10, -10])
        if not has_entities or max_coords[0] == float('-inf'):
            max_coords = np.array([10, 10, 10])
        
        # Check for any remaining infinities and replace with defaults
        for i in range(3):
            if not np.isfinite(min_coords[i]):
                min_coords[i] = -10
            if not np.isfinite(max_coords[i]):
                max_coords[i] = 10
        
        # Ensure the bounds are different
        for i in range(3):
            if min_coords[i] == max_coords[i]:
                min_coords[i] -= 5
                max_coords[i] += 5
        
        return min_coords, max_coords
    
    def _add_object_to_scene(self, ax, entity_name, entity_data, is_selected):
        """Add a classical object to the 3D scene"""
        positions = entity_data.get("positions", [])
        if len(positions) == 0:
            return
        
        # Get object properties
        props = entity_data.get("properties", {})
        shape = props.get("shape", "sphere")
        color = props.get("color", self.color_palette.get("object"))
        
        # Get positions over time
        pos_array = np.array(positions)
        
        # Add trajectory
        if pos_array.shape[1] >= 2:
            ax.plot(pos_array[:, 0], pos_array[:, 1], 
                   pos_array[:, 2] if pos_array.shape[1] > 2 else np.zeros(len(pos_array)),
                   '-', color=color, alpha=self.opacities["trajectory"], 
                   linewidth=2 if is_selected else 1,
                   label=f"{entity_name} path")
        
        # Get the final position for static visualization
        final_pos = positions[-1]
        if len(final_pos) < 3:
            final_pos = np.append(final_pos, [0] * (3 - len(final_pos)))
        
        # Draw the object based on shape
        marker_size = self.marker_sizes["object"] * 2 if is_selected else self.marker_sizes["object"]
        
        if shape == "sphere":
            ax.scatter(final_pos[0], final_pos[1], final_pos[2], 
                      s=marker_size**2, color=color, edgecolors='white' if is_selected else 'none',
                      label=entity_name)
        elif shape == "cube" or shape == "block":
            size = float(props.get("size", 1.0))
            self._add_cube(ax, final_pos, size, size, size, color, entity_name, is_selected)
        elif shape == "rectangle":
            width = float(props.get("width", 1.0))
            height = float(props.get("height", 1.0))
            depth = float(props.get("depth", 1.0))
            self._add_cube(ax, final_pos, width, height, depth, color, entity_name, is_selected)
    
    def _add_atom_to_scene(self, ax, entity_name, entity_data, is_selected):
        """Add a quantum particle to the 3D scene"""
        expected_position = entity_data.get("expected_position", [])
        if len(expected_position) == 0:
            return
        
        # Get the color
        props = entity_data.get("properties", {})
        color = props.get("color", self.color_palette.get("atom"))
        
        # Convert to numpy array if not already
        ep_array = np.array(expected_position)
        
        # Add trajectory based on dimensionality
        if ep_array.ndim > 1:
            # Multi-dimensional positions
            ax.plot(ep_array[:, 0], 
                   ep_array[:, 1] if ep_array.shape[1] > 1 else np.zeros(len(ep_array)),
                   ep_array[:, 2] if ep_array.shape[1] > 2 else np.zeros(len(ep_array)),
                   '--', color=color, alpha=self.opacities["trajectory"], 
                   linewidth=2 if is_selected else 1,
                   label=f"{entity_name} path")
            
            # Get the final position
            final_pos = ep_array[-1]
            if len(final_pos) < 3:
                final_pos = np.append(final_pos, [0] * (3 - len(final_pos)))
        else:
            # 1D positions - plot along x-axis
            ax.plot(ep_array, np.zeros(len(ep_array)), np.zeros(len(ep_array)),
                   '--', color=color, alpha=self.opacities["trajectory"], 
                   linewidth=2 if is_selected else 1,
                   label=f"{entity_name} path")
            
            # Get the final position
            final_pos = np.array([ep_array[-1], 0, 0])
        
        # Draw the quantum particle with a special marker
        marker_size = self.marker_sizes["atom"] * 2 if is_selected else self.marker_sizes["atom"]
        ax.scatter(final_pos[0], final_pos[1], final_pos[2], 
                  s=marker_size**2, color=color, marker='*', 
                  edgecolors='white' if is_selected else 'none',
                  label=entity_name)
        
        # Add probability density if available
        if "probability_density" in entity_data and "grid" in entity_data:
            prob_density = entity_data["probability_density"][-1]  # Use the final state
            grid = entity_data["grid"]
            
            # Scale for visibility
            max_pd = np.max(prob_density)
            if max_pd > 0:
                scaled_pd = prob_density / max_pd * 5
                
                # Plot 1D probability density along x-axis
                if ep_array.ndim == 1:
                    ax.plot(grid, np.zeros_like(grid), scaled_pd,
                           '-', color=color, alpha=0.5, linewidth=3,
                           label=f"{entity_name} probability")
    
    def _add_field_to_scene(self, ax, entity_name, entity_data, is_selected, min_coords, max_coords):
        """Add a field to the 3D scene"""
        props = entity_data.get("properties", {})
        field_type = props.get("type", "generic")
        color = props.get("color", self.color_palette.get("field"))
        
        # Get field region
        field_region = props.get("region", [])
        if not field_region:
            # If no region specified, use a default volume in the scene
            field_min = min_coords + 0.25 * (max_coords - min_coords)
            field_max = min_coords + 0.75 * (max_coords - min_coords)
        else:
            field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
            field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
        
        # Get center and dimensions
        center = (field_min + field_max) / 2
        width = field_max[0] - field_min[0]
        height = field_max[1] - field_min[1]
        depth = field_max[2] - field_min[2]
        
        # Add field representation
        opacity = self.opacities["field"] * 1.3 if is_selected else self.opacities["field"]
        
        if field_type == "barrier":
            # Draw a solid box for barriers
            self._add_cube(ax, center, width, height, depth, color, entity_name, is_selected, opacity)
        else:
            # Draw a wireframe for other fields
            self._add_cube(ax, center, width, height, depth, color, entity_name, is_selected, opacity, wireframe=True)
        
        # Add text for field name
        ax.text(center[0], center[1], center[2], entity_name, 
               color='white', fontsize=10 if is_selected else 8,
               horizontalalignment='center', verticalalignment='center')
    
    def _add_interactions_to_scene(self, ax, entities, interactions):
        """Add interactions between entities to the 3D scene"""
        for interaction in interactions:
            source = interaction.get("source", "")
            target = interaction.get("target", "")
            
            if source in entities and target in entities:
                source_pos = self._get_entity_final_position(entities[source])
                target_pos = self._get_entity_final_position(entities[target])
                
                if source_pos is not None and target_pos is not None:
                    interaction_type = interaction.get("properties", {}).get("type", "generic")
                    
                    if interaction_type == "gravity" or interaction_type == "force":
                        # Draw an arrow for forces
                        ax.plot([source_pos[0], target_pos[0]], 
                               [source_pos[1], target_pos[1]], 
                               [source_pos[2], target_pos[2]],
                               'c-', linewidth=2, label=f"{source} → {target}")
                        
                        # Add arrow head
                        mid_pos = (source_pos + target_pos) / 2
                        ax.scatter(mid_pos[0], mid_pos[1], mid_pos[2], 
                                  color='cyan', marker='>')
                    
                    elif interaction_type == "potential":
                        # Draw a dashed line for potentials
                        ax.plot([source_pos[0], target_pos[0]], 
                               [source_pos[1], target_pos[1]], 
                               [source_pos[2], target_pos[2]],
                               'y--', linewidth=1.5, label=f"{source} → {target}")
    
    def _add_energy_chart(self, ax, entity_data, time_points):
        """Add energy chart to the figure"""
        if entity_data.get("type") == "object":
            kinetic_energy = entity_data.get("kinetic_energy", [])
            if len(kinetic_energy) > 0:
                ax.plot(time_points, kinetic_energy, 'r-', linewidth=2, label='Kinetic Energy')
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy')
                ax.set_title('Energy')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        elif entity_data.get("type") == "atom":
            # For quantum particles, we could plot expected energy or another relevant metric
            pass
    
    def _add_position_chart(self, ax, entity_data, time_points):
        """Add position chart to the figure"""
        if entity_data.get("type") == "object":
            positions = entity_data.get("positions", [])
            if len(positions) > 0:
                pos_array = np.array(positions)
                
                # Extract x, y, z positions over time
                x_pos = pos_array[:, 0]
                if pos_array.shape[1] > 1:
                    y_pos = pos_array[:, 1]
                    ax.plot(time_points, y_pos, 'g-', linewidth=2, label='Y')
                
                if pos_array.shape[1] > 2:
                    z_pos = pos_array[:, 2]
                    ax.plot(time_points, z_pos, 'b-', linewidth=2, label='Z')
                
                ax.plot(time_points, x_pos, 'r-', linewidth=2, label='X')
                ax.set_xlabel('Time')
                ax.set_ylabel('Position')
                ax.set_title('Position')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        elif entity_data.get("type") == "atom":
            expected_position = entity_data.get("expected_position", [])
            if len(expected_position) > 0:
                ep_array = np.array(expected_position)
                
                # For 1D quantum particle
                if ep_array.ndim == 1:
                    ax.plot(time_points, ep_array, 'm-', linewidth=2, label='Expected Position')
                else:
                    # For multi-dimensional quantum particle
                    x_pos = ep_array[:, 0]
                    ax.plot(time_points, x_pos, 'm-', linewidth=2, label='Expected Position (X)')
                    
                    if ep_array.shape[1] > 1:
                        y_pos = ep_array[:, 1]
                        ax.plot(time_points, y_pos, 'c-', linewidth=2, label='Expected Position (Y)')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Position')
                ax.set_title('Expected Position')
                ax.grid(True, alpha=0.3)
                ax.legend()
    
    def _get_entity_final_position(self, entity_data):
        """Get the final position of an entity"""
        entity_type = entity_data.get("type", "object")
        
        if entity_type == "object":
            positions = entity_data.get("positions", [])
            if len(positions) > 0:
                pos = positions[-1]
                # Ensure 3D position
                if len(pos) < 3:
                    pos = np.append(pos, [0] * (3 - len(pos)))
                return np.array(pos)
        
        elif entity_type == "atom":
            expected_position = entity_data.get("expected_position", [])
            if len(expected_position) > 0:
                ep_array = np.array(expected_position)
                if ep_array.ndim > 1:
                    pos = ep_array[-1]
                    if len(pos) < 3:
                        pos = np.append(pos, [0] * (3 - len(pos)))
                    return np.array(pos)
                else:
                    return np.array([ep_array[-1], 0, 0])
        
        elif entity_type == "field":
            field_props = entity_data.get("properties", {})
            field_region = field_props.get("region", [])
            
            if field_region:
                field_min = np.array(field_region[0]) if isinstance(field_region[0], (list, tuple)) else np.array([field_region[0], -5, -5])
                field_max = np.array(field_region[1]) if isinstance(field_region[1], (list, tuple)) else np.array([field_region[1], 5, 5])
                return (field_min + field_max) / 2
        
        return None
    
    def _add_cube(self, ax, center, width, height, depth, color, name, is_selected, opacity=0.7, wireframe=False):
        """Add a cube or rectangular prism to the 3D scene"""
        # Create vertices
        x, y, z = center
        w, h, d = width/2, height/2, depth/2
        
        vertices = np.array([
            [x-w, y-h, z-d],  # 0: bottom left front
            [x+w, y-h, z-d],  # 1: bottom right front
            [x+w, y+h, z-d],  # 2: bottom right back
            [x-w, y+h, z-d],  # 3: bottom left back
            [x-w, y-h, z+d],  # 4: top left front
            [x+w, y-h, z+d],  # 5: top right front
            [x+w, y+h, z+d],  # 6: top right back
            [x-w, y+h, z+d]   # 7: top left back
        ])
        
        # Define the 6 faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        # Create the 3D polygons
        rgba_color = to_rgba(color, opacity)
        
        if wireframe:
            # Draw wireframe
            for face in faces:
                face_array = np.array(face)
                face_array = np.vstack([face_array, face_array[0]])  # Close the loop
                ax.plot(face_array[:, 0], face_array[:, 1], face_array[:, 2], 
                       color=color, alpha=opacity, linestyle='-')
        else:
            # Draw solid faces
            poly3d = Poly3DCollection(faces, alpha=opacity, linewidths=1 if is_selected else 0.5)
            poly3d.set_facecolor(rgba_color)
            if is_selected:
                poly3d.set_edgecolor('white')
            else:
                poly3d.set_edgecolor(rgba_color)
            
            ax.add_collection3d(poly3d) 