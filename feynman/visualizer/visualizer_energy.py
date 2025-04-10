import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np

from .base_visualizer import BaseVisualizer
# Import simulator only if needed for constants, but prefer defining PE funcs here
# from ..simulator.classical_simulator import ClassicalSimulator 

class VisualizerEnergy(BaseVisualizer):
    """Visualizer for plotting energy components (KE, PE, Total) over time."""

    def __init__(self, sim_data):
        """Initialize and pre-calculate energy time series."""
        super().__init__(sim_data)
        # Pre-calculate energies
        self.ke_series, self.pe_series, self.total_energy_series = self._calculate_energies()
        # Create the initial figure
        self.initial_figure = self._create_energy_figure()

    def _create_layout(self):
        """Override layout for energy plots."""
        # Note: Energy plots are typically static over the whole time range.
        # The time slider from the base class might not be strictly necessary 
        # but can be used to indicate the current time point on the graph.
        return html.Div([
            html.H1("Feynman Simulation Visualizer - Energy Plot"),
            html.Hr(),
            html.Div(id='visualization-content', children=[
                # Display the graph component, figure will be set by main visualizer callback
                dcc.Graph(id='energy-graph') # Removed figure=self.initial_figure
            ]),
            html.Hr(),
            self._create_controls() # Include base controls (slider, play/pause)
        ])

    def _register_callbacks(self):
        """Register callbacks for energy plot updates (e.g., time marker)."""
        # Remove the time marker callback here, it's handled by FeynmanVisualizer
        super()._register_callbacks() # Register base callbacks

    # --- Potential Energy Helper Functions ---
    # These calculate potential energy between *pairs* of objects
    
    def _potential_gravity(self, pos1, pos2, props1, props2, interaction_props):
        G = interaction_props.get('G', 6.67430e-11)
        # Ensure mass is float, default to 1.0 if missing or invalid
        try: m1 = float(props1.get('mass', 1.0))
        except (ValueError, TypeError): m1 = 1.0
        try: m2 = float(props2.get('mass', 1.0))
        except (ValueError, TypeError): m2 = 1.0
        r_vec = np.array(pos2) - np.array(pos1)
        r = np.linalg.norm(r_vec)
        if r < 1e-10: return 0 # Avoid singularity, PE approaches -inf
        return -G * m1 * m2 / r

    def _potential_coulomb(self, pos1, pos2, props1, props2, interaction_props):
        k = interaction_props.get('k', 8.99e9)
        # Ensure charge is float, default to 0.0 if missing or invalid
        try: q1 = float(props1.get('charge', 0.0)) 
        except (ValueError, TypeError): q1 = 0.0
        try: q2 = float(props2.get('charge', 0.0))
        except (ValueError, TypeError): q2 = 0.0
        r_vec = np.array(pos2) - np.array(pos1)
        r = np.linalg.norm(r_vec)
        if r < 1e-10: 
            # PE is infinite at r=0, return large value or handle differently?
            # Returning 0 avoids breaking plot scale, but isn't physically accurate at the limit.
            return 0 
        return k * q1 * q2 / r

    def _potential_spring(self, pos1, pos2, props1, props2, interaction_props):
        # Ensure k and rest_length are floats
        try: k = float(interaction_props.get('k', 1.0))
        except (ValueError, TypeError): k = 1.0
        try: rest_length = float(interaction_props.get('rest_length', 0.0))
        except (ValueError, TypeError): rest_length = 0.0
        r_vec = np.array(pos2) - np.array(pos1)
        r = np.linalg.norm(r_vec)
        return 0.5 * k * (r - rest_length)**2
        
    # --- Main Energy Calculation --- 
    
    def _calculate_energies(self):
        """Calculate Kinetic, Potential, and Total energy for each time step."""
        if self.num_steps == 0:
            return np.array([]), np.array([]), np.array([])
            
        ke_total = np.zeros(self.num_steps)
        pe_total = np.zeros(self.num_steps)
        
        potential_calculators = {
            'gravity': self._potential_gravity,
            'coulomb': self._potential_coulomb,
            'spring': self._potential_spring
            # Add other potential types here if defined
        }

        # Calculate Kinetic Energy (sum over entities)
        for name, data in self.entities.items():
            ke_array = data.get('time_series', {}).get('kinetic_energy', None)
            # Ensure KE is a numpy array of the correct length
            if isinstance(ke_array, np.ndarray) and len(ke_array) == self.num_steps:
                ke_total += ke_array
            elif isinstance(ke_array, list) and len(ke_array) == self.num_steps:
                 try:
                      ke_total += np.array(ke_array, dtype=float)
                 except (ValueError, TypeError):
                      print(f"Warning: Could not convert kinetic energy list to float array for entity '{name}'. Skipping.")
            elif ke_array is not None:
                print(f"Warning: Kinetic energy data for entity '{name}' has incorrect length ({len(ke_array)} vs {self.num_steps}) or type. Skipping.")
            # If KE is None, we just don't add it, assuming it's 0 or not applicable.

        # Calculate Potential Energy (sum over interactions)
        for t in range(self.num_steps):
            pe_step = 0.0
            processed_pairs = set() # To avoid double counting A->B and B->A
            
            for interaction in self.interactions:
                source_name = interaction.get('source')
                target_name = interaction.get('target')
                props = interaction.get('properties', {})
                interaction_type = None
                interaction_params = {}
                
                # Determine interaction type and params (handling string or dict format)
                force_info = props.get("force")
                if isinstance(force_info, str):
                    interaction_type = force_info
                elif isinstance(force_info, dict) and "function" in force_info:
                    interaction_type = force_info["function"]
                    # Extract args into interaction_params (simple key-value for now)
                    for arg in force_info.get("args", []):
                         if isinstance(arg, dict) and "param" in arg and "value" in arg:
                             # Try to convert value to float if possible, otherwise keep as is
                             try: interaction_params[arg["param"]] = float(arg["value"])
                             except (ValueError, TypeError): interaction_params[arg["param"]] = arg["value"]
                
                # Check if interaction type has a potential calculator
                if not interaction_type or interaction_type not in potential_calculators:
                    continue
                
                # Ensure entities exist and have position data
                if source_name not in self.entities or target_name not in self.entities:
                    continue
                    
                source_data = self.entities[source_name]
                target_data = self.entities[target_name]
                source_pos_series = source_data.get('time_series', {}).get('positions')
                target_pos_series = target_data.get('time_series', {}).get('positions')

                # Validate position data format and length for the current timestep
                valid_pos = True
                pos1, pos2 = None, None
                if not isinstance(source_pos_series, np.ndarray) or source_pos_series.ndim != 2 or t >= source_pos_series.shape[0]: valid_pos = False
                if not isinstance(target_pos_series, np.ndarray) or target_pos_series.ndim != 2 or t >= target_pos_series.shape[0]: valid_pos = False
                
                if not valid_pos:
                    # Avoid printing warning repeatedly for the same issue
                    if t == 0: print(f"Warning: Invalid or missing position data for interaction {source_name}-{target_name} at step {t}. Skipping PE calculation for this interaction.")
                    continue 
                
                pos1 = source_pos_series[t]
                pos2 = target_pos_series[t]
                
                # Get entity initial properties (needed for mass, charge etc.)
                props1 = source_data.get('initial_properties', {})
                props2 = target_data.get('initial_properties', {})
                
                # Calculate potential energy for this interaction pair
                calculator = potential_calculators[interaction_type]
                # Merge interaction-specific params with general props
                # Interaction params override general props if names clash
                combined_interaction_props = {**props, **interaction_params}
                try:
                    pe_interaction = calculator(pos1, pos2, props1, props2, combined_interaction_props)
                    # Ensure PE is a valid float
                    if isinstance(pe_interaction, (int, float)) and np.isfinite(pe_interaction):
                        pe_step += pe_interaction
                    else:
                         if t==0: print(f"Warning: Non-finite PE calculated for interaction {tuple(sorted((source_name, target_name))) + (interaction_type,)}. Setting to 0.")
                except Exception as e:
                     # Avoid printing repeatedly
                     if t == 0: print(f"Error calculating potential energy for interaction {tuple(sorted((source_name, target_name))) + (interaction_type,)} at step {t}: {e}")

            pe_total[t] = pe_step
            
        total_energy = ke_total + pe_total
        return ke_total, pe_total, total_energy

    def _create_energy_figure(self):
        """Creates the Plotly figure for the energy plots."""
        fig = go.Figure()
        
        if self.num_steps == 0:
             fig.update_layout(title="No simulation data available for energy plot.")
             return fig

        # Check if energy series are valid before plotting
        valid_ke = isinstance(self.ke_series, np.ndarray) and len(self.ke_series) == self.num_steps
        valid_pe = isinstance(self.pe_series, np.ndarray) and len(self.pe_series) == self.num_steps
        valid_total = isinstance(self.total_energy_series, np.ndarray) and len(self.total_energy_series) == self.num_steps

        if valid_ke:
            fig.add_trace(go.Scatter(
                x=self.time_points,
                y=self.ke_series,
                mode='lines',
                name='Kinetic Energy (KE)',
                line=dict(color='red')
            ))
        if valid_pe:
            fig.add_trace(go.Scatter(
                x=self.time_points,
                y=self.pe_series,
                mode='lines',
                name='Potential Energy (PE)',
                line=dict(color='blue')
            ))
        if valid_total:
            fig.add_trace(go.Scatter(
                x=self.time_points,
                y=self.total_energy_series,
                mode='lines',
                name='Total Energy (KE + PE)',
                line=dict(color='green')
            ))

        if not valid_ke and not valid_pe and not valid_total:
             fig.update_layout(title="Could not calculate valid energy series for plotting.")
        else:
            fig.update_layout(
                title="Energy vs. Time",
                xaxis_title="Time",
                yaxis_title="Energy",
                showlegend=True,
                margin=dict(l=40, r=40, t=80, b=40)
            )
        return fig 