import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np

from .base_visualizer import BaseVisualizer
from .visualizer_2d import Visualizer2D
from .visualizer_3d import Visualizer3D
from .visualizer_phase_space import VisualizerPhaseSpace
from .visualizer_energy import VisualizerEnergy

class FeynmanVisualizer(BaseVisualizer):
    """Main visualizer integrating different plot types using Tabs."""

    def __init__(self, sim_data):
        """Initialize the main visualizer, detect data type, prepare specific data."""
        super().__init__(sim_data)
        
        # --- Detect Data Type and Prepare --- 
        self.has_classical_data = False
        self.has_quantum_data = False
        self.quantum_dims = None
        self.quantum_grid_info = None
        self.quantum_entity_name = None # Assume one quantum entity for now
        self.available_exp_values = []
        self.available_eigenstates = None
        
        # Check entities for classical or quantum types
        for name, data in self.entities.items():
             entity_type = data.get('initial_properties', {}).get('type') # Check initial props
             if not entity_type:
                 # Fallback: check time_series data if initial props lack type
                 entity_type = data.get('time_series', {}).get('type')
                 
             if entity_type == 'object':
                 self.has_classical_data = True
             elif entity_type == 'atom':
                 self.has_quantum_data = True
                 # Assume first atom found provides the relevant quantum info
                 if self.quantum_entity_name is None:
                     self.quantum_entity_name = name
                     ts_data = data.get('time_series', {})
                     self.quantum_dims = ts_data.get('dimensions')
                     self.quantum_grid_info = ts_data.get('grid_info')
                     # Identify available expectation values
                     for key in ts_data.keys():
                         if key.startswith('expected_'):
                             self.available_exp_values.append(key)
                     # Check for eigenstates
                     if 'eigenstates' in ts_data:
                          self.available_eigenstates = ts_data['eigenstates']
                          # Store number of states for dropdown
                          self.num_eigenstates = len(self.available_eigenstates.get('eigenvalues', []))

        # Instantiate classical helper visualizers if needed (for figure logic reuse)
        # We only need them if classical data is present
        if self.has_classical_data:
            self.viz_2d = Visualizer2D(sim_data)
            self.viz_3d = Visualizer3D(sim_data)
            self.viz_phase = VisualizerPhaseSpace(sim_data)
            self.viz_energy = VisualizerEnergy(sim_data)
        else:
             # Avoid errors if trying to access these later
             self.viz_2d = None
             self.viz_3d = None
             self.viz_phase = None
             self.viz_energy = None

        # --- Final Setup: Create Layout and Register Callbacks ---
        # This must happen *after* super().__init__() and data detection
        self.app.layout = self._create_layout() # Call layout creation
        self._register_callbacks()

    def _create_layout(self):
        """Override layout to use Tabs for switching visualizations, adapting to data type."""
        tabs_list = []
        default_tab = 'tab-info' # Default if no specific data
        
        # Add Classical Tabs if data exists
        if self.has_classical_data:
             tabs_list.extend([
                 dcc.Tab(label='2D Animation', value='tab-2d'),
                 dcc.Tab(label='3D Animation', value='tab-3d'),
                 dcc.Tab(label='Phase Space', value='tab-phase'),
                 dcc.Tab(label='Classical Energy', value='tab-energy'),
             ])
             if default_tab == 'tab-info': default_tab = 'tab-2d'
             
        # Add Quantum Tabs if data exists
        if self.has_quantum_data:
            tabs_list.extend([
                dcc.Tab(label='Probability Density', value='tab-prob'),
                dcc.Tab(label='Expectation Values', value='tab-expval'),
            ])
            if self.available_eigenstates:
                 tabs_list.append(dcc.Tab(label='Eigenstates', value='tab-eigen'))
            if default_tab == 'tab-info': default_tab = 'tab-prob'
            
        # Fallback Info Tab
        if not tabs_list:
            tabs_list.append(dcc.Tab(label='Info', value='tab-info'))

        tabs = dcc.Tabs(id='visualization-tabs', value=default_tab, children=tabs_list)

        return html.Div([
            html.H1("Feynman Simulation Visualizer"),
            html.Hr(),
            tabs,
            # Content area for the selected tab
            html.Div(id='tab-content'),
            html.Hr(),
            # Common controls below the tabs
            self._create_controls()
        ])

    def _create_controls(self):
         """Override controls to include elements needed by specific tabs."""
         controls_list = []
         
         # --- Phase Space Controls (Classical) ---
         if self.has_classical_data:
             axis_options = [
                {'label': 'X Position (x)', 'value': 'pos_0'},
                {'label': 'Y Position (y)', 'value': 'pos_1'},
                {'label': 'Z Position (z)', 'value': 'pos_2'},
                {'label': 'X Velocity (vx)', 'value': 'vel_0'},
                {'label': 'Y Velocity (vy)', 'value': 'vel_1'},
                {'label': 'Z Velocity (vz)', 'value': 'vel_2'}
             ]
             default_entity = self.entity_names[0] if self.entity_names else None
             phase_space_dropdowns = html.Div([
                 html.Div([ html.Label("Select Entity:"), dcc.Dropdown(id='phase-space-entity-selector', options=[{'label': name, 'value': name} for name in self.entity_names], value=default_entity) ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                 html.Div([ html.Label("Select X-Axis:"), dcc.Dropdown(id='phase-space-x-axis-selector', options=axis_options, value='pos_0') ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
                 html.Div([ html.Label("Select Y-Axis:"), dcc.Dropdown(id='phase-space-y-axis-selector', options=axis_options, value='vel_0') ], style={'width': '30%', 'display': 'inline-block'}),
             ])
             controls_list.append(html.Div(id='phase-space-controls-container', children=[phase_space_dropdowns], style={'display': 'none', 'marginBottom': '20px'}))

         # --- Expectation Value Controls (Quantum) ---
         if self.has_quantum_data and self.available_exp_values:
            exp_val_options = [{'label': key.replace('expected_','').replace('_',' ').title(), 'value': key} for key in self.available_exp_values]
            exp_val_dropdown = html.Div([
                html.Label("Select Expectation Value(s):"),
                dcc.Dropdown(
                    id='exp-val-selector',
                    options=exp_val_options,
                    value=[exp_val_options[0]['value']] if exp_val_options else [], # Default to first
                    multi=True
                )
            ], style={'width': '90%', 'margin': 'auto'})
            controls_list.append(html.Div(id='exp-val-controls-container', children=[exp_val_dropdown], style={'display': 'none', 'marginBottom': '20px'}))

         # --- Eigenstate Controls (Quantum) ---
         if self.has_quantum_data and self.available_eigenstates:
            eigen_options = [{'label': f'State {i} (E={self.available_eigenstates["eigenvalues"][i]:.3e})', 'value': i} 
                             for i in range(self.num_eigenstates)]
            eigen_dropdown = html.Div([
                 html.Label("Select Eigenstate:"),
                 dcc.Dropdown(
                     id='eigenstate-selector',
                     options=eigen_options,
                     value=0, # Default to ground state
                     clearable=False
                 )
            ], style={'width': '90%', 'margin': 'auto'})
            controls_list.append(html.Div(id='eigenstate-controls-container', children=[eigen_dropdown], style={'display': 'none', 'marginBottom': '20px'}))

         # --- Base Time Controls ---
         # Always add time controls if num_steps > 1
         if self.num_steps > 1:
             time_controls = super()._create_controls()
             controls_list.append(time_controls)
         elif self.has_classical_data or self.has_quantum_data:
             # Show message if simulation has only one step but data exists
             controls_list.append(html.Div(html.P("Simulation has only one time step. Animation controls disabled."))) 

         # Combine all controls
         return html.Div(controls_list)

    def _register_callbacks(self):
        """Register callbacks for the main visualizer, including quantum plots."""
        # Register base callbacks for play/pause and slider animation
        if self.num_steps > 1:
            super()._register_callbacks()

        # Callback to render tab content based on selected tab
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('visualization-tabs', 'value')
        )
        def render_tab_content(tab_value):
            # Classical Tabs
            if tab_value == 'tab-2d': return dcc.Graph(id='2d-animation-graph', style={'height': '600px'})
            if tab_value == 'tab-3d': return dcc.Graph(id='3d-animation-graph', style={'height': '700px'})
            if tab_value == 'tab-phase': return dcc.Graph(id='phase-space-graph', style={'height': '600px'})
            if tab_value == 'tab-energy': 
                 return dcc.Graph(id='energy-graph', figure=self.viz_energy.initial_figure if self.viz_energy else go.Figure())
            # Quantum Tabs
            if tab_value == 'tab-prob': return dcc.Graph(id='prob-density-graph', style={'height': '600px'})
            if tab_value == 'tab-expval': return dcc.Graph(id='exp-val-graph', style={'height': '500px'})
            if tab_value == 'tab-eigen': return dcc.Graph(id='eigenstate-graph', style={'height': '600px'})
            # Default/Info Tab
            if tab_value == 'tab-info': return html.P("No simulation data loaded or available to visualize.")
            return html.P(f"Tab '{tab_value}' selected.") # Fallback

        # --- Callbacks to toggle visibility of controls ---
        @self.app.callback(
             Output('phase-space-controls-container', 'style', allow_duplicate=True),
             Input('visualization-tabs', 'value'),
             prevent_initial_call=True
        )
        def toggle_phase_space_controls(tab_value):
             # Check if the container exists in the layout first
             if not self.has_classical_data: return dash.no_update
             return {'display': 'block'} if tab_value == 'tab-phase' else {'display': 'none'}

        @self.app.callback(
             Output('exp-val-controls-container', 'style', allow_duplicate=True),
             Input('visualization-tabs', 'value'),
             prevent_initial_call=True
        )
        def toggle_exp_val_controls(tab_value):
             if not self.has_quantum_data: return dash.no_update
             return {'display': 'block'} if tab_value == 'tab-expval' else {'display': 'none'}
             
        @self.app.callback(
             Output('eigenstate-controls-container', 'style', allow_duplicate=True),
             Input('visualization-tabs', 'value'),
             prevent_initial_call=True
        )
        def toggle_eigenstate_controls(tab_value):
             if not self.has_quantum_data or not self.available_eigenstates: return dash.no_update
             return {'display': 'block'} if tab_value == 'tab-eigen' else {'display': 'none'}

        # --- Callbacks to Update Figures --- 
        
        # Classical Plot Updates (keep existing logic, guarded by checks)
        if self.has_classical_data:
            self._register_classical_callbacks()
            
        # Quantum Plot Updates
        if self.has_quantum_data:
            self._register_quantum_callbacks()

    def _register_classical_callbacks(self):
        """Register callbacks specifically for classical visualizations."""
        # Update 2D Animation Graph
        @self.app.callback(Output('2d-animation-graph', 'figure'), Input('time-slider', 'value'), prevent_initial_call=True)
        def update_2d_graph_main(time_index):
            if time_index is None or not self.viz_2d: return dash.no_update
            try: return self.viz_2d._create_2d_figure(time_index)
            except Exception as e: print(f"Error creating 2D figure: {e}"); return go.Figure(layout=go.Layout(title="Error creating 2D figure"))

        # Update 3D Animation Graph
        @self.app.callback(Output('3d-animation-graph', 'figure'), Input('time-slider', 'value'), prevent_initial_call=True)
        def update_3d_graph_main(time_index):
            if time_index is None or not self.viz_3d: return dash.no_update
            try: return self.viz_3d._create_3d_figure(time_index)
            except Exception as e: print(f"Error creating 3D figure: {e}"); return go.Figure(layout=go.Layout(title="Error creating 3D figure"))

        # Update Phase Space Graph
        @self.app.callback(Output('phase-space-graph', 'figure'), [Input('time-slider', 'value'), Input('phase-space-entity-selector', 'value'), Input('phase-space-x-axis-selector', 'value'), Input('phase-space-y-axis-selector', 'value')], State('visualization-tabs', 'value'), State('phase-space-entity-selector', 'value'), State('phase-space-x-axis-selector', 'value'), State('phase-space-y-axis-selector', 'value'), prevent_initial_call=True)
        def update_phase_space_graph_main(time_index, entity_trigger, x_axis_trigger, y_axis_trigger, active_tab, entity, x_axis, y_axis):
            if active_tab != 'tab-phase': raise dash.exceptions.PreventUpdate
            if time_index is None or not entity or not x_axis or not y_axis or not self.viz_phase: return go.Figure(layout=go.Layout(title="Select entity and axes."))
            try: return self.viz_phase._create_phase_space_figure(time_index, entity, x_axis, y_axis)
            except Exception as e: print(f"Error creating phase space figure: {e}"); return go.Figure(layout=go.Layout(title=f"Error creating phase space plot for {entity}"))

        # Update Energy graph time marker
        @self.app.callback(Output('energy-graph', 'figure', allow_duplicate=True), Input('time-slider', 'value'), State('visualization-tabs', 'value'), prevent_initial_call=True)
        def update_energy_marker_main(time_index, active_tab):
            if active_tab != 'tab-energy' or time_index is None or time_index >= self.num_steps or not self.viz_energy: raise dash.exceptions.PreventUpdate
            fig = go.Figure(self.viz_energy.initial_figure)
            try:
                current_time = self.time_points[time_index]
                fig.update_layout(shapes=[s for s in fig.layout.shapes if s.get('name') != 'time_marker'])
                fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="grey", name="time_marker")
                return fig
            except IndexError: print(f"Warning: Time index {time_index} out of bounds."); raise dash.exceptions.PreventUpdate
            except Exception as e: print(f"Error updating energy marker: {e}"); raise dash.exceptions.PreventUpdate

    def _register_quantum_callbacks(self):
        """Register callbacks specifically for quantum visualizations."""
        # Update Probability Density Graph
        @self.app.callback(Output('prob-density-graph', 'figure'), Input('time-slider', 'value'), prevent_initial_call=True)
        def update_prob_density_graph(time_index):
            if time_index is None: return dash.no_update
            try: return self._create_probability_density_figure(time_index)
            except Exception as e: print(f"Error creating probability density figure: {e}"); return go.Figure(layout=go.Layout(title="Error creating probability density plot"))

        # Update Expectation Values Graph
        @self.app.callback(Output('exp-val-graph', 'figure'), Input('exp-val-selector', 'value'), prevent_initial_call=True)
        def update_exp_val_graph(selected_values):
             if not selected_values: return go.Figure(layout=go.Layout(title="Select expectation value(s) to plot."))
             try: return self._create_expectation_values_figure(selected_values)
             except Exception as e: print(f"Error creating expectation value figure: {e}"); return go.Figure(layout=go.Layout(title="Error creating expectation value plot"))

        # Update Eigenstate Graph (if eigenstates exist)
        if self.available_eigenstates:
            @self.app.callback(Output('eigenstate-graph', 'figure'), Input('eigenstate-selector', 'value'), prevent_initial_call=True)
            def update_eigenstate_graph(state_index):
                 if state_index is None: return dash.no_update
                 try: return self._create_eigenstates_figure(state_index)
                 except Exception as e: print(f"Error creating eigenstate figure: {e}"); return go.Figure(layout=go.Layout(title="Error creating eigenstate plot"))

    # --- Quantum Plotting Helper Methods ---
    
    def _get_quantum_data(self):
         """Helper to safely get data for the primary quantum entity."""
         if not self.has_quantum_data or not self.quantum_entity_name:
              return None
         return self.entities.get(self.quantum_entity_name, {}).get('time_series')

    def _create_probability_density_figure(self, time_index):
        """Create plot for probability density at a given time."""
        q_data = self._get_quantum_data()
        if not q_data or self.quantum_dims is None or self.quantum_grid_info is None:
            return go.Figure(layout=go.Layout(title="Quantum data not available"))
            
        prob_flat = q_data.get('probability_density_flat')
        if prob_flat is None or time_index >= prob_flat.shape[0]:
             return go.Figure(layout=go.Layout(title="Probability density data not available for this time step"))
        
        prob_step_flat = prob_flat[time_index, :]
        grid_coords = self.quantum_grid_info['coords']
        n_points = self.quantum_grid_info['n_points']
        prob_reshaped = prob_step_flat.reshape(n_points)
        current_time = self.time_points[time_index]

        fig = go.Figure()
        title = f"Probability Density |ψ({('x' if self.quantum_dims >= 1 else '')}{(',y' if self.quantum_dims >= 2 else '')}{ (',z' if self.quantum_dims >= 3 else '')}, t={current_time:.2f})|²"

        if self.quantum_dims == 1:
            x_coords = grid_coords[0]
            fig.add_trace(go.Scatter(x=x_coords, y=prob_reshaped, mode='lines', name='|ψ|²'))
            fig.update_layout(title=title, xaxis_title="Position (x)", yaxis_title="Probability Density")
        elif self.quantum_dims == 2:
            x_coords, y_coords = grid_coords[0], grid_coords[1]
            fig.add_trace(go.Heatmap(z=prob_reshaped.T, x=x_coords, y=y_coords, colorscale='Viridis'))
            # Ensure aspect ratio is equal for 2D plot
            fig.update_layout(title=title, xaxis_title="Position (x)", yaxis_title="Position (y)", yaxis_scaleanchor='x')
        elif self.quantum_dims == 3:
            # Placeholder for 3D visualization (e.g., isosurface)
            # This requires plotly.graph_objects.Isosurface which needs x,y,z flat arrays
            layout = go.Layout(title=title + " (3D View TBD)")
            return go.Figure(layout=layout)
        else:
             return go.Figure(layout=go.Layout(title="Unsupported dimension for probability density plot"))

        fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))
        return fig

    def _create_expectation_values_figure(self, selected_values):
        """Create plot for selected expectation values vs. time."""
        q_data = self._get_quantum_data()
        if not q_data:
            return go.Figure(layout=go.Layout(title="Quantum data not available"))

        fig = go.Figure()
        title_parts = []
        for key in selected_values:
            data_series = q_data.get(key)
            if data_series is not None and len(data_series) == self.num_steps:
                label = key.replace('expected_','').replace('_',' ').title()
                title_parts.append(label)
                # Handle multi-dimensional expectation values (pos, mom)
                if data_series.ndim == 2:
                    dims = data_series.shape[1]
                    dim_labels = ['x', 'y', 'z']
                    for d in range(dims):
                        fig.add_trace(go.Scatter(x=self.time_points, y=data_series[:, d], mode='lines', name=f'<{label} ({dim_labels[d]})>'))
                else: # 1D values (energy)
                    fig.add_trace(go.Scatter(x=self.time_points, y=data_series, mode='lines', name=f'<{label}>'))
            else:
                 print(f"Warning: Data for expectation value '{key}' not found or invalid length.")
        
        fig.update_layout(
            title=f"Expectation Values ({ ', '.join(title_parts) }) vs. Time",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig

    def _create_eigenstates_figure(self, state_index):
        """Create plot for the probability density of a selected eigenstate."""
        if not self.available_eigenstates or state_index >= self.num_eigenstates:
             return go.Figure(layout=go.Layout(title="Selected eigenstate data not available"))
        if self.quantum_dims is None or self.quantum_grid_info is None:
             return go.Figure(layout=go.Layout(title="Quantum grid info not available"))
             
        eigenvalues = self.available_eigenstates['eigenvalues']
        eigenvectors_flat = self.available_eigenstates['eigenvectors_flat']
        
        eigenvector_flat = eigenvectors_flat[:, state_index]
        eigenvalue = eigenvalues[state_index]
        prob_density_flat = np.abs(eigenvector_flat)**2
        
        grid_coords = self.quantum_grid_info['coords']
        n_points = self.quantum_grid_info['n_points']
        prob_reshaped = prob_density_flat.reshape(n_points)

        fig = go.Figure()
        title = f"Eigenstate {state_index} Probability Density |ϕ_{state_index}({('x' if self.quantum_dims >= 1 else '')}{ (',y' if self.quantum_dims >= 2 else '')}{ (',z' if self.quantum_dims >= 3 else '')})|² (E={eigenvalue:.3e})"

        if self.quantum_dims == 1:
            x_coords = grid_coords[0]
            fig.add_trace(go.Scatter(x=x_coords, y=prob_reshaped, mode='lines', name=f'|ϕ_{{{state_index}}}|²'))
            fig.update_layout(title=title, xaxis_title="Position (x)", yaxis_title="Probability Density")
        elif self.quantum_dims == 2:
            x_coords, y_coords = grid_coords[0], grid_coords[1]
            fig.add_trace(go.Heatmap(z=prob_reshaped.T, x=x_coords, y=y_coords, colorscale='Viridis'))
            fig.update_layout(title=title, xaxis_title="Position (x)", yaxis_title="Position (y)", yaxis_scaleanchor='x')
        elif self.quantum_dims == 3:
            # Placeholder for 3D visualization
            layout = go.Layout(title=title + " (3D View TBD)")
            return go.Figure(layout=layout)
        else:
             return go.Figure(layout=go.Layout(title="Unsupported dimension for eigenstate plot"))

        fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))
        return fig

    # run_server method is inherited from BaseVisualizer

