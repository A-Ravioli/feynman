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
        """Initialize the main visualizer and specific visualizer instances."""
        # We need to initialize the base first to set up self.app
        super().__init__(sim_data)
        
        # Instantiate the specific visualizers to reuse their figure creation logic.
        # They don't run their own Dash apps; we use their methods.
        # Avoid passing self.app to prevent duplicate callback registrations.
        # We handle all callbacks centrally in this main visualizer.
        self.viz_2d = Visualizer2D(sim_data)
        self.viz_3d = Visualizer3D(sim_data)
        self.viz_phase = VisualizerPhaseSpace(sim_data)
        self.viz_energy = VisualizerEnergy(sim_data) # Energy figures are pre-calculated

        # Re-assign the layout *after* specific visualizers are initialized
        # This ensures things like phase space controls can be extracted
        self.app.layout = self._create_layout()
        # Clear any callbacks potentially registered by BaseVisualizer's __init__ 
        # before registering the main visualizer's callbacks.
        self.app.callback_map = {} 
        self.app._callback_list = []
        # Register the main visualizer's callbacks *after* the final layout is set.
        self._register_callbacks()

    def _create_layout(self):
        """Override layout to use Tabs for switching between visualizations."""
        # Define Tabs
        tabs = dcc.Tabs(id='visualization-tabs', value='tab-2d', children=[
            dcc.Tab(label='2D Animation', value='tab-2d'),
            dcc.Tab(label='3D Animation', value='tab-3d'),
            dcc.Tab(label='Phase Space', value='tab-phase'),
            dcc.Tab(label='Energy Plot', value='tab-energy'),
        ])

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
         """Override controls to include base controls and phase space selectors."""
         # Basic time controls from BaseVisualizer
         time_controls = super()._create_controls()

         # --- Recreate Phase Space Controls Directly --- 
         axis_options = [
            {'label': 'X Position (x)', 'value': 'pos_0'},
            {'label': 'Y Position (y)', 'value': 'pos_1'},
            {'label': 'Z Position (z)', 'value': 'pos_2'},
            {'label': 'X Velocity (vx)', 'value': 'vel_0'},
            {'label': 'Y Velocity (vy)', 'value': 'vel_1'},
            {'label': 'Z Velocity (vz)', 'value': 'vel_2'}
         ]
         
         # Default value for entity selector
         default_entity = self.entity_names[0] if self.entity_names else None
         
         phase_space_dropdowns = html.Div([
             html.Div([
                 html.Label("Select Entity:"),
                 dcc.Dropdown(
                     id='phase-space-entity-selector',
                     options=[{'label': name, 'value': name} for name in self.entity_names],
                     value=default_entity
                 ),
             ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
             
             html.Div([
                 html.Label("Select X-Axis:"),
                 dcc.Dropdown(
                     id='phase-space-x-axis-selector',
                     options=axis_options,
                     value='pos_0' # Default to X position
                 ),
             ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
             
             html.Div([
                 html.Label("Select Y-Axis:"),
                 dcc.Dropdown(
                     id='phase-space-y-axis-selector',
                     options=axis_options,
                     value='vel_0' # Default to X velocity
                 ),
             ], style={'width': '30%', 'display': 'inline-block'}),
         ])
         # --- End Recreated Controls --- 
         
         # Container for phase space specific controls (initially hidden)
         phase_space_controls_container = html.Div(
             id='phase-space-controls-container',
             children=[phase_space_dropdowns], # Use the recreated dropdowns
             style={'display': 'none', 'marginBottom': '20px'} # Hide by default
         )

         # Combine controls: Phase space selectors first (when visible), then time controls
         return html.Div([phase_space_controls_container, time_controls])

    def _register_callbacks(self):
        """Register callbacks for the main visualizer (tab switching, control visibility, updates)."""
        # Register base callbacks for play/pause and slider animation FIRST
        # These are defined in BaseVisualizer and operate on 'time-slider' and 'play-pause-button'
        super()._register_callbacks()

        # Callback to render tab content based on selected tab
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('visualization-tabs', 'value')
        )
        def render_tab_content(tab_value):
            # Return the appropriate Graph component for the selected tab
            if tab_value == 'tab-2d':
                return dcc.Graph(id='2d-animation-graph', style={'height': '600px'})
            elif tab_value == 'tab-3d':
                return dcc.Graph(id='3d-animation-graph', style={'height': '700px'})
            elif tab_value == 'tab-phase':
                return dcc.Graph(id='phase-space-graph', style={'height': '600px'})
            elif tab_value == 'tab-energy':
                # Energy plot figure is pre-calculated and static, just needs marker update
                return dcc.Graph(id='energy-graph', figure=self.viz_energy.initial_figure)
            return html.P("Select a tab") # Default message

        # Callback to toggle visibility of phase space controls container
        @self.app.callback(
             Output('phase-space-controls-container', 'style'),
             Input('visualization-tabs', 'value')
        )
        def toggle_phase_space_controls(tab_value):
             if tab_value == 'tab-phase':
                  return {'display': 'block', 'marginBottom': '20px'} # Show controls
             else:
                  return {'display': 'none'} # Hide controls

        # --- Callbacks to Update Figures Based on Time Slider --- 
        # These target the Graph components created by render_tab_content

        # Update 2D Animation Graph
        @self.app.callback(
            Output('2d-animation-graph', 'figure'),
            Input('time-slider', 'value'),
            prevent_initial_call=True # Avoid initial computation
        )
        def update_2d_graph_main(time_index):
            # This callback fires whenever the slider changes.
            # Dash handles the fact that the Output ('2d-animation-graph') 
            # might not be present in the layout if a different tab is selected.
            if time_index is None or not hasattr(self, 'viz_2d'):
                 return dash.no_update
            try:
                return self.viz_2d._create_2d_figure(time_index)
            except Exception as e:
                 print(f"Error creating 2D figure: {e}")
                 return go.Figure(layout=go.Layout(title="Error creating 2D figure"))

        # Update 3D Animation Graph
        @self.app.callback(
            Output('3d-animation-graph', 'figure'),
            Input('time-slider', 'value'),
            prevent_initial_call=True
        )
        def update_3d_graph_main(time_index):
            if time_index is None or not hasattr(self, 'viz_3d'):
                 return dash.no_update
            try:
                 return self.viz_3d._create_3d_figure(time_index)
            except Exception as e:
                 print(f"Error creating 3D figure: {e}")
                 return go.Figure(layout=go.Layout(title="Error creating 3D figure"))

        # Update Phase Space Graph (triggered by slider OR dropdown changes when tab is active)
        @self.app.callback(
            Output('phase-space-graph', 'figure'),
            [
                Input('time-slider', 'value'), # Triggered by time slider
                # Trigger update explicitly when dropdowns change 
                Input('phase-space-entity-selector', 'value'),
                Input('phase-space-x-axis-selector', 'value'),
                Input('phase-space-y-axis-selector', 'value')
            ],
            State('visualization-tabs', 'value'), # Check if the tab is active
            # Get current values from dropdowns using State, as slider might be the trigger
            State('phase-space-entity-selector', 'value'),
            State('phase-space-x-axis-selector', 'value'),
            State('phase-space-y-axis-selector', 'value'),
            prevent_initial_call=True
        )
        def update_phase_space_graph_main(time_index, 
                                          entity_trigger, x_axis_trigger, y_axis_trigger, # Inputs
                                          active_tab, # State
                                          entity, x_axis, y_axis): # State
            # Only update if the phase space tab is active
            if active_tab != 'tab-phase':
                 raise dash.exceptions.PreventUpdate # More efficient than returning no_update

            # Use the latest state values for entity/axes
            if time_index is None or not entity or not x_axis or not y_axis or not hasattr(self, 'viz_phase'):
                 return go.Figure(layout=go.Layout(title="Select entity and axes for phase space plot."))

            try:
                 return self.viz_phase._create_phase_space_figure(time_index, entity, x_axis, y_axis)
            except Exception as e:
                 print(f"Error creating phase space figure: {e}")
                 # Also check if entity/axis selection is valid within the method
                 return go.Figure(layout=go.Layout(title=f"Error creating phase space plot for {entity}"))

        # Update Energy graph time marker
        @self.app.callback(
            Output('energy-graph', 'figure', allow_duplicate=True), # Allow duplicate output target
            Input('time-slider', 'value'),
            State('visualization-tabs', 'value'), # Check if energy tab is active
            prevent_initial_call=True
        )
        def update_energy_marker_main(time_index, active_tab):
            if active_tab != 'tab-energy' or time_index is None or time_index >= self.num_steps or not hasattr(self, 'viz_energy'):
                 raise dash.exceptions.PreventUpdate

            # Use the pre-calculated figure and just add/update the time marker
            fig = go.Figure(self.viz_energy.initial_figure)
            try:
                current_time = self.time_points[time_index]
                # Remove existing marker before adding a new one
                fig.update_layout(shapes=[s for s in fig.layout.shapes if s.get('name') != 'time_marker'])
                fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="grey", name="time_marker")
                return fig
            except IndexError:
                 print(f"Warning: Time index {time_index} out of bounds for time points.")
                 raise dash.exceptions.PreventUpdate
            except Exception as e:
                 print(f"Error updating energy marker: {e}")
                 raise dash.exceptions.PreventUpdate

    # run_server method is inherited from BaseVisualizer

