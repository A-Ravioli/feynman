#!/usr/bin/env python3
import argparse
import sys
import os
import json
import webbrowser
import tempfile
from pathlib import Path
from typing import Dict, Any

from .interpreter.interpreter import Interpreter
from .visualizer.visualizer import Visualizer

class PhysicaLangCLI:
    """Command-line interface for PhysicaLang"""
    
    def __init__(self):
        self.interpreter = Interpreter()
        self.visualizer = Visualizer()
    
    def run(self, args=None):
        """Run the CLI with the given arguments"""
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.subcommand == "run":
            self.run_simulation(parsed_args.file[0], parsed_args.visualize)
        elif parsed_args.subcommand == "visualize":
            self.visualize_results(parsed_args.results[0])
        else:
            parser.print_help()
    
    def create_argument_parser(self):
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="PhysicaLang - A universal DSL for classical and quantum physics simulations"
        )
        subparsers = parser.add_subparsers(dest="subcommand", help="Subcommands")
        
        # Run simulation command
        run_parser = subparsers.add_parser("run", help="Run a physics simulation from a PhysicaLang script")
        run_parser.add_argument("file", nargs=1, help="PhysicaLang script file")
        run_parser.add_argument(
            "--output", "-o", 
            help="Output file for simulation results (default: results.json)"
        )
        run_parser.add_argument(
            "--visualize", "-v", action="store_true",
            help="Visualize results after simulation"
        )
        
        # Visualize results command
        vis_parser = subparsers.add_parser("visualize", help="Visualize existing simulation results")
        vis_parser.add_argument("results", nargs=1, help="Simulation results JSON file")
        
        return parser
    
    def run_simulation(self, script_file: str, visualize=False):
        """Run a simulation from a PhysicaLang script file"""
        if not os.path.exists(script_file):
            print(f"Error: File '{script_file}' not found")
            sys.exit(1)
        
        print(f"Running simulation from {script_file}...")
        
        try:
            with open(script_file, "r") as f:
                code = f.read()
            
            # Run the interpreter
            results = self.interpreter.interpret(code)
            
            # Save results
            output_file = "results.json"
            with open(output_file, "w") as f:
                # Convert NumPy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"Simulation completed. Results saved to {output_file}")
            
            # Generate visualization if requested
            if visualize:
                self.visualize_results(output_file)
            
        except Exception as e:
            print(f"Error running simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def visualize_results(self, results_file: str):
        """Visualize simulation results from a JSON file"""
        if not os.path.exists(results_file):
            print(f"Error: Results file '{results_file}' not found")
            sys.exit(1)
        
        print(f"Visualizing results from {results_file}...")
        
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Generate HTML visualization
            html = self._generate_html_report(results)
            
            # Save to a temporary file and open in browser
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                f.write(html.encode("utf-8"))
                html_file = f.name
            
            print(f"Opening visualization in browser...")
            webbrowser.open(f"file://{html_file}")
            
        except Exception as e:
            print(f"Error visualizing results: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _make_json_serializable(self, obj):
        """Convert NumPy arrays to lists for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {
                    "real": obj.real.tolist(),
                    "imag": obj.imag.tolist()
                }
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return obj
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate an HTML report for visualization results"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PhysicaLang Simulation Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background-color: #3498db;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .section {
                    background-color: white;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                }
                .entity {
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    display: block;
                    margin: 10px 0;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .animation {
                    text-align: center;
                    margin: 20px 0;
                }
                .classical { background-color: #f1c40f; color: black; padding: 3px 6px; border-radius: 3px; }
                .quantum { background-color: #9b59b6; color: white; padding: 3px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PhysicaLang Simulation Results</h1>
            </div>
        """
        
        # Add simulation results section
        if "simulation_results" in results:
            sim_results = results["simulation_results"]
            
            for model_name, model_results in sim_results.items():
                html += f"""
                <div class="section">
                    <h2>Model: {model_name}</h2>
                """
                
                # Add entity sections
                if "entities" in model_results:
                    entities = model_results["entities"]
                    
                    for entity_name, entity_data in entities.items():
                        entity_type = entity_data.get("type", "unknown")
                        type_class = "quantum" if entity_type == "atom" else "classical"
                        
                        html += f"""
                        <div class="entity">
                            <h3>{entity_name} <span class="{type_class}">{entity_type}</span></h3>
                        """
                        
                        # Add visualizations
                        if "visualization_results" in results:
                            for vis_result in results["visualization_results"]:
                                if entity_name in vis_result:
                                    entity_vis = vis_result[entity_name]
                                    
                                    # Add plots
                                    if entity_type == "object":
                                        if "trajectory_plot" in entity_vis:
                                            html += f"""
                                            <h4>Trajectory</h4>
                                            <img src="data:image/png;base64,{entity_vis['trajectory_plot']}" alt="Trajectory">
                                            """
                                        
                                        if "energy_plot" in entity_vis:
                                            html += f"""
                                            <h4>Energy</h4>
                                            <img src="data:image/png;base64,{entity_vis['energy_plot']}" alt="Energy">
                                            """
                                    
                                    elif entity_type == "atom":
                                        if "wavefunction_plot" in entity_vis:
                                            html += f"""
                                            <h4>Wavefunction</h4>
                                            <img src="data:image/png;base64,{entity_vis['wavefunction_plot']}" alt="Wavefunction">
                                            """
                                        
                                        if "probability_evolution" in entity_vis:
                                            html += f"""
                                            <h4>Probability Density Evolution</h4>
                                            <img src="data:image/png;base64,{entity_vis['probability_evolution']}" alt="Probability Density Evolution">
                                            """
                                    
                                    # Add animation if available
                                    if "animation" in entity_vis:
                                        html += f"""
                                        <div class="animation">
                                            <h4>Animation</h4>
                                            <img src="data:image/gif;base64,{entity_vis['animation']}" alt="Animation">
                                        </div>
                                        """
                        
                        html += "</div>"  # Close entity div
                
                html += "</div>"  # Close section div
        
        html += """
        </body>
        </html>
        """
        
        return html

def main():
    """Main entry point for the CLI"""
    cli = PhysicaLangCLI()
    cli.run()

if __name__ == "__main__":
    main() 