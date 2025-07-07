"""
Flask API server for Feynman Physics Lab frontend
Provides endpoints for running simulations and serving results
"""
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback

from feynman.main import PhysicaLangCLI

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize the CLI
cli = PhysicaLangCLI()

# Store for simulation results
RESULTS_DIR = Path("api_results")
RESULTS_DIR.mkdir(exist_ok=True)

@app.route('/status', methods=['GET'])
def get_status():
    """Get API status and health check"""
    return jsonify({
        'status': 'online',
        'version': '1.0.0',
        'uptime': 'unknown',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/simulate', methods=['POST'])
def run_simulation():
    """Run a simulation from PhysicsLang code"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'No code provided'}), 400
        
        code = data['code']
        options = data.get('options', {})
        
        # Create temporary file for the simulation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.phys', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Run the simulation
            output_file = RESULTS_DIR / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cli.run_simulation(temp_file_path, str(output_file), visualize=False)
            
            # Load and return results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            return jsonify(results)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Simulation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/results', methods=['GET'])
def get_results():
    """Get the latest simulation results"""
    try:
        # Find the most recent results file
        result_files = list(RESULTS_DIR.glob("*.json"))
        if not result_files:
            return jsonify({'error': 'No results found'}), 404
        
        latest_file = max(result_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error getting results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-simulate', methods=['POST'])
def upload_and_simulate():
    """Upload a .phys file and run simulation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.phys'):
            return jsonify({'error': 'Invalid file. Please upload a .phys file'}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w', suffix='.phys', delete=False) as temp_file:
            content = file.read().decode('utf-8')
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Run the simulation
            output_file = RESULTS_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            cli.run_simulation(temp_file_path, str(output_file), visualize=False)
            
            # Load and return results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            return jsonify(results)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Upload simulation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate_code():
    """Validate PhysicsLang code without running simulation"""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'No code provided'}), 400
        
        code = data['code']
        
        # Try to parse the code using the interpreter
        try:
            cli.interpreter.interpret(code)
            return jsonify({
                'valid': True,
                'errors': [],
                'warnings': []
            })
        except Exception as e:
            return jsonify({
                'valid': False,
                'errors': [str(e)],
                'warnings': []
            })
            
    except Exception as e:
        print(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get list of available example files"""
    try:
        examples_dir = Path("examples")
        if not examples_dir.exists():
            return jsonify({'examples': []})
        
        example_files = [f.name for f in examples_dir.glob("*.phys")]
        return jsonify({'examples': example_files})
        
    except Exception as e:
        print(f"Error getting examples: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/examples/<filename>', methods=['POST'])
def load_example(filename):
    """Load and run a specific example"""
    try:
        examples_dir = Path("examples")
        example_file = examples_dir / filename
        
        if not example_file.exists():
            return jsonify({'error': f'Example {filename} not found'}), 404
        
        # Run the example simulation
        output_file = RESULTS_DIR / f"example_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        cli.run_simulation(str(example_file), str(output_file), visualize=False)
        
        # Load and return results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error loading example: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    # Mock metrics for now
    return jsonify({
        'cpu_usage': 25.5,
        'memory_usage': 45.2,
        'simulation_time': 0.123,
        'entity_count': 3
    })

@app.route('/simulations', methods=['GET'])
def list_simulations():
    """List saved simulations"""
    try:
        result_files = list(RESULTS_DIR.glob("*.json"))
        simulations = []
        
        for file_path in result_files:
            stat = file_path.stat()
            simulations.append({
                'id': file_path.stem,
                'name': file_path.name,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'type': 'simulation'
            })
        
        # Sort by creation time, newest first
        simulations.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({'simulations': simulations})
        
    except Exception as e:
        print(f"Error listing simulations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/simulations/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id):
    """Get a specific simulation by ID"""
    try:
        file_path = RESULTS_DIR / f"{simulation_id}.json"
        
        if not file_path.exists():
            return jsonify({'error': 'Simulation not found'}), 404
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error getting simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/simulations/<simulation_id>', methods=['DELETE'])
def delete_simulation(simulation_id):
    """Delete a simulation"""
    try:
        file_path = RESULTS_DIR / f"{simulation_id}.json"
        
        if not file_path.exists():
            return jsonify({'error': 'Simulation not found'}), 404
        
        file_path.unlink()
        return jsonify({'message': 'Simulation deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting simulation: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Feynman Physics Lab API Server...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:8001")
    print()
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8001, debug=True)