#!/bin/bash

echo "🚀 Setting up Feynman Physics Lab with Full Frontend/Backend"
echo "==========================================================="

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down servers..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "   ✓ API server stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "   ✓ Frontend server stopped"
    fi
    if [ -d "venv" ]; then
        deactivate 2>/dev/null
        echo "   ✓ Virtual environment deactivated"
    fi
    echo "🏁 Cleanup complete!"
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Ensure the examples directory exists
mkdir -p examples
mkdir -p api_results

echo "📦 Setting up Python backend..."

# Setup a Python virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Created virtual environment"
else
    echo "   ✓ Using existing virtual environment"
fi

source venv/bin/activate
echo "   ✓ Activated virtual environment"

# Install the package and dependencies
uv pip install -e . > /dev/null 2>&1
uv pip install flask flask-cors > /dev/null 2>&1
echo "   ✓ Installed Python dependencies"

echo "🌐 Starting API server..."
# Start the API server in background
uv run python api_server.py &
API_PID=$!
echo "   ✓ API server started (PID: $API_PID) on http://localhost:8001"

# Wait a moment for API server to start
sleep 3

echo "⚛️  Setting up React frontend..."

# Check if we're in the right directory for frontend
if [ -d "frontend/frontend" ]; then
    cd frontend/frontend
    
    # Install npm dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "   📥 Installing npm dependencies..."
        npm install > /dev/null 2>&1
        echo "   ✓ Installed npm dependencies"
    else
        echo "   ✓ Using existing npm dependencies"
    fi
    
    echo "🎨 Starting frontend development server..."
    # Start the frontend dev server in background
    npm run dev &
    FRONTEND_PID=$!
    echo "   ✓ Frontend server started (PID: $FRONTEND_PID) on http://localhost:3000"
    
    cd ../..
else
    echo "   ❌ Frontend directory not found at frontend/frontend"
    exit 1
fi

echo ""
echo "🎉 Feynman Physics Lab is now running!"
echo "=================================="
echo "📊 Frontend Dashboard: http://localhost:3000"
echo "🔌 Backend API:        http://localhost:8001"
echo ""
echo "Available examples to try:"
echo "  • double_slit.phys       - Quantum double slit experiment"
echo "  • quantum_3d.phys        - 3D quantum visualization"
echo "  • two_body_gravity.phys  - Classical gravitational system"
echo "  • collision_demo.phys    - Particle collision simulation"
echo ""
echo "💡 Usage:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Use the file upload to run .phys examples"
echo "  3. Or write PhysicsLang code directly in the editor"
echo "  4. View real-time 3D visualizations and results"
echo ""
echo "🛑 Press Ctrl+C to stop all servers and exit"
echo ""

# Wait for user to interrupt
while true; do
    sleep 1
done 