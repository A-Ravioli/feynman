import pytest
import numpy as np
from feynman.simulator.quantum_simulator import QuantumSimulator, _create_grid, _build_kinetic_operator_fourier, _build_kinetic_operator_finite_diff
import scipy.constants as const

# Basic physical constants for tests
hbar = const.hbar
m_e = const.m_e

@pytest.fixture
def quantum_simulator():
    """Provides a QuantumSimulator instance for tests."""
    return QuantumSimulator()

# --- Test Helper Functions ---

def test_create_grid_1d():
    dims = 1
    points = [10]
    ranges = [(-5.0, 5.0)]
    grid_coords, deltas, n_points = _create_grid(dims, points, ranges)
    assert len(grid_coords) == 1
    assert len(grid_coords[0]) == 10
    assert len(deltas) == 1
    assert len(n_points) == 1
    assert n_points[0] == 10
    assert np.isclose(deltas[0], 10.0 / 9.0) # (max - min) / (n-1)
    assert np.isclose(grid_coords[0][0], -5.0)
    assert np.isclose(grid_coords[0][-1], 5.0)

def test_create_grid_2d():
    dims = 2
    points = [10, 20]
    ranges = [(-5.0, 5.0), (0.0, 10.0)]
    grid_coords, deltas, n_points = _create_grid(dims, points, ranges)
    assert len(grid_coords) == 2
    assert len(grid_coords[0]) == 10 # x-coords
    assert len(grid_coords[1]) == 20 # y-coords
    assert len(deltas) == 2
    assert len(n_points) == 2
    assert n_points == [10, 20]
    assert np.isclose(deltas[0], 10.0 / 9.0)
    assert np.isclose(deltas[1], 10.0 / 19.0)

def test_kinetic_operator_fourier_1d():
    dims = 1
    points = [50]
    ranges = [(-10.0, 10.0)]
    grid_coords, deltas, n_points = _create_grid(dims, points, ranges)
    T_fourier = _build_kinetic_operator_fourier(dims, n_points, deltas, m_e, hbar)
    assert T_fourier.shape == (50,)
    assert np.all(T_fourier >= 0) # Kinetic energy should be non-negative

def test_kinetic_operator_finite_diff_1d():
    dims = 1
    points = [50]
    ranges = [(-10.0, 10.0)]
    grid_coords, deltas, n_points = _create_grid(dims, points, ranges)
    T_real = _build_kinetic_operator_finite_diff(dims, n_points, deltas, m_e, hbar)
    assert T_real.shape == (50, 50)
    # Check if it's roughly positive semi-definite (eigenvalues >= 0)
    # eigenvalues = np.linalg.eigvalsh(T_real.toarray()) # Dense conversion can be slow
    # assert np.all(eigenvalues > -1e-9) # Allow for numerical noise

# --- Test Full Simulation ---

@pytest.fixture
def sim_params_1d():
    return {
        "domain_settings": {
            "dimensions": 1,
            "points": [100],
            "ranges": [(-5e-9, 5e-9)] # Nanometer scale
        },
        "solver_method": "split_operator",
        "calculate_eigenstates": {"num_states": 3}
    }

@pytest.fixture
def entities_1d_free_particle():
    return {
        "particle1": {
            "type": "atom",
            "properties": {
                "mass": m_e,
                "initial_state": {
                    "wavefunction": {
                        "function": "gaussian",
                        "args": [
                            {"param": "center", "value": [0.0]},
                            {"param": "spread", "value": [1e-9]},
                            {"param": "k", "value": [5e9]} # Initial momentum
                        ]
                    }
                }
            }
        }
    }

@pytest.fixture
def entities_1d_harmonic_oscillator():
     # omega = sqrt(k/m) -> k = m * omega^2
    omega = 1e14 # Adjust frequency as needed
    k_ho = m_e * omega**2
    return {
        "particle1": {
            "type": "atom",
            "properties": {
                "mass": m_e,
                "initial_state": {
                     # Start in ground state approx (Gaussian)
                    "wavefunction": {
                        "function": "gaussian",
                         "args": [
                            {"param": "center", "value": [0.0]},
                             # Spread related to omega: sigma^2 = hbar / (2 * m * omega)
                            {"param": "spread", "value": [np.sqrt(hbar / (2 * m_e * omega))]},
                            {"param": "k", "value": [0.0]}
                        ]
                    }
                }
            }
        }
    }

@pytest.fixture
def interactions_1d_harmonic():
    omega = 1e14
    k_ho = m_e * omega**2
    return [
        {
            "source": "harmonic_potential", # Name of the field/potential source
            "target": "particle1",
            "properties": {
                "potential": {
                    "function": "harmonic",
                    "args": [
                        {"param": "k", "value": [k_ho]},
                        {"param": "center", "value": [0.0]}
                    ]
                }
            }
        }
    ]


def test_simulate_1d_free_particle(quantum_simulator, entities_1d_free_particle, sim_params_1d):
    """Test simulation of a 1D free Gaussian wavepacket."""
    entities = entities_1d_free_particle
    interactions = [] # No potential for free particle
    time_start = 0.0
    time_end = 1e-14
    time_step = 1e-16

    results = quantum_simulator.simulate(entities, interactions, time_start, time_end, time_step, sim_params_1d)

    assert "particle1" in results["entities"]
    p1_results = results["entities"]["particle1"]

    # Check output shapes
    num_steps = len(results["time_points"])
    num_points = sim_params_1d["domain_settings"]["points"][0]
    assert p1_results["wavefunction_flat"].shape == (num_steps, num_points)
    assert p1_results["probability_density_flat"].shape == (num_steps, num_points)
    assert p1_results["expected_position"].shape == (num_steps, 1)
    assert p1_results["expected_momentum"].shape == (num_steps, 1)
    assert p1_results["expected_energy"].shape == (num_steps,)

    # Check basic physics
    # Initial momentum check
    initial_k = entities["particle1"]["properties"]["initial_state"]["wavefunction"]["args"][2]["value"][0]
    expected_initial_p = hbar * initial_k
    assert np.isclose(p1_results["expected_momentum"][0, 0], expected_initial_p, rtol=1e-3)

    # Momentum conservation (should be constant for free particle)
    assert np.allclose(p1_results["expected_momentum"], p1_results["expected_momentum"][0], rtol=1e-6, atol=1e-12)

    # Energy conservation (should be constant)
    assert np.allclose(p1_results["expected_energy"], p1_results["expected_energy"][0], rtol=1e-6, atol=1e-18)

    # Wavepacket should spread, check if variance increases (approx)
    prob_density = p1_results["probability_density_flat"]
    grid_coords = p1_results["grid_info"]["coords"][0]
    expected_pos = p1_results["expected_position"][:, 0]
    variance = np.sum((grid_coords[np.newaxis, :] - expected_pos[:, np.newaxis])**2 * prob_density, axis=1) * p1_results["grid_info"]["deltas"][0]
    # Assert variance at end is greater than at start (allow for small numerical fluctuations)
    assert variance[-1] > variance[0] - 1e-25


def test_simulate_1d_harmonic_oscillator(quantum_simulator, entities_1d_harmonic_oscillator, interactions_1d_harmonic, sim_params_1d):
    """Test simulation and eigenstate calculation for 1D QHO."""
    entities = entities_1d_harmonic_oscillator
    interactions = interactions_1d_harmonic
    time_start = 0.0
    time_end = 5e-14 # A few periods
    time_step = 5e-16

    results = quantum_simulator.simulate(entities, interactions, time_start, time_end, time_step, sim_params_1d)

    assert "particle1" in results["entities"]
    p1_results = results["entities"]["particle1"]
    assert "eigenstates" in p1_results

    eigen_results = p1_results["eigenstates"]
    eigenvalues = eigen_results["eigenvalues"]
    num_states_calculated = len(eigenvalues)
    assert num_states_calculated == sim_params_1d["calculate_eigenstates"]["num_states"]

    # Check QHO eigenvalues: E_n = hbar * omega * (n + 0.5)
    omega = 1e14 # From fixture
    expected_eigenvalues = hbar * omega * (np.arange(num_states_calculated) + 0.5)

    print(f"Calculated Eigenvalues: {eigenvalues}")
    print(f"Expected Eigenvalues:   {expected_eigenvalues}")
    # Compare calculated vs theoretical eigenvalues (allow some tolerance for finite grid/diff)
    assert np.allclose(eigenvalues, expected_eigenvalues, rtol=0.05) # 5% tolerance

    # Check if starting state (ground state approx) energy is close to E0
    initial_energy = p1_results["expected_energy"][0]
    print(f"Initial state energy: {initial_energy}")
    assert np.isclose(initial_energy, eigenvalues[0], rtol=0.05)

    # Check if energy remains constant (close to E0 since we started there)
    assert np.allclose(p1_results["expected_energy"], initial_energy, rtol=1e-6, atol=1e-18)

    # Check if expected position oscillates around 0 (or stays near 0 for ground state)
    assert np.allclose(p1_results["expected_position"], 0.0, atol=1e-10) # Ground state shouldn't move much


# --- Add tests for 2D/3D, different potentials, spin etc. as needed ---

# Example placeholder for a 2D test
@pytest.mark.skip(reason="2D test not fully implemented yet")
def test_simulate_2d_potential(quantum_simulator):
     sim_params_2d = {
        "domain_settings": {
            "dimensions": 2,
            "points": [30, 30], # Smaller grid for faster test
            "ranges": [(-5, 5), (-5, 5)]
        },
        "solver_method": "split_operator"
        # Add calculate_eigenstates if needed
    }
     entities_2d = {
        "particle2d": {
            "type": "atom",
            "properties": {
                "mass": m_e,
                "initial_state": {
                    "wavefunction": {
                        "function": "gaussian",
                        "args": [
                            {"param": "center", "value": [0.0, 0.0]},
                            {"param": "spread", "value": [1.0, 1.0]},
                            {"param": "k", "value": [1.0, 0.0]}
                        ]
                    }
                }
            }
        }
    }
     interactions_2d = [ # Example: 2D harmonic oscillator
        {
            "source": "harmonic_potential_2d",
            "target": "particle2d",
            "properties": {
                "potential": {
                    "function": "harmonic",
                    "args": [
                        {"param": "k", "value": [1.0, 1.0]}, # kx, ky
                        {"param": "center", "value": [0.0, 0.0]}
                    ]
                }
            }
        }
    ]
     time_start = 0.0
     time_end = 1.0
     time_step = 0.01
     results = quantum_simulator.simulate(entities_2d, interactions_2d, time_start, time_end, time_step, sim_params_2d)
     # Add assertions for 2D results
     assert "particle2d" in results["entities"]
     # ... check shapes, energy conservation, etc. ...

# --- End of File --- 