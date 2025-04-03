# Feynman

A universal DSL for classical and quantum physics simulations. SQL for reality.

## Overview

Feynman is a domain-specific language (DSL) designed to unify classical and quantum physics simulations. It provides a clean, intuitive syntax for defining physical systems, running simulations, and visualizing results.

## Features

- **Domain-specific language** for defining physics simulations
- **Unified framework** for both classical and quantum systems
- **Built-in visualization** of simulation results
- **Extensible architecture** for adding new physics models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/feynman.git
cd feynman

# Install the package
pip install -e .
```

## Usage

### Basic Syntax

```physica
model <name>:
    type: [classical | quantum]
    time: <range>
    resolution: <step>

object <name>:
    mass: <value>
    position: [x, y, z]
    velocity: [vx, vy, vz]

atom <name>:
    mass: <value>
    initial_state:
        wavefunction: <wavefunction_function>(<parameters>)

field <name>:
    type: <field_type>
    <properties>

interaction <source> -> <target>:
    [force | potential | action]: <value>

simulate <model_name>
visualize <entity_name> [on <target_name>]
```

### Running Simulations

```bash
# Run a simulation
feynman run examples/double_slit.phys --visualize

# Visualize existing results
feynman visualize results.json
```

## Examples

### Double Slit Experiment (Quantum)

```physica
model double_slit_experiment:
    type: quantum
    time: 0..2
    resolution: 0.01

atom electron:
    mass: 9.11e-31
    initial_state:
        wavefunction: gaussian(center=-5.0, spread=0.5, k=5.0)

field barrier:
    type: barrier
    position: [0, 0, 0]
    holes: [[-0.5, -0.3], [0.3, 0.5]]

interaction electron -> barrier:
    potential: double_slit(position=0.0, slit_width=0.2, slit_separation=0.8)

simulate double_slit_experiment
visualize electron
```

### Two-Body Gravitational System (Classical)

```physica
model two_body_system:
    type: classical
    time: 0..100
    resolution: 0.1

object sun:
    mass: 1.989e30
    position: [0, 0, 0]
    velocity: [0, 0, 0]

object earth:
    mass: 5.972e24
    position: [1.496e11, 0, 0]  # 1 AU distance
    velocity: [0, 29780, 0]     # Initial orbital velocity (m/s)

interaction sun -> earth:
    force: gravity(G=6.67430e-11)

simulate two_body_system
visualize earth
```

## Core Components

1. **Models**: Define the simulation container and its parameters
2. **Objects**: Classical entities with position, velocity, and mass
3. **Atoms**: Quantum entities with wavefunctions and quantum states
4. **Fields**: Space-dependent fields like potentials or detectors
5. **Interactions**: Define how entities interact via forces, potentials, or measurements

## License

[MIT](LICENSE.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
