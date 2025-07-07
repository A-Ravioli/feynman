import pytest
import numpy as np
from feynman.validation import PhysicsValidator, ValidationError, validate_and_report
from feynman.interpreter.ast import Program, Model, Object, Atom

def test_physics_validator_init():
    """Test PhysicsValidator initialization"""
    validator = PhysicsValidator()
    assert len(validator.PHYSICAL_CONSTANTS) > 0
    assert len(validator.RANGES) > 0
    assert validator.warnings == []
    assert validator.errors == []

def test_validate_model_classical():
    """Test model validation for classical physics"""
    validator = PhysicsValidator()
    
    # Valid classical model
    valid_model = {
        'type': 'classical',
        'time': '0..10', 
        'resolution': 0.01
    }
    assert validator.validate_model(valid_model)
    
    # Invalid time range
    invalid_time = {
        'type': 'classical',
        'time': '10..5',  # End before start
        'resolution': 0.01
    }
    with pytest.raises(ValidationError, match="End time.*must be greater than start time"):
        validator.validate_model(invalid_time)
    
    # Negative resolution
    invalid_resolution = {
        'type': 'classical',
        'time': '0..10',
        'resolution': -0.01
    }
    with pytest.raises(ValidationError, match="Resolution must be positive"):
        validator.validate_model(invalid_resolution)

def test_validate_model_quantum():
    """Test model validation for quantum physics"""
    validator = PhysicsValidator()
    
    # Valid quantum model
    valid_quantum = {
        'type': 'quantum',
        'time': '0..1',
        'resolution': 0.01,
        'domain_settings': {
            'dimensions': 2,
            'points': [50, 50],
            'ranges': [[-5, 5], [-5, 5]]
        }
    }
    assert validator.validate_model(valid_quantum)
    
    # Invalid dimensions
    invalid_dims = {
        'type': 'quantum',
        'time': '0..1',
        'resolution': 0.01,
        'domain_settings': {
            'dimensions': 5,  # Invalid
            'points': [50],
            'ranges': [[-5, 5]]
        }
    }
    with pytest.raises(ValidationError, match="Dimensions must be 1, 2, or 3"):
        validator.validate_model(invalid_dims)

def test_validate_entity_object():
    """Test entity validation for classical objects"""
    validator = PhysicsValidator()
    
    # Valid object
    valid_object = {
        'mass': 1.0,
        'position': [0, 1, 2],
        'velocity': [0.5, 0.2, 0.1]
    }
    assert validator.validate_entity('object', valid_object)
    
    # Invalid mass
    invalid_mass = {
        'mass': -1.0,  # Negative mass
        'position': [0, 1, 2],
        'velocity': [0.5, 0.2, 0.1]
    }
    with pytest.raises(ValidationError, match="Mass must be positive"):
        validator.validate_entity('object', invalid_mass)
    
    # Invalid velocity (faster than light)
    c = validator.PHYSICAL_CONSTANTS['c']
    invalid_velocity = {
        'mass': 1.0,
        'position': [0, 1, 2],
        'velocity': [c, 0, 0]  # Speed of light
    }
    with pytest.raises(ValidationError, match="exceeds speed of light"):
        validator.validate_entity('object', invalid_velocity)

def test_validate_entity_atom():
    """Test entity validation for quantum atoms"""
    validator = PhysicsValidator()
    
    # Valid atom
    valid_atom = {
        'mass': 9.11e-31,
        'initial_state': {
            'wavefunction': {
                'function': 'gaussian',
                'kwargs': {
                    'center': [0.0],
                    'spread': [1.0],
                    'k': [0.0]
                }
            }
        }
    }
    assert validator.validate_entity('atom', valid_atom)
    
    # Invalid wavefunction (negative spread)
    invalid_wf = {
        'mass': 9.11e-31,
        'initial_state': {
            'wavefunction': {
                'function': 'gaussian',
                'kwargs': {
                    'center': [0.0],
                    'spread': [-1.0],  # Negative spread
                    'k': [0.0]
                }
            }
        }
    }
    with pytest.raises(ValidationError, match="spread.*must be positive"):
        validator.validate_entity('atom', invalid_wf)

def test_validate_interaction():
    """Test interaction validation"""
    validator = PhysicsValidator()
    
    # Valid interaction
    valid_interaction = {
        'source': 'object1',
        'target': 'object2',
        'properties': {
            'force': 'gravity'
        }
    }
    assert validator.validate_interaction(valid_interaction)
    
    # Missing source/target
    invalid_interaction = {
        'properties': {
            'force': 'gravity'
        }
    }
    with pytest.raises(ValidationError, match="must specify source and target"):
        validator.validate_interaction(invalid_interaction)

def test_validate_program():
    """Test full program validation"""
    validator = PhysicsValidator()
    
    # Create mock program with valid data
    program = Program()
    
    # Add valid model
    model = Model(name="test_model", properties={
        'type': 'classical',
        'time': '0..10',
        'resolution': 0.01
    })
    program.add_model(model)
    
    # Add valid object
    obj = Object(name="test_object", properties={
        'mass': 1.0,
        'position': [0, 0, 0],
        'velocity': [1, 0, 0]
    })
    program.add_object(obj)
    
    is_valid, warnings, errors = validator.validate_program(program)
    assert is_valid
    assert len(errors) == 0

def test_validate_with_warnings():
    """Test validation that generates warnings"""
    validator = PhysicsValidator()
    
    # Model with very large mass (should warn but not error)
    model_props = {
        'type': 'classical',
        'time': '0..10',
        'resolution': 0.01
    }
    
    entity_props = {
        'mass': 1e60,  # Very large mass - should generate warning
        'position': [0, 0, 0],
        'velocity': [1, 0, 0]
    }
    
    # Validate entity - should succeed but generate warning
    result = validator.validate_entity('object', entity_props)
    assert result
    assert len(validator.warnings) > 0

def test_validation_ranges():
    """Test parameter range validation"""
    validator = PhysicsValidator()
    
    # Test time range
    assert validator._in_range(1.0, 'time')
    assert not validator._in_range(-1.0, 'time')
    
    # Test mass range  
    assert validator._in_range(1e-30, 'mass')
    assert not validator._in_range(1e60, 'mass')
    
    # Test position range
    assert validator._in_range(1e10, 'position')
    assert validator._in_range(-1e10, 'position')

def test_convenience_function():
    """Test validate_and_report convenience function"""
    # Create simple valid program
    program = Program()
    model = Model(name="simple", properties={'type': 'classical', 'time': '0..1', 'resolution': 0.1})
    program.add_model(model)
    
    # Should return True for valid program
    result = validate_and_report(program)
    assert result

if __name__ == "__main__":
    pytest.main([__file__])