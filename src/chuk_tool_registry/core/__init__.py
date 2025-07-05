# chuk_tool_registry/core/__init__.py
"""
Core components for the async-native tool registry.

This module provides the foundational components for the chuk_tool_registry:
- Tool metadata models
- Registry interface definitions
- Exception classes
- Global registry provider access
- Registration management
- Validation support

Example usage:
    >>> import asyncio
    >>> from chuk_tool_registry.core import get_registry, ToolMetadata, ToolRegistrationManager
    >>> 
    >>> async def example():
    ...     registry = await get_registry()
    ...     # Register a simple tool
    ...     async def my_tool(x: int) -> int:
    ...         return x * 2
    ...     await registry.register_tool(my_tool, name="doubler")
    ...     
    ...     # Retrieve and use the tool
    ...     tool = await registry.get_tool("doubler")
    ...     result = await tool(5)
    ...     print(f"Result: {result}")  # Result: 10
    >>> 
    >>> asyncio.run(example())

Registration Management:
    >>> from chuk_tool_registry.core import create_registration_manager
    >>> 
    >>> # Create isolated registration context
    >>> manager = create_registration_manager("test_context")
    >>> # Use manager for isolated tool registrations

Validation Support:
    >>> from chuk_tool_registry.core import ValidationConfig, with_validation
    >>> 
    >>> @with_validation
    >>> class MyTool:
    ...     async def execute(self, x: int, y: int) -> int:
    ...         return x + y
"""

# Version information - import from package metadata
try:
    from importlib.metadata import version
    __version__ = version("chuk-tool-registry")
except ImportError:
    # Fallback for Python < 3.8
    try:
        from importlib_metadata import version
        __version__ = version("chuk-tool-registry")
    except ImportError:
        # Final fallback for development
        __version__ = "0.0.0-dev"

# Core interface and protocol definitions
from .interface import ToolRegistryInterface

# Metadata models
from .metadata import (
    ToolMetadata,
    RateLimitConfig,
    StreamingToolMetadata,
)

# Exception classes
from .exceptions import (
    ToolProcessorError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    ParserError,
)

# Global registry provider functions
from .provider import (
    get_registry,
    set_registry,
    ToolRegistryProvider,
    temporary_registry,
    set_registry_factory,
    get_registry_info,
    reset_all_registry_state,
    create_registry,
    ensure_registry_interface,
)

# Registration management
from .registration import (
    ToolRegistrationManager,
    ToolRegistrationInfo,
    GlobalRegistrationManager,
    create_registration_manager,
    get_global_registration_manager,
    ensure_registrations,
    reset_global_registrations,
    get_registration_statistics,
    validate_registration_manager,
)

# Validation support
from .validation import (
    validate_arguments,
    validate_result,
    with_validation,
    ValidationConfig,
    create_validation_wrapper,
    validate_tool_execution,
)

# Public API - organized by functionality
__all__ = [
    # === CORE INTERFACE ===
    "ToolRegistryInterface",
    
    # === METADATA MODELS ===
    "ToolMetadata",
    "RateLimitConfig",
    "StreamingToolMetadata",
    
    # === EXCEPTIONS ===
    "ToolProcessorError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError", 
    "ToolValidationError",
    "ParserError",
    
    # === GLOBAL REGISTRY ACCESS ===
    "get_registry",
    "set_registry",
    "ToolRegistryProvider",
    "temporary_registry",
    "set_registry_factory",
    "get_registry_info",
    "reset_all_registry_state",
    "create_registry",
    "ensure_registry_interface",
    
    # === REGISTRATION MANAGEMENT ===
    "ToolRegistrationManager",
    "ToolRegistrationInfo",
    "GlobalRegistrationManager",
    "create_registration_manager",
    "get_global_registration_manager",
    "ensure_registrations",
    "reset_global_registrations",
    "get_registration_statistics",
    "validate_registration_manager",
    
    # === VALIDATION SUPPORT ===
    "validate_arguments",
    "validate_result",
    "with_validation",
    "ValidationConfig",
    "create_validation_wrapper",
    "validate_tool_execution",
    
    # === VERSION ===
    "__version__",
]


# Core package information
CORE_COMPONENTS = {
    "interface": ["ToolRegistryInterface"],
    "metadata": ["ToolMetadata", "RateLimitConfig", "StreamingToolMetadata"],
    "exceptions": ["ToolProcessorError", "ToolNotFoundError", "ToolExecutionError", 
                   "ToolTimeoutError", "ToolValidationError", "ParserError"],
    "provider": ["get_registry", "set_registry", "ToolRegistryProvider"],
    "registration": ["ToolRegistrationManager", "create_registration_manager", "ensure_registrations"],
    "validation": ["ValidationConfig", "with_validation", "validate_arguments", "validate_result"],
}


def get_core_info() -> dict:
    """
    Get information about the core module components.
    
    Returns:
        Dict with core module information
    """
    return {
        "version": __version__,
        "components": CORE_COMPONENTS,
        "total_exports": len(__all__),
        "description": "Core components for async-native tool registry with validation support",
        "validation_available": True,
    }


def validate_core_setup() -> bool:
    """
    Validate that core components are properly set up.
    
    Returns:
        True if all core components are available
    """
    try:
        # Test that we can import all major components
        from .interface import ToolRegistryInterface
        from .metadata import ToolMetadata
        from .provider import get_registry
        from .registration import ToolRegistrationManager
        from .exceptions import ToolNotFoundError
        from .validation import ValidationConfig
        
        # Basic validation that classes are properly defined
        assert issubclass(ToolNotFoundError, Exception)
        assert hasattr(ToolMetadata, 'name')
        assert hasattr(ToolRegistrationManager, 'add_registration')
        assert callable(True)
        
        return True
    except (ImportError, AssertionError, AttributeError):
        return False


# Add utility functions to exports
__all__.extend(["get_core_info", "validate_core_setup"])


# Convenience imports for common patterns
async def quick_setup() -> ToolRegistryInterface:
    """
    Quick setup function that returns a ready-to-use registry.
    
    Returns:
        A configured registry instance
        
    Example:
        >>> registry = await quick_setup()
        >>> # Registry is ready to use
    """
    return await get_registry()


async def create_isolated_context() -> tuple[ToolRegistryInterface, ToolRegistrationManager]:
    """
    Create an isolated registry and registration manager for testing.
    
    Returns:
        Tuple of (registry, registration_manager)
        
    Example:
        >>> registry, manager = await create_isolated_context()
        >>> # Use for isolated testing
    """
    from ..providers.memory import InMemoryToolRegistry
    
    registry = InMemoryToolRegistry()
    manager = create_registration_manager("isolated_context")
    
    return registry, manager


async def create_validated_context(
    validation_config: ValidationConfig = None
) -> tuple[ToolRegistryInterface, ToolRegistrationManager]:
    """
    Create an isolated registry with validation enabled.
    
    Args:
        validation_config: Optional validation configuration
        
    Returns:
        Tuple of (validated_registry, registration_manager)
        
    Example:
        >>> config = ValidationConfig(strict_mode=True)
        >>> registry, manager = await create_validated_context(config)
        >>> # Use for validated testing
    """
    from ..providers.memory import InMemoryToolRegistry
    
    config = validation_config or ValidationConfig()
    registry = InMemoryToolRegistry(
        enable_validation=True,
        validation_config=config
    )
    manager = create_registration_manager("validated_context")
    
    return registry, manager


# Add convenience functions to exports
__all__.extend(["quick_setup", "create_isolated_context", "create_validated_context"])


# Module-level validation on import (only in debug mode)
def _validate_on_import():
    """Validate core setup on import if debug mode is enabled."""
    import os
    if os.environ.get('CHUK_TOOL_REGISTRY_DEBUG'):
        if validate_core_setup():
            print("✓ Core module validation passed")
        else:
            print("⚠ Core module validation failed")


# Run validation if debug is enabled
_validate_on_import()