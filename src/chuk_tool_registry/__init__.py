# src/chuk_tool_registry/__init__.py
"""
Chuk Tool Registry - Async-native tool registration and discovery system with validation.
"""

# Version information - import from package metadata
try:
    from importlib.metadata import version, metadata
    __version__ = version("chuk-tool-registry")
    _metadata = metadata("chuk-tool-registry")
    __author__ = _metadata.get("Author", "Chuk Development Team")
    __description__ = _metadata.get("Summary", "Async-native tool registration and discovery system")
except ImportError:
    # Fallback for Python < 3.8
    try:
        from importlib_metadata import version, metadata
        __version__ = version("chuk-tool-registry")
        _metadata = metadata("chuk-tool-registry")
        __author__ = _metadata.get("Author", "Chuk Development Team")
        __description__ = _metadata.get("Summary", "Async-native tool registration and discovery system")
    except ImportError:
        # Final fallback for development
        __version__ = "0.0.0-dev"
        __author__ = "Chuk Development Team"
        __description__ = "Async-native tool registration and discovery system"

# Core registry components
from .core import (
    # Registry interface and provider
    ToolRegistryInterface,
    get_registry,
    set_registry,
    ToolRegistryProvider,
    
    # Metadata models
    ToolMetadata,
    RateLimitConfig,
    StreamingToolMetadata,
    
    # Exceptions
    ToolProcessorError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolValidationError,
    ParserError,
    
    # Validation support
    validate_arguments,
    validate_result,
    with_validation,
    ValidationConfig,
    create_validation_wrapper,
    validate_tool_execution,
)

# Registration management
from .core.registration import (
    ToolRegistrationManager,
    ToolRegistrationInfo,
    create_registration_manager,
    get_global_registration_manager,
    # ensure_registrations - using enhanced version from discovery
    reset_global_registrations,
    get_registration_statistics,
)

# Discovery and registration decorators
from .discovery import (
    # Core decorator
    register_tool,
    
    # IMPORTANT: Enhanced ensure_registrations with deferred registration support
    ensure_registrations,
    
    # Function registration
    register_fn_tool,
    register_langchain_tool,
    register_function_batch,
    register_module_functions,
    
    # Utility functions
    discover_decorated_tools,
    make_tool_serializable,
    validate_tool_function,
    get_registered_function_tools,
    
    # Statistics and introspection
    get_discovery_stats,
    list_tools_by_source,
    validate_all_registered_tools,
    
    # Convenience functions
    quick_register_function,
    
    # Tool wrappers
    FunctionToolWrapper,
    LangChainToolWrapper,
    
    # Exceptions
    SchemaGenerationError,
    ToolRegistrationError,
)

# Enhanced validation decorators
from .discovery.decorators import (
    validated_tool,
    requires_pydantic,
    discover_validation_enabled_tools,
)

# Enhanced auto-registration functions
from .discovery.auto_register import (
    get_validation_statistics,
    validate_registry_tools,
    create_validated_function_tool,
    create_validated_langchain_tool,
)

# Provider implementations
from .providers import (
    get_registry as get_provider_registry,
    clear_registry_cache,
)

# Main public API - organized by category
__all__ = [
    # === CORE REGISTRY ===
    "ToolRegistryInterface",
    "get_registry", 
    "set_registry",
    "ToolRegistryProvider",
    
    # === REGISTRATION MANAGEMENT ===
    "ToolRegistrationManager",
    "ToolRegistrationInfo", 
    "create_registration_manager",
    "get_global_registration_manager",
    "ensure_registrations",  # Enhanced version from discovery
    "reset_global_registrations",
    "get_registration_statistics",
    
    # === TOOL REGISTRATION ===
    "register_tool",           # Class/function decorator
    "validated_tool",          # Validation-enabled decorator
    "requires_pydantic",       # Pydantic requirement decorator
    "register_fn_tool",        # Function registration
    "register_langchain_tool", # LangChain integration
    "register_function_batch", # Batch operations
    "register_module_functions",
    "quick_register_function",
    
    # === DISCOVERY AND UTILITIES ===
    "discover_decorated_tools",
    "discover_validation_enabled_tools",
    "make_tool_serializable",
    "validate_tool_function",
    "get_registered_function_tools",
    "get_discovery_stats",
    "list_tools_by_source",
    "validate_all_registered_tools",
    
    # === VALIDATION FEATURES ===
    "get_validation_statistics",
    "validate_registry_tools",
    "create_validated_function_tool",
    "create_validated_langchain_tool",
    
    # === METADATA MODELS ===
    "ToolMetadata",
    "RateLimitConfig", 
    "StreamingToolMetadata",
    
    # === VALIDATION SUPPORT ===
    "validate_arguments",
    "validate_result",
    "with_validation",
    "ValidationConfig",
    "create_validation_wrapper",
    "validate_tool_execution",
    
    # === TOOL WRAPPERS ===
    "FunctionToolWrapper",
    "LangChainToolWrapper",
    
    # === EXCEPTIONS ===
    "ToolProcessorError",
    "ToolNotFoundError",
    "ToolExecutionError", 
    "ToolTimeoutError",
    "ToolValidationError",
    "ParserError",
    "SchemaGenerationError",
    "ToolRegistrationError",
    
    # === PROVIDERS ===
    "get_provider_registry",
    "clear_registry_cache",
    
    # === VERSION ===
    "__version__",
]

# Convenience aliases for common operations
register = register_tool  # Shorter alias for the decorator
register_function = register_fn_tool  # Clear alias for function registration

# Add aliases to exports
__all__.extend(["register", "register_function"])


# Package-level convenience functions with validation support
async def setup_registry(provider_type: str = "memory", **kwargs) -> ToolRegistryInterface:
    """
    Set up a registry with the specified provider.
    
    Args:
        provider_type: Type of registry provider ("memory", "redis", etc.)
        **kwargs: Additional configuration for the provider
        
    Returns:
        The configured registry instance
        
    Example:
        >>> registry = await setup_registry("memory", enable_statistics=True)
        >>> await set_registry(registry)
    """
    registry = await get_provider_registry(provider_type, **kwargs)
    await set_registry(registry)
    return registry


async def setup_validated_registry(
    provider_type: str = "memory", 
    validation_config: ValidationConfig = None,
    enable_validation_by_default: bool = False,
    **kwargs
) -> ToolRegistryInterface:
    """
    Set up a registry with validation enabled.
    
    Args:
        provider_type: Type of registry provider ("memory", "redis", etc.)
        validation_config: Validation configuration
        enable_validation_by_default: Whether to enable validation for all tools by default
        **kwargs: Additional configuration for the provider
        
    Returns:
        The configured registry instance with validation enabled
        
    Example:
        >>> config = ValidationConfig(strict_mode=True)
        >>> registry = await setup_validated_registry("memory", config, True)
        >>> await set_registry(registry)
    """
    validation_config = validation_config or ValidationConfig()
    
    # Add validation parameters to kwargs
    kwargs.update({
        'enable_validation': True,
        'default_validation_config': validation_config,
        'enable_validation_by_default': enable_validation_by_default,
    })
    
    registry = await get_provider_registry(provider_type, **kwargs)
    await set_registry(registry)
    return registry


async def register_and_setup(
    *tools_or_functions, 
    namespace: str = "default", 
    enable_validation: bool = False,
    validation_config: ValidationConfig = None,
    **metadata
) -> dict:
    """
    Convenience function to register multiple tools and set up the registry with validation support.
    
    Args:
        *tools_or_functions: Tool classes or functions to register
        namespace: Namespace for all tools
        enable_validation: Whether to enable validation for registered tools
        validation_config: Validation configuration for tools
        **metadata: Additional metadata for all tools
        
    Returns:
        Dict with registration results
        
    Example:
        >>> async def my_func(x: int) -> int:
        ...     return x * 2
        >>> 
        >>> @register_tool("calculator")
        >>> class Calculator:
        ...     async def execute(self, a: int, b: int) -> int:
        ...         return a + b
        >>> 
        >>> results = await register_and_setup(
        ...     my_func, Calculator, 
        ...     namespace="demo",
        ...     enable_validation=True,
        ...     validation_config=ValidationConfig(strict_mode=True)
        ... )
    """
    results = {"functions": [], "classes": [], "errors": []}
    
    for item in tools_or_functions:
        try:
            if callable(item) and not hasattr(item, '__init__'):
                # It's a function
                await register_fn_tool(
                    item, 
                    namespace=namespace,
                    enable_validation=enable_validation,
                    validation_config=validation_config,
                    **metadata
                )
                results["functions"].append(item.__name__)
            elif hasattr(item, '_tool_registration_info'):
                # It's already decorated
                results["classes"].append(item.__name__)
            else:
                # Try to register as a tool class
                enhanced_item = register_tool(
                    namespace=namespace, 
                    enable_validation=enable_validation,
                    validation_config=validation_config,
                    **metadata
                )(item)
                results["classes"].append(enhanced_item.__name__)
        except Exception as e:
            results["errors"].append(f"{getattr(item, '__name__', str(item))}: {e}")
    
    # Process any pending registrations
    await ensure_registrations()
    
    return results


async def get_all_tools(namespace: str = None, validation_only: bool = False) -> dict:
    """
    Get information about all registered tools with validation information.
    
    Args:
        namespace: Optional namespace filter
        validation_only: If True, only return validation-enabled tools
        
    Returns:
        Dict with comprehensive tool information including validation status
        
    Example:
        >>> tools_info = await get_all_tools()
        >>> for ns, tools in tools_info["by_namespace"].items():
        ...     print(f"{ns}: {len(tools)} tools")
        >>> 
        >>> validation_tools = await get_all_tools(validation_only=True)
        >>> print(f"Validation-enabled tools: {validation_tools['total_tools']}")
    """
    registry = await get_registry()
    
    # Get basic tool list
    all_tools = await registry.list_tools(namespace)
    
    # Get all metadata
    all_metadata = await registry.list_metadata(namespace)
    
    # Filter for validation-enabled tools if requested
    if validation_only:
        if hasattr(registry, 'list_validation_enabled_tools'):
            validation_tools = await registry.list_validation_enabled_tools(namespace)
            all_tools = validation_tools
        else:
            # Fallback: filter by metadata
            validation_metadata = []
            for metadata in all_metadata:
                validation_enabled = getattr(metadata, 'execution_options', {}).get('validation_enabled', False)
                if validation_enabled:
                    validation_metadata.append(metadata)
            all_metadata = validation_metadata
            all_tools = [(m.namespace, m.name) for m in all_metadata]
    
    # Organize by namespace
    by_namespace = {}
    for ns, name in all_tools:
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(name)
    
    # Get source breakdown
    source_breakdown = {}
    validation_breakdown = {"enabled": 0, "disabled": 0}
    
    for metadata in all_metadata:
        source = getattr(metadata, 'source', 'unknown')
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        # Count validation-enabled tools
        validation_enabled = getattr(metadata, 'execution_options', {}).get('validation_enabled', False)
        if validation_enabled:
            validation_breakdown["enabled"] += 1
        else:
            validation_breakdown["disabled"] += 1
    
    return {
        "total_tools": len(all_tools),
        "namespaces": list(by_namespace.keys()),
        "by_namespace": by_namespace,
        "source_breakdown": source_breakdown,
        "validation_breakdown": validation_breakdown,
        "validation_available": True,  # Always available now
        "metadata": {m.name: m for m in all_metadata}
    }


def get_version_info() -> dict:
    """
    Get detailed version information.
    
    Returns:
        Dict with version details
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "validation_available": True,  # Always available now
    }


# Add convenience functions to exports
__all__.extend([
    "setup_registry",
    "setup_validated_registry",
    "register_and_setup", 
    "get_all_tools",
    "get_version_info",
])


# Package initialization message (only in debug mode)
def _debug_info():
    """Print debug information about the package."""
    import os
    if os.environ.get('CHUK_TOOL_REGISTRY_DEBUG'):
        print(f"🔧 Chuk Tool Registry v{__version__} initialized")
        print(f"   Available providers: memory")
        print(f"   Core components: {len([x for x in __all__ if not x.startswith('_')])} exports")
        
        print(f"   ✅ Validation support: enabled (Pydantic always available)")


# Only show debug info if explicitly requested
_debug_info()


# Documentation helper
def get_package_help() -> str:
    """
    Get comprehensive help for the chuk_tool_registry package.
    
    Returns:
        Formatted help text with examples and API overview
    """
    validation_status = "✅ Available"  # Always available now
    
    return f"""
Chuk Tool Registry v{__version__}
================================

QUICK START:
  1. Decorate tool functions:
     @register_tool("my_tool", namespace="utilities")
     async def my_function(x: int) -> int:
         return x * 2

  2. Register with validation:
     @validated_tool("add", strict_mode=True)
     async def add_numbers(a: int, b: int) -> int:
         return a + b

  3. Process registrations:
     await ensure_registrations()

  4. Use tools:
     registry = await get_registry()
     tool = await registry.get_tool("my_tool", "utilities")
     result = await tool.execute(5)

MAIN COMPONENTS:
  • Core Registry: ToolRegistryInterface, get_registry(), ToolMetadata
  • Registration: @register_tool, register_fn_tool(), ToolRegistrationManager
  • Discovery: discover_decorated_tools(), get_discovery_stats()
  • Validation: ValidationConfig, @validated_tool, @with_validation ({validation_status})
  • Providers: InMemoryToolRegistry (more coming)

VALIDATION FEATURES:
  • Type-safe argument validation with Pydantic
  • Return value validation
  • Configurable validation levels (strict/lenient)
  • Runtime type checking and coercion
  • Clear validation error messages
  • Validation statistics and monitoring

ADVANCED FEATURES:
  • Isolated Contexts: create_registration_manager()
  • Batch Operations: register_function_batch()
  • LangChain Support: register_langchain_tool()
  • Statistics: get_discovery_stats(), validate_all_registered_tools()
  • Validation: ValidationConfig, @with_validation, get_validation_statistics()

For detailed documentation, see individual module and function docstrings.
Use help(chuk_tool_registry.function_name) for specific help.
    """.strip()


__all__.append("get_package_help")