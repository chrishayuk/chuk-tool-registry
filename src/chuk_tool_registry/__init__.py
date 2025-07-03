# src/chuk_tool_registry/__init__.py
"""
Chuk Tool Registry - Async-native tool registration and discovery system.
"""

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
)

# Registration management (newly exposed) - NOTE: ensure_registrations NOT imported here
from .core.registration import (
    ToolRegistrationManager,
    ToolRegistrationInfo,
    create_registration_manager,
    get_global_registration_manager,
    # ensure_registrations,  # REMOVED - using enhanced version from discovery
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

# Provider implementations
from .providers import (
    get_registry as get_provider_registry,
    clear_registry_cache,
)

# Version information
__version__ = "2.0.0"
__author__ = "Chuk Development Team"
__description__ = "Async-native tool registration and discovery system"

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
    "register_tool",           # Class decorator
    "register_fn_tool",        # Function registration
    "register_langchain_tool", # LangChain integration
    "register_function_batch", # Batch operations
    "register_module_functions",
    "quick_register_function",
    
    # === DISCOVERY AND UTILITIES ===
    "discover_decorated_tools",
    "make_tool_serializable",
    "validate_tool_function",
    "get_registered_function_tools",
    "get_discovery_stats",
    "list_tools_by_source",
    "validate_all_registered_tools",
    
    # === METADATA MODELS ===
    "ToolMetadata",
    "RateLimitConfig", 
    "StreamingToolMetadata",
    
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


# Package-level convenience functions
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


async def register_and_setup(*tools_or_functions, namespace: str = "default") -> dict:
    """
    Convenience function to register multiple tools and set up the registry.
    
    Args:
        *tools_or_functions: Tool classes or functions to register
        namespace: Namespace for all tools
        
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
        >>> results = await register_and_setup(my_func, Calculator, namespace="demo")
    """
    results = {"functions": [], "classes": [], "errors": []}
    
    for item in tools_or_functions:
        try:
            if callable(item) and not hasattr(item, '__init__'):
                # It's a function
                await register_fn_tool(item, namespace=namespace)
                results["functions"].append(item.__name__)
            elif hasattr(item, '_tool_registration_info'):
                # It's already decorated
                results["classes"].append(item.__name__)
            else:
                # Try to register as a tool class
                enhanced_item = register_tool(namespace=namespace)(item)
                results["classes"].append(enhanced_item.__name__)
        except Exception as e:
            results["errors"].append(f"{getattr(item, '__name__', str(item))}: {e}")
    
    # Process any pending registrations
    await ensure_registrations()
    
    return results


async def get_all_tools(namespace: str = None) -> dict:
    """
    Get information about all registered tools.
    
    Args:
        namespace: Optional namespace filter
        
    Returns:
        Dict with comprehensive tool information
        
    Example:
        >>> tools_info = await get_all_tools()
        >>> for ns, tools in tools_info["by_namespace"].items():
        ...     print(f"{ns}: {len(tools)} tools")
    """
    registry = await get_registry()
    
    # Get basic tool list
    all_tools = await registry.list_tools(namespace)
    
    # Get all metadata
    all_metadata = await registry.list_metadata(namespace)
    
    # Organize by namespace
    by_namespace = {}
    for ns, name in all_tools:
        if ns not in by_namespace:
            by_namespace[ns] = []
        by_namespace[ns].append(name)
    
    # Get source breakdown
    source_breakdown = {}
    for metadata in all_metadata:
        source = getattr(metadata, 'source', 'unknown')
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
    
    return {
        "total_tools": len(all_tools),
        "namespaces": list(by_namespace.keys()),
        "by_namespace": by_namespace,
        "source_breakdown": source_breakdown,
        "metadata": {m.name: m for m in all_metadata}
    }


# Add convenience functions to exports
__all__.extend([
    "setup_registry",
    "register_and_setup", 
    "get_all_tools",
])


# Package initialization message (only in debug mode)
def _debug_info():
    """Print debug information about the package."""
    import os
    if os.environ.get('CHUK_TOOL_REGISTRY_DEBUG'):
        print(f"🔧 Chuk Tool Registry v{__version__} initialized")
        print(f"   Available providers: memory")
        print(f"   Core components: {len([x for x in __all__ if not x.startswith('_')])} exports")


# Only show debug info if explicitly requested
_debug_info()


# Documentation helper
def get_package_help() -> str:
    """
    Get comprehensive help for the chuk_tool_registry package.
    
    Returns:
        Formatted help text with examples and API overview
    """
    return f"""
Chuk Tool Registry v{__version__}
================================

QUICK START:
  1. Decorate tool classes:
     @register_tool("my_tool", namespace="utilities")
     class MyTool:
         async def execute(self, x: int) -> int:
             return x * 2

  2. Register functions:
     await register_fn_tool(my_function, name="my_func")

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
  • Providers: InMemoryToolRegistry (more coming)

ADVANCED FEATURES:
  • Isolated Contexts: create_registration_manager()
  • Batch Operations: register_function_batch()
  • LangChain Support: register_langchain_tool()
  • Statistics: get_discovery_stats(), validate_all_registered_tools()
  • Pydantic Support: Automatic compatibility with Pydantic models

For detailed documentation, see individual module and function docstrings.
Use help(chuk_tool_registry.function_name) for specific help.
    """.strip()


__all__.append("get_package_help")