# src/chuk_tool_registry/discovery/__init__.py
"""
Tool discovery and auto-registration for the async-native tool registry.

This module provides convenient ways to automatically register tools with
enhanced type safety, better error handling, and reduced global state.
"""

# Core decorator and registration functions
from .decorators import (
    register_tool,
    ensure_registrations,
    discover_decorated_tools,
    make_tool_serializable,
    create_registration_manager,
    ToolRegistrationManager,
    ToolRegistrationInfo,
    SerializableTool,
)

# Auto-registration functions
from .auto_register import (
    register_fn_tool,
    register_langchain_tool,
    register_function_batch,
    register_module_functions,
    validate_tool_function,
    get_registered_function_tools,
    FunctionToolWrapper,
    LangChainToolWrapper,
    SchemaGenerationError,
    ToolRegistrationError,
)

# Version information
__version__ = "2.0.0"

# Public API - these are the main exports that users should import
__all__ = [
    # Core decorators and class registration
    "register_tool",
    "ensure_registrations", 
    "discover_decorated_tools",
    "make_tool_serializable",
    
    # Registration management
    "create_registration_manager",
    "ToolRegistrationManager",
    "ToolRegistrationInfo",
    "SerializableTool",
    
    # Function and external tool registration
    "register_fn_tool",
    "register_langchain_tool",
    "register_function_batch",
    "register_module_functions",
    
    # Utility functions
    "validate_tool_function",
    "get_registered_function_tools",
    
    # Tool wrapper classes
    "FunctionToolWrapper",
    "LangChainToolWrapper",
    
    # Exceptions
    "SchemaGenerationError",
    "ToolRegistrationError",
    
    # Version
    "__version__",
]


# Convenience functions for backward compatibility
register_all_pending = ensure_registrations

# Add backward compatibility exports
__all__.extend([
    "register_all_pending",
])


# Convenience function for quick function registration
async def quick_register_function(
    func,
    name: str = None,
    description: str = None,
    namespace: str = "default"
) -> None:
    """
    Quick registration function for interactive use.
    
    Args:
        func: Function to register
        name: Optional tool name
        description: Optional description
        namespace: Namespace for the tool
    """
    await register_fn_tool(
        func,
        name=name,
        description=description,
        namespace=namespace
    )


# Add convenience functions to exports
__all__.extend([
    "quick_register_function",
])


# Registry interaction helpers
async def get_discovery_stats() -> dict:
    """
    Get statistics about discovered and registered tools.
    
    Returns:
        Dict with discovery and registration statistics
    """
    from chuk_tool_registry.core.provider import ToolRegistryProvider
    
    registry = await ToolRegistryProvider.get_registry()
    
    # Get basic registry stats
    if hasattr(registry, 'get_statistics'):
        registry_stats = await registry.get_statistics()
    else:
        # Fallback for registries without statistics
        all_tools = await registry.list_tools()
        namespaces = await registry.list_namespaces()
        registry_stats = {
            "total_tools": len(all_tools),
            "total_namespaces": len(namespaces),
            "namespaces": {ns: len([t for t in all_tools if t[0] == ns]) for ns in namespaces}
        }
    
    # Get metadata for source analysis
    all_metadata = await registry.list_metadata()
    
    source_breakdown = {}
    for metadata in all_metadata:
        source = getattr(metadata, 'source', 'unknown')
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
    
    return {
        **registry_stats,
        "source_breakdown": source_breakdown,
        "discovery_version": __version__,
    }


async def list_tools_by_source(source: str, namespace: str = None) -> list:
    """
    List all tools registered from a specific source.
    
    Args:
        source: Source type to filter by ("function", "langchain", "class", etc.)
        namespace: Optional namespace filter
        
    Returns:
        List of ToolMetadata objects matching the source
    """
    from chuk_tool_registry.core.provider import ToolRegistryProvider
    
    registry = await ToolRegistryProvider.get_registry()
    all_metadata = await registry.list_metadata(namespace)
    
    return [
        metadata for metadata in all_metadata 
        if getattr(metadata, 'source', None) == source
    ]


async def validate_all_registered_tools() -> dict:
    """
    Validate all currently registered tools.
    
    Returns:
        Dict with validation results
    """
    from chuk_tool_registry.core.provider import ToolRegistryProvider
    
    registry = await ToolRegistryProvider.get_registry()
    all_tools = await registry.list_tools()
    
    valid_tools = []
    invalid_tools = []
    validation_errors = {}
    
    for namespace, tool_name in all_tools:
        try:
            tool = await registry.get_tool(tool_name, namespace)
            if tool is None:
                invalid_tools.append((namespace, tool_name))
                validation_errors[f"{namespace}.{tool_name}"] = "Tool is None"
                continue
                
            # Check if tool has execute method
            if not hasattr(tool, 'execute'):
                invalid_tools.append((namespace, tool_name))
                validation_errors[f"{namespace}.{tool_name}"] = "Missing execute method"
                continue
                
            if not callable(getattr(tool, 'execute')):
                invalid_tools.append((namespace, tool_name))
                validation_errors[f"{namespace}.{tool_name}"] = "execute is not callable"
                continue
                
            valid_tools.append((namespace, tool_name))
            
        except Exception as e:
            invalid_tools.append((namespace, tool_name))
            validation_errors[f"{namespace}.{tool_name}"] = str(e)
    
    return {
        "total_checked": len(all_tools),
        "valid_tools": valid_tools,
        "invalid_tools": invalid_tools,
        "validation_errors": validation_errors,
        "success_rate": len(valid_tools) / len(all_tools) if all_tools else 1.0,
    }


# Add the new functions to exports
__all__.extend([
    "get_discovery_stats",
    "list_tools_by_source", 
    "validate_all_registered_tools",
])