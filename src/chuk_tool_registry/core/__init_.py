# chuk_tool_registry/core/__init__.py
"""
Core components for the async-native tool registry.

This module provides the foundational components for the chuk_tool_registry:
- Tool metadata models
- Registry interface definitions  
- Exception classes
- Global registry provider access

Example usage:
    >>> import asyncio
    >>> from chuk_tool_registry.core import get_registry, ToolMetadata
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
"""

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
)

# Version information
__version__ = "1.0.0"

# Public API - these are the main exports that users should import
__all__ = [
    # Core interface
    "ToolRegistryInterface",
    
    # Metadata models
    "ToolMetadata",
    "RateLimitConfig", 
    "StreamingToolMetadata",
    
    # Exceptions
    "ToolProcessorError",
    "ToolNotFoundError", 
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    "ParserError",
    
    # Global registry access
    "get_registry",
    "set_registry", 
    "ToolRegistryProvider",
    
    # Version
    "__version__",
]