# chuk_tool_registry/discovery/__init__.py
"""
Tool discovery and auto-registration for the async-native tool registry.

This module provides convenient ways to automatically register tools:
- Decorators for class-based tools
- Auto-registration functions for plain functions and LangChain tools
- Pydantic model compatibility helpers
- Discovery utilities for finding decorated tools

Example usage:
    >>> import asyncio
    >>> from chuk_tool_registry.discovery import register_tool, register_fn_tool
    >>> 
    >>> # Class-based tool with decorator
    >>> @register_tool("my_calculator")
    ... class Calculator:
    ...     async def execute(self, operation: str, a: int, b: int) -> int:
    ...         if operation == "add":
    ...             return a + b
    ...         elif operation == "multiply":
    ...             return a * b
    ...         else:
    ...             raise ValueError(f"Unknown operation: {operation}")
    >>> 
    >>> # Function-based tool
    >>> async def greet(name: str) -> str:
    ...     return f"Hello, {name}!"
    >>> 
    >>> async def setup_tools():
    ...     # Register the function
    ...     await register_fn_tool(greet, name="greeter")
    ...     
    ...     # Ensure decorated tools are registered
    ...     await ensure_registrations()
    >>> 
    >>> asyncio.run(setup_tools())

For LangChain integration:
    >>> from chuk_tool_registry.discovery import register_langchain_tool
    >>> 
    >>> # Register a LangChain tool
    >>> async def register_langchain_tools():
    ...     # Assuming you have a LangChain tool instance
    ...     await register_langchain_tool(my_langchain_tool)
"""

# Core decorator and registration functions
from .decorators import (
    register_tool,
    ensure_registrations,
    discover_decorated_tools,
    make_pydantic_tool_compatible,
)

# Auto-registration functions
from .auto_register import (
    register_fn_tool,
    register_langchain_tool,
)

# Version information
__version__ = "1.0.0"

# Public API - these are the main exports that users should import
__all__ = [
    # Decorators and class registration
    "register_tool",
    "ensure_registrations", 
    "discover_decorated_tools",
    "make_pydantic_tool_compatible",
    
    # Function and external tool registration
    "register_fn_tool",
    "register_langchain_tool",
    
    # Version
    "__version__",
]


# Convenience function to register all pending tools
register_all_pending = ensure_registrations


# Add the convenience function to exports
__all__.append("register_all_pending")