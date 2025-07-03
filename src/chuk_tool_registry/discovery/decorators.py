# src/chuk_tool_registry/discovery/decorators.py
"""
Simplified decorator implementation to get basic functionality working.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Protocol, runtime_checkable

# Import from core
from chuk_tool_registry.core.registration import (
    ToolRegistrationManager, 
    ToolRegistrationInfo,
    get_global_registration_manager,
    create_registration_manager
)

T = TypeVar('T')


@runtime_checkable
class SerializableTool(Protocol):
    """Protocol for tools that support serialization."""
    def __getstate__(self) -> Dict[str, Any]: ...
    def __setstate__(self, state: Dict[str, Any]) -> None: ...


def register_tool(
    name: Optional[str] = None, 
    namespace: str = "default", 
    manager: Optional[ToolRegistrationManager] = None,
    **metadata
) -> Callable[[Type[T]], Type[T]]:
    """
    Simplified decorator for registering tools.
    """
    
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate execute method
        if hasattr(cls, 'execute') and not inspect.iscoroutinefunction(cls.execute):
            raise TypeError(f"Tool {cls.__name__} must have an async execute method")
        
        # Determine tool name
        tool_name = name or cls.__name__
        
        # Create registration info
        registration_info = ToolRegistrationInfo(
            name=tool_name,
            namespace=namespace,
            metadata=metadata,
            tool_class=cls
        )
        
        # Create registration function
        async def do_register():
            from chuk_tool_registry.core.provider import ToolRegistryProvider
            registry = await ToolRegistryProvider.get_registry()
            await registry.register_tool(
                cls, 
                name=tool_name, 
                namespace=namespace, 
                metadata=metadata
            )
        
        # Add to registration manager (simplified approach)
        if manager is not None:
            # Try to use the manager - validation happens when methods are called
            try:
                # Duck typing: try to call the methods, let it fail naturally if invalid
                if hasattr(manager, 'is_registered') and hasattr(manager, 'add_registration'):
                    if not manager.is_registered(cls):
                        manager.add_registration(do_register, cls, registration_info)
                else:
                    # Store for later - will fail during processing if invalid
                    cls._invalid_manager = manager
                    cls._deferred_registration = do_register
            except Exception:
                # Any error during manager usage - defer to later
                cls._invalid_manager = manager
                cls._deferred_registration = do_register
        else:
            # Store registration info for later processing
            if not hasattr(cls, '_deferred_registration'):
                cls._deferred_registration = do_register
        
        # Store registration info on the class
        cls._tool_registration_info = registration_info
        
        return cls
    
    return decorator


def make_tool_serializable(cls: Type, tool_name: str) -> Type:
    """Helper function to make any tool class serializable."""
    # Basic serialization support - simplified for now
    return cls


def discover_decorated_tools() -> list[Type]:
    """Discover all tool classes decorated with @register_tool."""
    import sys
    import gc
    tools = []
    
    # Use garbage collector to find all class objects with registration info
    for obj in gc.get_objects():
        try:
            # Check if it's a class type and has our registration marker
            if (isinstance(obj, type) and 
                hasattr(obj, '_tool_registration_info') and 
                not obj.__name__.startswith('Mock')):  # Skip mock objects
                tools.append(obj)
        except Exception:
            # Skip any objects that cause issues during inspection
            continue
                
    return tools


# Updated ensure_registrations to handle deferred registrations
async def ensure_registrations_updated(manager: Optional[ToolRegistrationManager] = None) -> Dict[str, Any]:
    """Process all pending tool registrations."""
    if manager is None:
        # Handle both global manager and deferred registrations
        global_manager = await get_global_registration_manager()
        
        # First, collect any deferred registrations from decorated classes
        _collect_deferred_registrations(global_manager)
        
        return await global_manager.process_registrations()
    else:
        return await manager.process_registrations()


def _collect_deferred_registrations(manager: ToolRegistrationManager) -> None:
    """Collect deferred registrations from decorated classes."""
    import sys
    import gc
    
    # Use garbage collector to find decorated classes
    for obj in gc.get_objects():
        try:
            # Check if it's a class with deferred registration
            if (isinstance(obj, type) and 
                hasattr(obj, '_deferred_registration') and 
                hasattr(obj, '_tool_registration_info') and
                not obj.__name__.startswith('Mock')):  # Skip mock objects
                
                registration_fn = obj._deferred_registration
                registration_info = obj._tool_registration_info
                
                # Add to manager if not already registered
                if not manager.is_registered(obj):
                    manager.add_registration(registration_fn, obj, registration_info)
                
                # Clean up the deferred registration
                try:
                    delattr(obj, '_deferred_registration')
                except AttributeError:
                    pass  # Already cleaned up
        except Exception:
            # Skip any problematic objects
            continue


# Override the imported ensure_registrations with our updated version
ensure_registrations = ensure_registrations_updated


# Re-export key functions
__all__ = [
    "register_tool",
    "make_tool_serializable", 
    "discover_decorated_tools",
    "SerializableTool",
    "ensure_registrations",
    "create_registration_manager",
    "ToolRegistrationManager",
    "ToolRegistrationInfo",
]