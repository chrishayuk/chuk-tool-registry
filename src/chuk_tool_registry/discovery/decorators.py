# src/chuk_tool_registry/discovery/decorators.py
"""
Simplified decorator implementation to get basic functionality working.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Protocol

# Import from core
from chuk_tool_registry.core.registration import (
    ToolRegistrationManager, 
    ToolRegistrationInfo,
    get_global_registration_manager,
    create_registration_manager
)

T = TypeVar('T')


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
            # Use provided manager directly
            if not manager.is_registered(cls):
                manager.add_registration(do_register, cls, registration_info)
        else:
            # Store registration info for later processing
            if not hasattr(cls, '_deferred_registration'):
                cls._deferred_registration = (do_register, registration_info)
        
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
    tools = []
    
    for module_name, module in list(sys.modules.items()):
        if not hasattr(module, '__dict__'):
            continue
            
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if hasattr(attr, '_tool_registration_info'):
                    tools.append(attr)
            except (AttributeError, ImportError):
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
    
    for module_name, module in list(sys.modules.items()):
        if not hasattr(module, '__dict__'):
            continue
            
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if hasattr(attr, '_deferred_registration') and hasattr(attr, '_tool_registration_info'):
                    registration_fn, registration_info = attr._deferred_registration
                    
                    # Add to manager if not already registered
                    if not manager.is_registered(attr):
                        manager.add_registration(registration_fn, attr, registration_info)
                    
                    # Clean up the deferred registration
                    delattr(attr, '_deferred_registration')
                    
            except (AttributeError, ImportError):
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