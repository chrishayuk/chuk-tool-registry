# src/chuk_tool_registry/discovery/decorators.py
"""
Decorators
"""

import functools
import inspect
import asyncio
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Protocol, runtime_checkable, Union

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
) -> Callable[[Union[Type[T], Callable]], Union[Type[T], Callable]]:
    """
    Enhanced decorator for registering both functions and classes as tools.
    """
    
    def decorator(target: Union[Type[T], Callable]) -> Union[Type[T], Callable]:
        # Determine if target is a function or class
        is_function = inspect.isfunction(target) or inspect.iscoroutinefunction(target)
        is_class = inspect.isclass(target)
        
        if not (is_function or is_class):
            raise TypeError(f"@register_tool can only be used on functions or classes, got {type(target)}")
        
        # Determine tool name
        tool_name = name or target.__name__
        
        if is_function:
            # Handle function registration
            return _register_function_tool(target, tool_name, namespace, manager, metadata)
        else:
            # Handle class registration (existing logic)
            return _register_class_tool(target, tool_name, namespace, manager, metadata)
    
    return decorator


def _register_function_tool(
    func: Callable, 
    tool_name: str, 
    namespace: str, 
    manager: Optional[ToolRegistrationManager],
    metadata: Dict[str, Any]
) -> Callable:
    """Register a function as a tool."""
    
    # Create registration function
    async def do_register():
        from chuk_tool_registry.core.provider import ToolRegistryProvider
        from chuk_tool_registry.discovery.auto_register import FunctionToolWrapper
        
        # Create function wrapper
        description = metadata.get("description", func.__doc__ or f"Function tool: {tool_name}")
        wrapper = FunctionToolWrapper(func, tool_name, description)
        
        # Build complete metadata
        tool_metadata = {
            "description": description,
            "is_async": True,  # Wrapper is always async
            "source": "function_decorator",
            "source_name": func.__qualname__,
            "original_function": func.__name__,
            **metadata
        }
        
        # Register with the registry
        registry = await ToolRegistryProvider.get_registry()
        await registry.register_tool(
            wrapper,
            name=tool_name,
            namespace=namespace,
            metadata=tool_metadata
        )
    
    # Store registration info on the function
    func._tool_name = tool_name
    func._tool_namespace = namespace
    func._tool_metadata = metadata
    func._is_function_tool = True
    func._deferred_registration = do_register
    
    # Try immediate registration if event loop is available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task for immediate registration
            task = loop.create_task(do_register())
            func._registration_task = task
        else:
            # No running loop, will be processed by ensure_registrations
            pass
    except RuntimeError:
        # No event loop available, will be processed by ensure_registrations
        pass
    
    return func


def _register_class_tool(
    cls: Type[T], 
    tool_name: str, 
    namespace: str, 
    manager: Optional[ToolRegistrationManager],
    metadata: Dict[str, Any]
) -> Type[T]:
    """Register a class tool using the existing deferred registration pattern."""
    
    # Validate execute method
    if hasattr(cls, 'execute') and not inspect.iscoroutinefunction(cls.execute):
        raise TypeError(f"Tool {cls.__name__} must have an async execute method")
    
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
    
    # Add to registration manager if provided
    if manager is not None:
        try:
            if hasattr(manager, 'is_registered') and hasattr(manager, 'add_registration'):
                if not manager.is_registered(cls):
                    manager.add_registration(do_register, cls, registration_info)
            else:
                cls._invalid_manager = manager
                cls._deferred_registration = do_register
        except Exception:
            cls._invalid_manager = manager
            cls._deferred_registration = do_register
    else:
        # Store registration info for later processing
        if not hasattr(cls, '_deferred_registration'):
            cls._deferred_registration = do_register
    
    # Store registration info on the class
    cls._tool_registration_info = registration_info
    
    return cls


async def ensure_registrations_updated(manager: Optional[ToolRegistrationManager] = None) -> Dict[str, Any]:
    """Process all pending tool registrations including both classes and functions."""
    if manager is None:
        # Handle both global manager and function registrations
        global_manager = await get_global_registration_manager()
        
        # First, await any pending function registration tasks
        function_task_results = await _await_pending_function_tasks()
        
        # Then, collect any deferred registrations from decorated classes
        _collect_deferred_registrations(global_manager)
        
        # Process any remaining deferred function registrations
        function_deferred_results = await _process_deferred_function_registrations()
        
        # Process class registrations through the manager
        manager_results = await global_manager.process_registrations()
        
        # Combine results
        total_function_processed = function_task_results["processed"] + function_deferred_results["processed"]
        
        return {
            "processed": manager_results["processed"] + total_function_processed,
            "total_processed": manager_results["total_processed"] + total_function_processed,
            "pending": manager_results["pending"],
            "errors": manager_results["errors"] + function_task_results["errors"] + function_deferred_results["errors"],
            "manager": manager_results["manager"],
            "function_registrations": total_function_processed
        }
    else:
        return await manager.process_registrations()


async def _await_pending_function_tasks() -> Dict[str, Any]:
    """Await any pending function registration tasks."""
    import gc
    
    processed = 0
    errors = []
    
    # Find functions with pending registration tasks
    for obj in gc.get_objects():
        try:
            if (inspect.isfunction(obj) and 
                hasattr(obj, '_registration_task') and
                hasattr(obj, '_is_function_tool')):
                
                task = obj._registration_task
                
                if not task.done():
                    try:
                        await task
                        processed += 1
                        # Clean up the task
                        delattr(obj, '_registration_task')
                    except Exception as e:
                        errors.append(f"Function {obj.__name__} task: {str(e)}")
                elif task.done() and not task.cancelled():
                    # Task completed successfully
                    processed += 1
                    delattr(obj, '_registration_task')
                elif task.cancelled():
                    errors.append(f"Function {obj.__name__} task was cancelled")
                    
        except Exception:
            # Skip any problematic objects
            continue
    
    return {
        "processed": processed,
        "errors": errors
    }


async def _process_deferred_function_registrations() -> Dict[str, Any]:
    """Process any function registrations that were deferred."""
    import gc
    
    processed = 0
    errors = []
    
    # Find functions with deferred registrations (no task created)
    for obj in gc.get_objects():
        try:
            if (inspect.isfunction(obj) and 
                hasattr(obj, '_deferred_registration') and
                hasattr(obj, '_is_function_tool') and
                not hasattr(obj, '_registration_task')):  # Only process if no task was created
                
                registration_fn = obj._deferred_registration
                
                try:
                    await registration_fn()
                    processed += 1
                    # Clean up the deferred registration
                    delattr(obj, '_deferred_registration')
                except Exception as e:
                    errors.append(f"Function {obj.__name__}: {str(e)}")
        except Exception:
            # Skip any problematic objects
            continue
    
    return {
        "processed": processed,
        "errors": errors
    }


def _collect_deferred_registrations(manager: ToolRegistrationManager) -> None:
    """Collect deferred registrations from decorated classes."""
    import gc
    
    # Use garbage collector to find decorated classes
    for obj in gc.get_objects():
        try:
            # Check if it's a class with deferred registration
            if (inspect.isclass(obj) and 
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


def make_tool_serializable(cls: Type, tool_name: str) -> Type:
    """Helper function to make any tool class serializable."""
    return cls


def discover_decorated_tools() -> list[Type]:
    """Discover all tool classes decorated with @register_tool."""
    import gc
    tools = []
    
    for obj in gc.get_objects():
        try:
            if (inspect.isclass(obj) and 
                hasattr(obj, '_tool_registration_info') and 
                not obj.__name__.startswith('Mock')):
                tools.append(obj)
        except Exception:
            continue
                
    return tools


def discover_decorated_functions() -> list[Callable]:
    """Discover all functions decorated with @register_tool."""
    import gc
    functions = []
    
    for obj in gc.get_objects():
        try:
            if (inspect.isfunction(obj) and 
                hasattr(obj, '_is_function_tool') and 
                not obj.__name__.startswith('Mock')):
                functions.append(obj)
        except Exception:
            continue
                
    return functions


# Override the imported ensure_registrations with our updated version
ensure_registrations = ensure_registrations_updated


# Re-export key functions
__all__ = [
    "register_tool",
    "make_tool_serializable", 
    "discover_decorated_tools",
    "discover_decorated_functions",
    "SerializableTool",
    "ensure_registrations",
    "create_registration_manager",
    "ToolRegistrationManager",
    "ToolRegistrationInfo",
]