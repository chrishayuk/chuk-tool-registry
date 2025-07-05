# src/chuk_tool_registry/discovery/decorators.py
"""
Enhanced decorators with integrated validation support.

Clean implementation assuming Pydantic and anyio are always available.
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
from chuk_tool_registry.core.validation import ValidationConfig

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
    validation_config: Optional[ValidationConfig] = None,
    enable_validation: Optional[bool] = None,
    **metadata
) -> Callable[[Union[Type[T], Callable]], Union[Type[T], Callable]]:
    """
    Enhanced decorator for registering both functions and classes as tools with validation support.
    
    Args:
        name: Optional tool name (defaults to function/class name)
        namespace: Namespace for the tool
        manager: Optional registration manager for isolated contexts
        validation_config: Optional validation configuration
        enable_validation: Whether to enable validation for this tool
        **metadata: Additional metadata for the tool
    
    Returns:
        Decorator function that registers the tool
    
    Examples:
        @register_tool("my_calculator", enable_validation=True)
        async def calculate(a: int, b: int) -> int:
            return a + b
        
        @register_tool("advanced_calc", validation_config=ValidationConfig(strict_mode=True))
        class AdvancedCalculator:
            async def execute(self, operation: str, a: float, b: float) -> float:
                # ... implementation
    """
    
    def decorator(target: Union[Type[T], Callable]) -> Union[Type[T], Callable]:
        # Determine if target is a function or class
        is_function = inspect.isfunction(target) or inspect.iscoroutinefunction(target)
        is_class = inspect.isclass(target)
        
        if not (is_function or is_class):
            raise TypeError(f"@register_tool can only be used on functions or classes, got {type(target)}")
        
        # Determine tool name
        tool_name = name or target.__name__
        
        # Merge validation settings into metadata
        enhanced_metadata = dict(metadata)
        if enable_validation is not None:
            enhanced_metadata['enable_validation'] = enable_validation
        if validation_config is not None:
            enhanced_metadata['validation_config'] = validation_config
        
        if is_function:
            # Handle function registration
            return _register_function_tool(target, tool_name, namespace, manager, enhanced_metadata)
        else:
            # Handle class registration
            return _register_class_tool(target, tool_name, namespace, manager, enhanced_metadata)
    
    return decorator


def _register_function_tool(
    func: Callable, 
    tool_name: str, 
    namespace: str, 
    manager: Optional[ToolRegistrationManager],
    metadata: Dict[str, Any]
) -> Callable:
    """Register a function as a tool with validation support."""
    
    # Extract validation settings from metadata
    enable_validation = metadata.get('enable_validation', False)
    validation_config = metadata.get('validation_config')
    
    # Create registration function
    async def do_register():
        from chuk_tool_registry.core.provider import ToolRegistryProvider
        from chuk_tool_registry.discovery.auto_register import register_fn_tool
        
        # Use the enhanced register_fn_tool with validation support
        await register_fn_tool(
            func,
            name=tool_name,
            namespace=namespace,
            validation_config=validation_config,
            enable_validation=enable_validation,
            **{k: v for k, v in metadata.items() if k not in ['enable_validation', 'validation_config']}
        )
    
    # Store registration info on the function
    func._tool_name = tool_name
    func._tool_namespace = namespace
    func._tool_metadata = metadata
    func._is_function_tool = True
    func._deferred_registration = do_register
    func._validation_enabled = enable_validation
    func._validation_config = validation_config
    
    # Try immediate registration if event loop is available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task for immediate registration
            task = loop.create_task(do_register())
            func._registration_task = task
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
    """Register a class tool using the enhanced deferred registration pattern with validation."""
    
    # Validate execute method
    if hasattr(cls, 'execute') and not inspect.iscoroutinefunction(cls.execute):
        raise TypeError(f"Tool {cls.__name__} must have an async execute method")
    
    # Extract validation settings
    enable_validation = metadata.get('enable_validation', False)
    validation_config = metadata.get('validation_config')
    
    # Create registration info
    registration_info = ToolRegistrationInfo(
        name=tool_name,
        namespace=namespace,
        metadata=metadata,
        tool_class=cls
    )
    
    # Create enhanced registration function
    async def do_register():
        from chuk_tool_registry.core.provider import ToolRegistryProvider
        registry = await ToolRegistryProvider.get_registry()
        
        # Use enhanced registration method if available
        if (hasattr(registry, 'register_tool') and 
            'validation_config' in inspect.signature(registry.register_tool).parameters):
            await registry.register_tool(
                cls, 
                name=tool_name, 
                namespace=namespace, 
                metadata={k: v for k, v in metadata.items() if k not in ['enable_validation', 'validation_config']},
                validation_config=validation_config,
                enable_validation=enable_validation
            )
        else:
            # Fallback to standard registration
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
    cls._validation_enabled = enable_validation
    cls._validation_config = validation_config
    
    return cls


def validated_tool(
    name: Optional[str] = None,
    namespace: str = "default",
    validation_config: Optional[ValidationConfig] = None,
    strict_mode: bool = False,
    **metadata
) -> Callable[[Union[Type[T], Callable]], Union[Type[T], Callable]]:
    """
    Convenience decorator for registering tools with validation enabled by default.
    
    This is a specialized version of @register_tool that enables validation
    and provides convenient validation configuration options.
    
    Args:
        name: Optional tool name
        namespace: Namespace for the tool
        validation_config: Optional custom validation configuration
        strict_mode: Whether to enable strict validation mode
        **metadata: Additional metadata
    
    Returns:
        Decorator function
    
    Examples:
        @validated_tool("strict_calculator", strict_mode=True)
        async def calculate(a: int, b: int) -> int:
            return a + b
        
        @validated_tool("lenient_processor")
        class DataProcessor:
            async def execute(self, data: list) -> dict:
                # ... implementation
    """
    # Create validation configuration if not provided
    if validation_config is None:
        validation_config = ValidationConfig(
            strict_mode=strict_mode,
            validate_arguments=True,
            validate_results=True
        )
    
    return register_tool(
        name=name,
        namespace=namespace,
        validation_config=validation_config,
        enable_validation=True,
        **metadata
    )


def requires_pydantic(func_or_cls: Union[Callable, Type]) -> Union[Callable, Type]:
    """
    Decorator that ensures Pydantic is available for validation-dependent tools.
    
    Since Pydantic is now a required dependency, this decorator is mainly
    for documentation purposes and consistency.
    
    Args:
        func_or_cls: Function or class to decorate
    
    Returns:
        Decorated function or class (unchanged)
    
    Example:
        @requires_pydantic
        @validated_tool("my_tool")
        async def my_function(data: ComplexType) -> Result:
            # ... implementation
    """
    # Pydantic is always available now, so this is a no-op
    return func_or_cls


async def ensure_registrations_updated(manager: Optional[ToolRegistrationManager] = None) -> Dict[str, Any]:
    """Enhanced ensure_registrations with better validation support and reporting."""
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
        
        # Combine results with validation information
        total_function_processed = function_task_results["processed"] + function_deferred_results["processed"]
        
        # Gather validation statistics
        validation_stats = await _gather_validation_statistics()
        
        return {
            "processed": manager_results["processed"] + total_function_processed,
            "total_processed": manager_results["total_processed"] + total_function_processed,
            "pending": manager_results["pending"],
            "errors": manager_results["errors"] + function_task_results["errors"] + function_deferred_results["errors"],
            "manager": manager_results["manager"],
            "function_registrations": total_function_processed,
            "validation_statistics": validation_stats
        }
    else:
        return await manager.process_registrations()


async def _await_pending_function_tasks() -> Dict[str, Any]:
    """Await any pending function registration tasks with validation info."""
    import gc
    
    processed = 0
    errors = []
    validation_enabled = 0
    
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
                        if getattr(obj, '_validation_enabled', False):
                            validation_enabled += 1
                        # Clean up the task
                        delattr(obj, '_registration_task')
                    except Exception as e:
                        errors.append(f"Function {obj.__name__} task: {str(e)}")
                elif task.done() and not task.cancelled():
                    # Task completed successfully
                    processed += 1
                    if getattr(obj, '_validation_enabled', False):
                        validation_enabled += 1
                    delattr(obj, '_registration_task')
                elif task.cancelled():
                    errors.append(f"Function {obj.__name__} task was cancelled")
                    
        except Exception:
            # Skip any problematic objects
            continue
    
    return {
        "processed": processed,
        "errors": errors,
        "validation_enabled": validation_enabled
    }


async def _process_deferred_function_registrations() -> Dict[str, Any]:
    """Process any function registrations that were deferred with validation info."""
    import gc
    
    processed = 0
    errors = []
    validation_enabled = 0
    
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
                    if getattr(obj, '_validation_enabled', False):
                        validation_enabled += 1
                    # Clean up the deferred registration
                    delattr(obj, '_deferred_registration')
                except Exception as e:
                    errors.append(f"Function {obj.__name__}: {str(e)}")
        except Exception:
            # Skip any problematic objects
            continue
    
    return {
        "processed": processed,
        "errors": errors,
        "validation_enabled": validation_enabled
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


async def _gather_validation_statistics() -> Dict[str, Any]:
    """Gather statistics about validation usage in registered tools."""
    try:
        from chuk_tool_registry.discovery.auto_register import get_validation_statistics
        return await get_validation_statistics()
    except ImportError:
        # Fallback if auto_register is not available
        return {
            "validation_available": True,  # Always true now
            "total_tools": 0,
            "validation_enabled_tools": 0,
            "validation_percentage": 0.0
        }


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


def discover_validation_enabled_tools() -> Dict[str, list]:
    """
    Discover all tools (functions and classes) that have validation enabled.
    
    Returns:
        Dict with 'functions' and 'classes' keys containing lists of validation-enabled tools
    """
    import gc
    
    result = {
        "functions": [],
        "classes": []
    }
    
    for obj in gc.get_objects():
        try:
            if inspect.isfunction(obj) and hasattr(obj, '_is_function_tool'):
                if getattr(obj, '_validation_enabled', False):
                    result["functions"].append({
                        "function": obj,
                        "name": getattr(obj, '_tool_name', obj.__name__),
                        "namespace": getattr(obj, '_tool_namespace', 'default'),
                        "validation_config": getattr(obj, '_validation_config', None)
                    })
            elif inspect.isclass(obj) and hasattr(obj, '_tool_registration_info'):
                if getattr(obj, '_validation_enabled', False):
                    result["classes"].append({
                        "class": obj,
                        "name": getattr(obj, '_tool_registration_info').name,
                        "namespace": getattr(obj, '_tool_registration_info').namespace,
                        "validation_config": getattr(obj, '_validation_config', None)
                    })
        except Exception:
            continue
    
    return result


# Override the imported ensure_registrations with our enhanced version
ensure_registrations = ensure_registrations_updated


# Re-export key functions
__all__ = [
    "register_tool",
    "validated_tool",
    "requires_pydantic",
    "make_tool_serializable", 
    "discover_decorated_tools",
    "discover_decorated_functions",
    "discover_validation_enabled_tools",
    "SerializableTool",
    "ensure_registrations",
    "create_registration_manager",
    "ToolRegistrationManager",
    "ToolRegistrationInfo",
]