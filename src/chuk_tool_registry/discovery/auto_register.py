# src/chuk_tool_registry/discovery/auto_register.py
"""
Async auto-register helpers for functions and LangChain tools.

Key improvements:
1. Better type safety and error handling
2. Simplified schema generation
3. Improved LangChain integration
4. Cleaner code organization
"""

from __future__ import annotations

import asyncio
import inspect
import types
from typing import (
    Callable, ForwardRef, Type, get_type_hints, Any, Optional, Dict, 
    Union, TypeVar, Protocol, runtime_checkable
)

try:
    import anyio
except ImportError:
    # Fallback if anyio is not available
    anyio = None

from pydantic import BaseModel, create_model, Field

try:  # optional dependency
    from langchain.tools.base import BaseTool  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    BaseTool = None

# registry
from chuk_tool_registry.core.provider import ToolRegistryProvider
from .decorators import register_tool

# Type variables
F = TypeVar('F', bound=Callable)
T = TypeVar('T')


@runtime_checkable
class ExecutableTool(Protocol):
    """Protocol for objects that can be executed as tools."""
    async def execute(self, **kwargs: Any) -> Any: ...


class SchemaGenerationError(Exception):
    """Raised when schema generation fails."""
    pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


# ────────────────────────────────────────────────────────────────────────────
# Schema generation utilities
# ────────────────────────────────────────────────────────────────────────────

def _resolve_type_hint(hint: Any, default_type: Type = str) -> Type:
    """
    Resolve a type hint to a concrete type, with fallbacks for problematic cases.
    
    Args:
        hint: The type hint to resolve
        default_type: Type to use as fallback
        
    Returns:
        A concrete type that can be used with Pydantic
    """
    # Handle empty/missing annotations
    if hint in (inspect._empty, None):
        return default_type
    
    # Handle string annotations and forward references
    if isinstance(hint, (str, ForwardRef)):
        return default_type
    
    # Handle Union types (including Optional)
    if hasattr(hint, '__origin__') and hint.__origin__ is Union:
        # For Optional[T] (Union[T, None]), extract T
        args = hint.__args__
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return non_none_args[0]
        return default_type
    
    # Return the hint if it's a valid type
    try:
        if isinstance(hint, type):
            return hint
    except TypeError:
        pass
    
    return default_type


def _create_function_schema(func: Callable) -> Type[BaseModel]:
    """
    Create a Pydantic model schema from a function signature.
    
    Args:
        func: Function to analyze
        
    Returns:
        Pydantic model class for the function's parameters
        
    Raises:
        SchemaGenerationError: If schema creation fails
    """
    try:
        # Get type hints, with fallback for problematic annotations
        try:
            hints = get_type_hints(func)
        except (NameError, AttributeError, TypeError) as e:
            # Fall back to raw annotations if get_type_hints fails
            hints = {}
            if hasattr(func, '__annotations__'):
                for name, annotation in func.__annotations__.items():
                    if name != 'return':
                        hints[name] = _resolve_type_hint(annotation)
    
        # Build Pydantic fields from function signature
        fields: Dict[str, tuple] = {}
        signature = inspect.signature(func)
        
        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # Resolve the type hint
            hint = hints.get(param_name, param.annotation)
            resolved_type = _resolve_type_hint(hint)
            
            # Determine if the parameter is required
            if param.default is inspect.Parameter.empty:
                # Required parameter
                fields[param_name] = (resolved_type, Field(..., description=f"Parameter {param_name}"))
            else:
                # Optional parameter with default
                fields[param_name] = (resolved_type, Field(default=param.default, description=f"Parameter {param_name}"))
        
        # Create the model
        model_name = f"{func.__name__.title()}Schema"
        return create_model(model_name, **fields)
    
    except Exception as e:
        raise SchemaGenerationError(f"Failed to create schema for {func.__name__}: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
# Tool wrapper classes
# ────────────────────────────────────────────────────────────────────────────

class FunctionToolWrapper:
    """
    Wrapper that adapts a function to the tool interface.
    
    This handles both sync and async functions uniformly.
    """
    
    def __init__(self, func: Callable, name: str, description: str):
        self.func = func
        self.name = name
        self.description = description
        self.is_async = inspect.iscoroutinefunction(func)
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function."""
        if self.is_async:
            return await self.func(**kwargs)
        else:
            # Run sync function in thread pool to avoid blocking if anyio is available
            if anyio is not None:
                # Create a wrapper function that takes no args and calls func with kwargs
                def wrapper():
                    return self.func(**kwargs)
                return await anyio.to_thread.run_sync(wrapper)
            else:
                # Fallback: run directly (not ideal but works for testing)
                return self.func(**kwargs)
    
    def __str__(self) -> str:
        return f"FunctionTool({self.name})"


class LangChainToolWrapper:
    """
    Wrapper that adapts a LangChain tool to the tool interface.
    """
    
    def __init__(self, langchain_tool: Any, name: str, description: str):
        self.langchain_tool = langchain_tool
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the LangChain tool."""
        # Prefer async method if available
        if hasattr(self.langchain_tool, 'arun') and callable(self.langchain_tool.arun):
            return await self.langchain_tool.arun(**kwargs)
        elif hasattr(self.langchain_tool, 'run') and callable(self.langchain_tool.run):
            # Run sync method in thread pool if anyio is available
            if anyio is not None:
                def wrapper():
                    return self.langchain_tool.run(**kwargs)
                return await anyio.to_thread.run_sync(wrapper)
            else:
                # Fallback: run directly
                return self.langchain_tool.run(**kwargs)
        else:
            raise AttributeError(f"LangChain tool {self.langchain_tool} has no run or arun method")
    
    def __str__(self) -> str:
        return f"LangChainTool({self.name})"


# ────────────────────────────────────────────────────────────────────────────
# Registration functions
# ────────────────────────────────────────────────────────────────────────────

async def register_fn_tool(
    func: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    namespace: str = "default",
    include_schema: bool = True,
    **metadata: Any
) -> None:
    """
    Register a plain function as a tool.
    
    Args:
        func: The function to register (can be sync or async)
        name: Optional name for the tool (defaults to function name)
        description: Optional description (defaults to function docstring)
        namespace: Registry namespace (defaults to "default")
        include_schema: Whether to generate and include argument schema
        **metadata: Additional metadata for the tool
        
    Raises:
        ToolRegistrationError: If registration fails
    """
    if not callable(func):
        raise ToolRegistrationError(f"Expected callable, got {type(func)}")
    
    try:
        # Determine tool properties
        tool_name = name or func.__name__
        tool_description = description or inspect.getdoc(func) or f"Function tool: {tool_name}"
        
        # Create the tool wrapper
        tool_wrapper = FunctionToolWrapper(func, tool_name, tool_description)
        
        # Build metadata
        tool_metadata = {
            "description": tool_description,
            "is_async": True,  # Wrapper is always async
            "source": "function",
            "source_name": func.__qualname__,
            "original_function": func.__name__,
            **metadata
        }
        
        # Add schema if requested
        if include_schema:
            try:
                schema = _create_function_schema(func)
                tool_metadata["argument_schema"] = schema.model_json_schema()
            except SchemaGenerationError as e:
                # Log warning but continue without schema
                tool_metadata["schema_generation_error"] = str(e)
        
        # Register with the registry
        registry = await ToolRegistryProvider.get_registry()
        await registry.register_tool(
            tool_wrapper,
            name=tool_name,
            namespace=namespace,
            metadata=tool_metadata
        )
        
    except Exception as e:
        raise ToolRegistrationError(f"Failed to register function {func.__name__}: {e}") from e


async def register_langchain_tool(
    tool: Any,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    namespace: str = "default",
    **metadata: Any
) -> None:
    """
    Register a LangChain BaseTool instance.
    
    Args:
        tool: The LangChain tool to register
        name: Optional name for the tool (defaults to tool.name)
        description: Optional description (defaults to tool.description)
        namespace: Registry namespace (defaults to "default")
        **metadata: Additional metadata for the tool
        
    Raises:
        RuntimeError: If LangChain isn't installed
        TypeError: If the object isn't a LangChain BaseTool
        ToolRegistrationError: If registration fails
    """
    if BaseTool is None:
        raise RuntimeError(
            "LangChain is required for register_langchain_tool(). "
            "Install with: pip install langchain"
        )

    if not isinstance(tool, BaseTool):
        raise TypeError(
            f"Expected langchain.tools.base.BaseTool, got {type(tool).__name__}"
        )

    try:
        # Extract tool properties
        tool_name = name or getattr(tool, 'name', None) or tool.__class__.__name__
        tool_description = (
            description or 
            getattr(tool, 'description', None) or 
            inspect.getdoc(tool) or 
            f"LangChain tool: {tool_name}"
        )
        
        # Create wrapper
        tool_wrapper = LangChainToolWrapper(tool, tool_name, tool_description)
        
        # Build metadata
        tool_metadata = {
            "description": tool_description,
            "is_async": True,  # Wrapper is always async
            "source": "langchain",
            "source_name": tool.__class__.__qualname__,
            "langchain_tool_type": type(tool).__name__,
            **metadata
        }
        
        # Add LangChain-specific metadata if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                tool_metadata["argument_schema"] = tool.args_schema.model_json_schema()
            except Exception:
                # Schema extraction failed, continue without it
                pass
        
        # Register with the registry
        registry = await ToolRegistryProvider.get_registry()
        await registry.register_tool(
            tool_wrapper,
            name=tool_name,
            namespace=namespace,
            metadata=tool_metadata
        )
        
    except Exception as e:
        raise ToolRegistrationError(f"Failed to register LangChain tool {tool}: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
# Batch registration utilities
# ────────────────────────────────────────────────────────────────────────────

async def register_function_batch(
    functions: Dict[str, Callable],
    *,
    namespace: str = "default",
    description_prefix: str = "",
    **common_metadata: Any
) -> Dict[str, bool]:
    """
    Register multiple functions as tools in batch.
    
    Args:
        functions: Dict mapping tool names to functions
        namespace: Namespace for all tools
        description_prefix: Prefix for auto-generated descriptions
        **common_metadata: Metadata applied to all tools
        
    Returns:
        Dict mapping tool names to success status
    """
    results = {}
    
    for tool_name, func in functions.items():
        try:
            description = f"{description_prefix}{func.__name__}" if description_prefix else None
            await register_fn_tool(
                func,
                name=tool_name,
                description=description,
                namespace=namespace,
                **common_metadata
            )
            results[tool_name] = True
        except Exception:
            results[tool_name] = False
    
    return results


async def register_module_functions(
    module: types.ModuleType,
    *,
    namespace: Optional[str] = None,
    include_private: bool = False,
    function_filter: Optional[Callable[[Callable], bool]] = None,
    **common_metadata: Any
) -> Dict[str, bool]:
    """
    Register all functions from a module as tools.
    
    Args:
        module: Module to scan for functions
        namespace: Namespace to use (defaults to module name)
        include_private: Whether to include functions starting with _
        function_filter: Optional filter function for selecting functions
        **common_metadata: Metadata applied to all tools
        
    Returns:
        Dict mapping function names to registration success status
    """
    if namespace is None:
        namespace = module.__name__
    
    functions = {}
    
    for name in dir(module):
        if not include_private and name.startswith('_'):
            continue
            
        obj = getattr(module, name)
        if not callable(obj) or not inspect.isfunction(obj):
            continue
            
        if function_filter and not function_filter(obj):
            continue
            
        functions[name] = obj
    
    return await register_function_batch(
        functions,
        namespace=namespace,
        **common_metadata
    )


# ────────────────────────────────────────────────────────────────────────────
# Utility functions
# ────────────────────────────────────────────────────────────────────────────

def validate_tool_function(func: Callable) -> bool:
    """
    Validate that a function can be used as a tool.
    
    Args:
        func: Function to validate
        
    Returns:
        True if the function is valid for tool registration
    """
    if not callable(func):
        return False
    
    try:
        # Check if we can introspect the signature
        inspect.signature(func)
        return True
    except (ValueError, TypeError):
        return False


async def get_registered_function_tools(namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all registered function tools.
    
    Args:
        namespace: Optional namespace filter
        
    Returns:
        Dict mapping tool names to tool metadata
    """
    registry = await ToolRegistryProvider.get_registry()
    all_metadata = await registry.list_metadata(namespace)
    
    function_tools = {}
    for metadata in all_metadata:
        if getattr(metadata, 'source', None) == "function":
            key = f"{metadata.namespace}.{metadata.name}" if namespace is None else metadata.name
            function_tools[key] = metadata
    
    return function_tools