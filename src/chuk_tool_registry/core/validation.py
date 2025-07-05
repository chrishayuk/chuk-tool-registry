# chuk_tool_registry/core/validation.py
"""
Async runtime validation for tool inputs/outputs with Pydantic.

This module provides validation capabilities for the chuk_tool_registry,
ensuring type safety and data integrity for tool arguments and results.

Pydantic and anyio are required dependencies.
"""
from __future__ import annotations

import inspect
import asyncio
from functools import wraps
from typing import Any, Callable, Dict, get_type_hints, Optional
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, create_model, ConfigDict
from pydantic import __version__ as pydantic_version

# Import exceptions from our core
from chuk_tool_registry.core.exceptions import ToolValidationError

# Check Pydantic version
PYDANTIC_V2 = pydantic_version.startswith('2.')


__all__ = [
    "validate_arguments",
    "validate_result", 
    "with_validation",
    "ValidationConfig",
    "create_validation_wrapper",
    "validate_tool_execution"
]


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for tool validation behavior."""
    validate_arguments: bool = True
    validate_results: bool = True
    strict_mode: bool = False
    allow_extra_args: bool = True
    coerce_types: bool = True

    def __hash__(self):
        """Make ValidationConfig hashable."""
        return hash((
            self.validate_arguments,
            self.validate_results,
            self.strict_mode,
            self.allow_extra_args,
            self.coerce_types
        ))


# --------------------------------------------------------------------------- #
# Schema Creation
# --------------------------------------------------------------------------- #

def _create_arg_model(tool_name: str, fn: Callable, config: ValidationConfig) -> type[BaseModel]:
    """Create argument validation model from function signature."""
    try:
        hints = get_type_hints(fn)
    except (NameError, AttributeError, TypeError):
        hints = getattr(fn, '__annotations__', {})
    
    # Remove return annotation
    hints.pop("return", None)

    sig = inspect.signature(fn)
    fields: Dict[str, tuple[Any, Any]] = {}
    
    # Build fields from function signature parameters
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter for method calls
        if param_name == 'self':
            continue
            
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        
        # Get type hint for this parameter
        hint = hints.get(param_name, param.annotation)
        if hint == inspect.Parameter.empty:
            hint = Any
            
        # Get default value - if no default, field is required
        if param.default is inspect.Parameter.empty:
            # Required field
            fields[param_name] = (hint, ...)
        else:
            # Optional field with default
            fields[param_name] = (hint, param.default)

    # Create model with proper configuration
    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra='allow' if config.allow_extra_args else 'forbid',
            strict=config.strict_mode,
        )
        
        return create_model(
            f"{tool_name}Args",
            __config__=model_config,
            **fields,
        )
    else:
        # Pydantic v1
        from pydantic import Extra
        
        config_class = type(
            "Config",
            (),
            {
                "extra": Extra.allow if config.allow_extra_args else Extra.forbid,
                "validate_assignment": config.strict_mode,
            },
        )
        
        return create_model(
            f"{tool_name}Args",
            __config__=config_class,
            **fields,
        )


def _create_result_model(tool_name: str, fn: Callable, config: ValidationConfig) -> type[BaseModel] | None:
    """Create result validation model from function return type hint."""
    try:
        hints = get_type_hints(fn)
    except (NameError, AttributeError, TypeError):
        hints = getattr(fn, '__annotations__', {})
    
    return_hint = hints.get("return")
    if return_hint is None or return_hint is type(None):  # noqa: E721
        return None

    # Create model for result validation
    if PYDANTIC_V2:
        model_config = ConfigDict(
            strict=config.strict_mode,
        )
        
        return create_model(
            f"{tool_name}Result",
            __config__=model_config,
            result=(return_hint, ...),
        )
    else:
        # Pydantic v1
        config_class = type(
            "Config",
            (),
            {
                "validate_assignment": config.strict_mode,
            },
        )
        
        return create_model(
            f"{tool_name}Result",
            __config__=config_class,
            result=(return_hint, ...),
        )


# --------------------------------------------------------------------------- #
# Public Validation Functions
# --------------------------------------------------------------------------- #

def validate_arguments(
    tool_name: str, 
    fn: Callable, 
    args: Dict[str, Any],
    config: Optional[ValidationConfig] = None
) -> Dict[str, Any]:
    """Validate function arguments against type hints."""
    config = config or ValidationConfig()
    
    try:
        model = _create_arg_model(tool_name, fn, config)
        validated = model(**args)
        
        # Return validated data
        if PYDANTIC_V2:
            validated_data = validated.model_dump()
        else:
            validated_data = validated.dict()
        
        # CRITICAL: Only return arguments that were explicitly provided by the user
        # Do NOT automatically add defaults - that's the function's responsibility
        result = {}
        for param_name in args.keys():
            if param_name in validated_data:
                result[param_name] = validated_data[param_name]
        
        return result
            
    except ValidationError as exc:
        raise ToolValidationError(tool_name, exc.errors()) from exc
    except Exception as e:
        # Log unexpected errors but don't fail validation
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Validation setup failed for {tool_name}: {e}, returning original args")
        return args


def validate_result(
    tool_name: str, 
    fn: Callable, 
    result: Any,
    config: Optional[ValidationConfig] = None
) -> Any:
    """Validate function return value against return type hint."""
    config = config or ValidationConfig()
    
    try:
        model = _create_result_model(tool_name, fn, config)
        
        if model is None:  # No return type annotation
            return result
            
        validated = model(result=result)
        return validated.result
        
    except ValidationError as exc:
        raise ToolValidationError(tool_name, exc.errors()) from exc
    except Exception as e:
        # Log unexpected errors but don't fail validation
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Result validation setup failed for {tool_name}: {e}, returning original result")
        return result


# --------------------------------------------------------------------------- #
# Decorators and Wrappers
# --------------------------------------------------------------------------- #

def with_validation(cls=None, *, config: Optional[ValidationConfig] = None):
    """
    Wrap an async execute method with argument & result validation.

    Examples:
        @with_validation
        class MyTool:
            async def execute(self, x: int, y: int) -> int:
                return x + y

        @with_validation(config=ValidationConfig(strict_mode=True))
        class StrictTool:
            async def execute(self, data: str) -> str:
                return data.upper()
    """
    if cls is None:
        # Called with arguments: @with_validation(config=...)
        return lambda cls: with_validation(cls, config=config)
    
    # Called without arguments: @with_validation
    config = config or ValidationConfig()
    
    # Find the execute method
    fn_name = "_execute" if hasattr(cls, "_execute") else "execute"
    
    if not hasattr(cls, fn_name):
        raise AttributeError(f"Tool {cls.__name__} must have an {fn_name} method")
        
    original = getattr(cls, fn_name)
    
    # Ensure the method is async
    if not inspect.iscoroutinefunction(original):
        raise TypeError(f"Tool {cls.__name__} must have an async {fn_name} method")

    @wraps(original)
    async def _validated(self, **kwargs):
        name = getattr(self, '__class__').__name__
        
        # Validate arguments if enabled
        if config.validate_arguments:
            kwargs = validate_arguments(name, original, kwargs, config)
        
        # Execute the original method
        result = await original(self, **kwargs)
        
        # Validate result if enabled
        if config.validate_results:
            result = validate_result(name, original, result, config)
            
        return result

    setattr(cls, fn_name, _validated)
    return cls


def create_validation_wrapper(
    tool_name: str,
    original_execute: Callable,
    config: Optional[ValidationConfig] = None
) -> Callable:
    """
    Create a validation wrapper for an execute function.
    
    Args:
        tool_name: Name of the tool for error reporting
        original_execute: The original execute function to wrap
        config: Validation configuration
        
    Returns:
        Wrapped async function with validation
    """
    config = config or ValidationConfig()
    
    if not inspect.iscoroutinefunction(original_execute):
        raise TypeError(f"Tool {tool_name} execute method must be async")

    @wraps(original_execute)
    async def _validated_wrapper(**kwargs):
        # Validate arguments if enabled
        if config.validate_arguments:
            kwargs = validate_arguments(tool_name, original_execute, kwargs, config)
        
        # Execute the original function
        result = await original_execute(**kwargs)
        
        # Validate result if enabled
        if config.validate_results:
            result = validate_result(tool_name, original_execute, result, config)
            
        return result

    return _validated_wrapper


# --------------------------------------------------------------------------- #
# Registry Integration
# --------------------------------------------------------------------------- #

async def validate_tool_execution(
    tool_name: str,
    tool_instance: Any,
    execute_method: Callable,
    kwargs: Dict[str, Any],
    config: Optional[ValidationConfig] = None
) -> Any:
    """
    Execute a tool with validation.
    
    This is the main integration point for the registry to add validation
    to any tool execution.
    
    Args:
        tool_name: Name of the tool
        tool_instance: The tool instance
        execute_method: The execute method to call
        kwargs: Arguments to pass to execute
        config: Validation configuration
        
    Returns:
        The validated result
    """
    config = config or ValidationConfig()
    
    # Validate arguments if enabled
    if config.validate_arguments:
        kwargs = validate_arguments(tool_name, execute_method, kwargs, config)
    
    # Execute the method
    if tool_instance is not None:
        # For bound methods, don't pass the instance again
        if hasattr(execute_method, '__self__'):
            result = await execute_method(**kwargs)
        else:
            # For unbound methods, pass the instance
            result = await execute_method(tool_instance, **kwargs)
    else:
        result = await execute_method(**kwargs)
    
    # Validate result if enabled
    if config.validate_results:
        result = validate_result(tool_name, execute_method, result, config)
    
    return result