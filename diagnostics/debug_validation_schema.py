#!/usr/bin/env python3
"""
Debug script to understand and fix the validation schema generation issue.
"""

import asyncio
import inspect
from typing import List, Dict, Any, get_type_hints

# Test the current schema generation
def test_schema_generation():
    """Test what's happening with schema generation."""
    
    # Test function similar to the failing ones
    async def test_function(a: int, b: int) -> int:
        """Test function for debugging."""
        return a + b
    
    async def test_function_complex(numbers: List[float], include_std: bool = True) -> Dict[str, float]:
        """Test function with complex types."""
        if not numbers:
            return {"count": 0, "mean": 0, "std": 0}
        
        mean = sum(numbers) / len(numbers)
        result = {
            "count": len(numbers),
            "mean": mean,
            "min": min(numbers),
            "max": max(numbers)
        }
        
        if include_std:
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            result["std"] = variance ** 0.5
        
        return result
    
    print("🔍 Debug: Schema Generation Analysis")
    print("=" * 50)
    
    for func in [test_function, test_function_complex]:
        print(f"\n📋 Function: {func.__name__}")
        
        # Check signature
        sig = inspect.signature(func)
        print(f"   Signature: {sig}")
        
        # Check type hints
        try:
            hints = get_type_hints(func)
            print(f"   Type hints: {hints}")
        except Exception as e:
            print(f"   Type hints error: {e}")
            hints = getattr(func, '__annotations__', {})
            print(f"   Raw annotations: {hints}")
        
        # Check parameters
        print(f"   Parameters:")
        for param_name, param in sig.parameters.items():
            print(f"     {param_name}: {param}")
            print(f"       Kind: {param.kind}")
            print(f"       Default: {param.default}")
            print(f"       Annotation: {param.annotation}")
        
        # Test creating a simple Pydantic model
        try:
            from pydantic import create_model, Field
            from typing import Any
            
            fields = {}
            hints_no_return = {k: v for k, v in hints.items() if k != 'return'}
            
            for param_name, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                
                # Get type hint
                hint = hints_no_return.get(param_name, param.annotation)
                if hint == inspect.Parameter.empty:
                    hint = Any
                
                # Get default
                default = param.default if param.default != inspect.Parameter.empty else ...
                
                fields[param_name] = (hint, default)
            
            print(f"   Fields for Pydantic: {fields}")
            
            # Create model
            model = create_model(f"{func.__name__}Schema", **fields)
            print(f"   Created model: {model}")
            
            # Test model fields
            if hasattr(model, 'model_fields'):
                print(f"   Model fields: {list(model.model_fields.keys())}")
            elif hasattr(model, '__fields__'):
                print(f"   Model fields (v1): {list(model.__fields__.keys())}")
            
        except Exception as e:
            print(f"   Schema creation error: {e}")
            import traceback
            traceback.print_exc()


def test_validation_config():
    """Test different ValidationConfig settings."""
    print("\n🔧 Debug: ValidationConfig Testing")
    print("=" * 50)
    
    from chuk_tool_registry.core.validation import ValidationConfig
    
    configs = {
        "default": ValidationConfig(),
        "strict": ValidationConfig(strict_mode=True, allow_extra_args=False),
        "lenient": ValidationConfig(strict_mode=False, allow_extra_args=True),
    }
    
    for name, config in configs.items():
        print(f"\n{name} config:")
        print(f"   allow_extra_args: {config.allow_extra_args}")
        print(f"   strict_mode: {config.strict_mode}")
        print(f"   validate_arguments: {config.validate_arguments}")


async def test_registry_validation():
    """Test the registry validation in isolation."""
    print("\n🧪 Debug: Registry Validation Testing")
    print("=" * 50)
    
    from chuk_tool_registry.providers.memory import InMemoryToolRegistry
    from chuk_tool_registry.core.validation import ValidationConfig
    
    # Create registry with validation
    registry = InMemoryToolRegistry(
        enable_validation_by_default=False,
        default_validation_config=ValidationConfig(allow_extra_args=True)
    )
    
    # Simple test function
    async def simple_add(a: int, b: int) -> int:
        return a + b
    
    # Register with validation
    await registry.register_tool(
        simple_add,
        name="simple_add",
        namespace="test",
        enable_validation=True,
        metadata={"description": "Simple addition"}
    )
    
    print("✅ Registered simple_add function")
    
    # Try to get and execute
    try:
        tool = await registry.get_tool("simple_add", "test")
        print(f"✅ Retrieved tool: {tool}")
        
        result = await tool.execute(a=5, b=3)
        print(f"✅ Execution result: {result}")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all debug tests."""
    test_schema_generation()
    test_validation_config()
    await test_registry_validation()


if __name__ == "__main__":
    asyncio.run(main())