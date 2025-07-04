#!/usr/bin/env python3
"""
Test script to verify the @register_tool decorator works with functions.
"""

import asyncio
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chuk_tool_registry import register_tool, get_registry, ensure_registrations


@register_tool("test_add", namespace="test_math")
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@register_tool("test_greet", namespace="test_utils")
async def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


async def test_function_decorators():
    """Test that function decorators work properly."""
    print("🧪 Testing @register_tool decorator with functions...")
    
    # Small delay to allow async registration to complete
    await asyncio.sleep(0.1)
    
    # Process any remaining registrations
    result = await ensure_registrations()
    print(f"📊 Processed {result['processed']} registrations")
    
    # Get registry and test tools
    registry = await get_registry()
    
    # Test add function
    add_tool = await registry.get_tool("test_add", "test_math")
    if add_tool:
        result = await add_tool.execute(a=5, b=3)
        print(f"✅ test_add(5, 3) = {result}")
    else:
        print("❌ test_add tool not found")
    
    # Test greet function
    greet_tool = await registry.get_tool("test_greet", "test_utils")
    if greet_tool:
        result = await greet_tool.execute(name="World")
        print(f"✅ test_greet('World') = '{result}'")
    else:
        print("❌ test_greet tool not found")
    
    # List all tools to debug
    all_tools = await registry.list_tools()
    test_tools = [(ns, name) for ns, name in all_tools if ns.startswith("test_")]
    print(f"🔍 Test tools found: {test_tools}")
    
    return len(test_tools) > 0


if __name__ == "__main__":
    success = asyncio.run(test_function_decorators())
    if success:
        print("🎉 Function decorator test passed!")
        sys.exit(0)
    else:
        print("❌ Function decorator test failed!")
        sys.exit(1)