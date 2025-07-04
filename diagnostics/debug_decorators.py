#!/usr/bin/env python3
"""
Debug script to understand what's happening with function decorators.
"""

import asyncio
import inspect
import gc
from typing import Callable

async def debug_decorator_issue():
    """Debug the decorator registration issue."""
    print("🔍 Debugging decorator registration...")
    
    # Import and test the decorator
    from chuk_tool_registry import register_tool, get_registry, ensure_registrations
    
    print("📝 Defining functions with decorators...")
    
    # Define test functions
    @register_tool("debug_add", namespace="debug_math")
    async def debug_add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    @register_tool("debug_greet", namespace="debug_utils")  
    async def debug_greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"
    
    print("🔍 Checking function attributes after decoration...")
    
    # Check what attributes were set on the functions
    for func_name, func in [("debug_add", debug_add), ("debug_greet", debug_greet)]:
        print(f"\n📋 Function: {func_name}")
        attrs = [attr for attr in dir(func) if attr.startswith('_tool') or attr.startswith('_registration') or attr.startswith('_is_function')]
        for attr in attrs:
            value = getattr(func, attr, None)
            print(f"  {attr}: {type(value).__name__} = {value}")
    
    # Look for functions with registration markers using garbage collection
    print("\n🔍 Scanning for decorated functions in memory...")
    decorated_functions = []
    
    for obj in gc.get_objects():
        try:
            if (inspect.isfunction(obj) and 
                hasattr(obj, '_is_function_tool')):
                decorated_functions.append(obj)
                print(f"  Found: {obj.__name__} (namespace: {getattr(obj, '_tool_namespace', 'unknown')})")
        except Exception as e:
            continue
    
    print(f"\n📊 Found {len(decorated_functions)} decorated functions")
    
    # Try to process any deferred registrations
    print("\n⚙️  Processing registrations...")
    result = await ensure_registrations()
    print(f"📊 Result: {result}")
    
    # Check the registry
    print("\n🔍 Checking registry contents...")
    registry = await get_registry()
    
    # Try to find our tools
    for tool_name, namespace in [("debug_add", "debug_math"), ("debug_greet", "debug_utils")]:
        tool = await registry.get_tool(tool_name, namespace)
        print(f"  {namespace}.{tool_name}: {'✅ Found' if tool else '❌ Not found'}")
    
    # List all tools to see what's actually registered
    all_tools = await registry.list_tools()
    debug_tools = [(ns, name) for ns, name in all_tools if 'debug' in ns]
    print(f"\n📁 Debug tools in registry: {debug_tools}")
    
    # Try manual registration of one function to compare
    print("\n🔧 Trying manual registration for comparison...")
    
    from chuk_tool_registry.discovery.auto_register import FunctionToolWrapper
    
    async def manual_add(a: int, b: int) -> int:
        return a + b
    
    # Create wrapper and register manually
    wrapper = FunctionToolWrapper(manual_add, "manual_add", "Manual add function")
    await registry.register_tool(
        wrapper,
        name="manual_add", 
        namespace="manual_test",
        metadata={"source": "manual", "description": "Manual add function"}
    )
    
    manual_tool = await registry.get_tool("manual_add", "manual_test")
    print(f"  Manual registration: {'✅ Success' if manual_tool else '❌ Failed'}")
    
    if manual_tool:
        result = await manual_tool.execute(a=5, b=3)
        print(f"  Manual tool result: 5 + 3 = {result}")
    
    return len(decorated_functions) > 0

if __name__ == "__main__":
    success = asyncio.run(debug_decorator_issue())