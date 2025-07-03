#!/usr/bin/env python3
"""
Chuk Tool Registry - Comprehensive Usage Examples

This script demonstrates all the key features and patterns for using the
chuk_tool_registry package, including:

- Tool registration (classes, functions, batch operations)
- Registry operations (retrieval, listing, metadata)
- Advanced features (namespaces, isolated contexts, statistics)
- Performance optimization and best practices

Run with: python usage_examples.py [--example EXAMPLE_NAME]
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


async def example_basic_usage():
    """Example 1: Basic tool registration and usage."""
    print("🔧 Example 1: Basic Tool Registration and Usage")
    print("=" * 50)
    
    from chuk_tool_registry import (
        get_registry, register_tool, register_fn_tool, ensure_registrations
    )
    
    # Get registry early for debugging
    registry = await get_registry()
    print(f"🔍 Registry type: {type(registry).__name__}")
    
    # 1. Register a simple function as a tool
    async def calculator_add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    print("📝 Registering function tool...")
    await register_fn_tool(
        calculator_add, 
        name="add", 
        namespace="calculator",
        description="Adds two integers"
    )
    
    # Verify function tool was registered
    add_tool_check = await registry.get_tool("add", "calculator")
    print(f"✅ Function tool registered: {add_tool_check is not None}")
    
    # 2. Register a tool class - create instance and register it directly
    class MultiplyTool:
        """Tool for multiplying two numbers."""
        
        async def execute(self, a: int, b: int) -> int:
            return a * b
    
    print("📝 Registering class tool...")
    # Register the class instance directly
    multiply_instance = MultiplyTool()
    await registry.register_tool(
        multiply_instance,
        name="multiply",
        namespace="calculator",
        metadata={"description": "Multiplies two integers"}
    )
    
    # Verify class tool was registered
    multiply_tool_check = await registry.get_tool("multiply", "calculator")
    print(f"✅ Class tool registered: {multiply_tool_check is not None}")
    
    # 3. Process any pending registrations (for decorator-based tools)
    result = await ensure_registrations()
    print(f"📊 Processed {result['processed']} pending registrations")
    
    # 4. List all tools to debug
    all_tools = await registry.list_tools()
    calc_tools = [(ns, name) for ns, name in all_tools if ns == "calculator"]
    print(f"🔍 Calculator tools found: {calc_tools}")
    
    # 5. Use tools
    # Use the function tool
    add_tool = await registry.get_tool("add", "calculator")
    if add_tool is None:
        print("❌ Add tool not found!")
        return
        
    add_result = await add_tool.execute(a=5, b=3)
    print(f"🧮 5 + 3 = {add_result}")
    
    # Use the class tool
    multiply_tool = await registry.get_tool("multiply", "calculator")
    if multiply_tool is None:
        print("❌ Multiply tool not found!")
        return
        
    multiply_result = await multiply_tool.execute(a=4, b=7)
    print(f"🧮 4 × 7 = {multiply_result}")
    
    # 6. Check metadata
    add_metadata = await registry.get_metadata("add", "calculator")
    if add_metadata:
        print(f"📋 Add tool metadata: {add_metadata.description}")
    
    print("✅ Basic usage complete!\n")


async def example_advanced_tool_classes():
    """Example 2: Advanced tool classes with rich metadata."""
    print("🏗️ Example 2: Advanced Tool Classes")
    print("=" * 50)
    
    from chuk_tool_registry import get_registry
    
    class TextProcessor:
        """Advanced text processing with multiple operations."""
        
        def __init__(self):
            self.operation_count = 0
        
        async def execute(self, text: str, operation: str = "upper") -> Dict[str, Any]:
            """Process text with various operations."""
            self.operation_count += 1
            
            operations = {
                "upper": text.upper(),
                "lower": text.lower(),
                "reverse": text[::-1],
                "word_count": len(text.split()),
                "char_count": len(text)
            }
            
            if operation not in operations:
                raise ValueError(f"Unsupported operation: {operation}")
            
            result = operations[operation]
            
            return {
                "original": text,
                "operation": operation,
                "result": result,
                "execution_count": self.operation_count,
                "timestamp": time.time()
            }
    
    # Register the tool with rich metadata - put custom fields in execution_options
    registry = await get_registry()
    processor_instance = TextProcessor()
    
    await registry.register_tool(
        processor_instance,
        name="text_processor",
        namespace="utilities",
        metadata={
            "description": "Advanced text processing tool",
            "version": "2.1.0",
            "tags": ["text", "processing", "utility"],
            "execution_options": {
                "author": "Demo Team",
                "supports_streaming": False,
                "rate_limit": {"requests": 100, "period": 60}
            }
        }
    )
    
    # Use the advanced tool
    processor = await registry.get_tool("text_processor", "utilities")
    if processor is None:
        print("❌ Text processor not found!")
        return
    
    # Test different operations
    operations = ["upper", "reverse", "word_count"]
    test_text = "Hello World from Chuk Tool Registry!"
    
    for op in operations:
        result = await processor.execute(text=test_text, operation=op)
        print(f"🔤 {op}: '{result['original']}' → {result['result']}")
    
    # Check rich metadata - accessing the correct attributes from ToolMetadata
    metadata = await registry.get_metadata("text_processor", "utilities")
    if metadata:
        print(f"📊 Tool version: {metadata.version}")
        print(f"🏷️  Tags: {metadata.tags}")
        # ToolMetadata stores custom fields in execution_options dict
        print(f"👤 Author: {metadata.execution_options.get('author', 'Unknown')}")
        print(f"🔧 Rate limit: {metadata.execution_options.get('rate_limit', 'None')}")
        print(f"📝 Description: {metadata.description}")
    
    print("✅ Advanced tool classes complete!\n")


async def example_decorator_registration():
    """Example 2.5: Working decorator registration patterns."""
    print("🎨 Example 2.5: Decorator Registration Patterns")
    print("=" * 50)
    
    from chuk_tool_registry import (
        create_registration_manager, 
        register_tool,
        get_registry
    )
    
    # Create a dedicated manager for decorator examples
    decorator_manager = create_registration_manager("decorator_demo")
    
    print("📝 Registering tools with decorators...")
    
    # For decorator registration to work properly with classes, we need to register instances
    # Let's demonstrate both approaches:
    
    # Approach 1: Register class and then instantiate manually
    @register_tool("string_reverser", namespace="decorators", manager=decorator_manager)
    class StringReverser:
        """Tool that reverses strings."""
        
        async def execute(self, text: str) -> str:
            return text[::-1]
    
    @register_tool(
        name="word_counter", 
        namespace="decorators", 
        manager=decorator_manager,
        description="Count words and characters in text",
        version="1.2.0",
        tags={"text", "analysis"},
        author="Decorator Team"
    )
    class WordCounter:
        """Tool for analyzing text statistics."""
        
        async def execute(self, text: str, include_chars: bool = True) -> Dict[str, Any]:
            words = text.split()
            result = {
                "text": text,
                "word_count": len(words),
                "words": words
            }
            
            if include_chars:
                result["char_count"] = len(text)
                result["char_count_no_spaces"] = len(text.replace(" ", ""))
            
            return result
    
    @register_tool(
        name="factorial_calculator",
        namespace="decorators", 
        manager=decorator_manager,
        description="Calculate factorial of a number",
        category="mathematics"
    )
    class FactorialCalculator:
        """Calculate factorial with optional caching."""
        
        def __init__(self):
            self._cache = {0: 1, 1: 1}
        
        async def execute(self, n: int, use_cache: bool = True) -> Dict[str, Any]:
            if n < 0:
                raise ValueError("Factorial not defined for negative numbers")
            
            if use_cache and n in self._cache:
                return {
                    "input": n,
                    "result": self._cache[n],
                    "from_cache": True
                }
            
            # Calculate factorial
            result = 1
            for i in range(2, n + 1):
                result *= i
            
            if use_cache:
                self._cache[n] = result
            
            return {
                "input": n,
                "result": result,
                "from_cache": False,
                "cache_size": len(self._cache) if use_cache else 0
            }
    
    # Process the decorator registrations
    print("⚙️  Processing decorator registrations...")
    registration_result = await decorator_manager.process_registrations()
    print(f"✅ Processed {registration_result['processed']} decorator registrations")
    
    if registration_result['errors']:
        print(f"❌ Registration errors: {registration_result['errors']}")
        return
    
    # Get the main registry and manually register instances of the decorated classes
    # This is necessary because the decorator registers the class, but we need instances
    registry = await get_registry()
    
    # Register instances of the decorated classes
    print("🔧 Registering class instances...")
    
    string_reverser_instance = StringReverser()
    await registry.register_tool(
        string_reverser_instance,
        name="string_reverser_instance",
        namespace="decorators",
        metadata={"description": "String reverser instance", "source": "decorator_instance"}
    )
    
    word_counter_instance = WordCounter()
    await registry.register_tool(
        word_counter_instance,
        name="word_counter_instance", 
        namespace="decorators",
        metadata={"description": "Word counter instance", "source": "decorator_instance"}
    )
    
    factorial_instance = FactorialCalculator()
    await registry.register_tool(
        factorial_instance,
        name="factorial_calculator_instance",
        namespace="decorators", 
        metadata={"description": "Factorial calculator instance", "source": "decorator_instance"}
    )
    
    # Test the instance-registered tools
    print("\n🧪 Testing decorator pattern tools:")
    
    # Test string reverser
    reverser = await registry.get_tool("string_reverser_instance", "decorators")
    if reverser:
        reverse_result = await reverser.execute(text="Hello Decorators!")
        print(f"🔄 String reverse: 'Hello Decorators!' → '{reverse_result}'")
    else:
        print("❌ String reverser not found")
    
    # Test word counter
    counter = await registry.get_tool("word_counter_instance", "decorators")
    if counter:
        count_result = await counter.execute(text="The quick brown fox jumps", include_chars=True)
        print(f"📊 Word analysis: {count_result['word_count']} words, {count_result['char_count']} chars")
    else:
        print("❌ Word counter not found")
    
    # Test factorial calculator
    factorial_calc = await registry.get_tool("factorial_calculator_instance", "decorators")
    if factorial_calc:
        factorial_result = await factorial_calc.execute(n=5, use_cache=True)
        print(f"❗ Factorial(5): {factorial_result['result']} (cached: {factorial_result['from_cache']})")
        
        # Test cache hit
        factorial_result2 = await factorial_calc.execute(n=5, use_cache=True)
        print(f"❗ Factorial(5) again: {factorial_result2['result']} (cached: {factorial_result2['from_cache']})")
    else:
        print("❌ Factorial calculator not found")
    
    # Show registered decorator tools
    decorator_tools = await registry.list_tools("decorators")
    decorator_tool_names = [name for ns, name in decorator_tools if ns == "decorators"]
    print(f"\n🎨 Decorator namespace tools: {', '.join(decorator_tool_names)}")
    
    # Show the difference between decorator registration and instance registration
    print(f"\n💡 Note: Decorators register classes, but tools need instances.")
    print(f"   This example shows how to bridge that gap by registering instances separately.")
    
async def example_function_decorators():
    """Example 2.75: Simple function decorators (easier than class decorators)."""
    print("🎯 Example 2.75: Function Decorators (Recommended Pattern)")
    print("=" * 50)
    
    from chuk_tool_registry import register_fn_tool, get_registry
    
    print("📝 Registering functions with decorator-style patterns...")
    
    # While there isn't a direct function decorator, we can create a pattern that feels like one
    # by using a helper function that mimics decorator syntax
    
    def tool_decorator(name: str, namespace: str = "default", **metadata):
        """Decorator-style helper for registering functions as tools."""
        def decorator(func):
            # Store registration info on the function for later processing
            func._tool_name = name
            func._tool_namespace = namespace
            func._tool_metadata = metadata
            func._needs_registration = True
            return func
        return decorator
    
    # Example 1: Simple mathematical functions with decorator pattern
    @tool_decorator("add_numbers", namespace="simple_math", description="Add two numbers")
    async def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    @tool_decorator("multiply_numbers", namespace="simple_math", 
                   description="Multiply two numbers", version="1.1.0")
    async def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    @tool_decorator("power", namespace="simple_math", 
                   description="Raise a number to a power", tags=["math", "power"])
    async def power_function(base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent
    
    # Example 2: String manipulation functions
    @tool_decorator("clean_text", namespace="text_utils", 
                   description="Clean and normalize text")
    async def clean_text(text: str, remove_extra_spaces: bool = True) -> str:
        """Clean and normalize text input."""
        cleaned = text.strip()
        if remove_extra_spaces:
            cleaned = ' '.join(cleaned.split())
        return cleaned
    
    @tool_decorator("count_words", namespace="text_utils",
                   description="Count words in text", version="1.0.1")
    async def count_words(text: str, min_length: int = 1) -> dict:
        """Count words in text with optional minimum length filter."""
        words = text.split()
        filtered_words = [word for word in words if len(word) >= min_length]
        return {
            "total_words": len(words),
            "filtered_words": len(filtered_words),
            "average_length": sum(len(word) for word in filtered_words) / len(filtered_words) if filtered_words else 0
        }
    
    # Example 3: Data validation functions
    @tool_decorator("validate_email_simple", namespace="validators",
                   description="Simple email validation", category="validation")
    async def validate_email_simple(email: str) -> dict:
        """Validate email format with detailed response."""
        has_at = "@" in email
        has_dot = "." in email
        has_valid_length = 5 <= len(email) <= 100
        
        is_valid = has_at and has_dot and has_valid_length
        
        return {
            "email": email,
            "is_valid": is_valid,
            "checks": {
                "has_at_symbol": has_at,
                "has_dot": has_dot,
                "valid_length": has_valid_length
            }
        }
    
    # Now register all the decorated functions
    decorated_functions = [
        add_numbers, multiply_numbers, power_function,
        clean_text, count_words, validate_email_simple
    ]
    
    registry = await get_registry()
    registered_count = 0
    
    print("⚙️  Processing function registrations...")
    for func in decorated_functions:
        if hasattr(func, '_needs_registration'):
            await register_fn_tool(
                func,
                name=func._tool_name,
                namespace=func._tool_namespace,
                **func._tool_metadata
            )
            registered_count += 1
            # Clean up the registration markers
            delattr(func, '_needs_registration')
    
    print(f"✅ Registered {registered_count} functions using decorator pattern")
    
    # Test the registered functions
    print("\n🧪 Testing decorator-registered functions:")
    
    # Test math functions
    add_tool = await registry.get_tool("add_numbers", "simple_math")
    if add_tool:
        result = await add_tool.execute(a=15.5, b=23.7)
        print(f"➕ Add: 15.5 + 23.7 = {result}")
    
    multiply_tool = await registry.get_tool("multiply_numbers", "simple_math")
    if multiply_tool:
        result = await multiply_tool.execute(a=6, b=7)
        print(f"✖️  Multiply: 6 × 7 = {result}")
    
    power_tool = await registry.get_tool("power", "simple_math")
    if power_tool:
        result = await power_tool.execute(base=2, exponent=8)
        print(f"🔢 Power: 2^8 = {result}")
    
    # Test text functions
    clean_tool = await registry.get_tool("clean_text", "text_utils")
    if clean_tool:
        result = await clean_tool.execute(text="  Hello    World  ", remove_extra_spaces=True)
        print(f"🧹 Clean text: '  Hello    World  ' → '{result}'")
    
    count_tool = await registry.get_tool("count_words", "text_utils")
    if count_tool:
        result = await count_tool.execute(text="The quick brown fox jumps over lazy dogs", min_length=4)
        print(f"📊 Word count: {result['total_words']} total, {result['filtered_words']} with 4+ chars")
    
    # Test validation function
    validator = await registry.get_tool("validate_email_simple", "validators")
    if validator:
        result = await validator.execute(email="user@example.com")
        print(f"✉️  Email validation: {result['email']} → {'Valid' if result['is_valid'] else 'Invalid'}")
    
    # Show namespace organization
    math_tools = await registry.list_tools("simple_math")
    text_tools = await registry.list_tools("text_utils")
    validator_tools = await registry.list_tools("validators")
    
    print(f"\n📁 Function tools by namespace:")
    print(f"  🧮 simple_math: {[name for ns, name in math_tools if ns == 'simple_math']}")
    print(f"  📝 text_utils: {[name for ns, name in text_tools if ns == 'text_utils']}")
    print(f"  ✅ validators: {[name for ns, name in validator_tools if ns == 'validators']}")
    
    # Show metadata for one of the tools
    multiply_metadata = await registry.get_metadata("multiply_numbers", "simple_math")
    if multiply_metadata:
        print(f"\n📋 Sample metadata (multiply_numbers):")
        print(f"   Description: {multiply_metadata.description}")
        print(f"   Version: {multiply_metadata.version}")
        print(f"   Source: {multiply_metadata.source}")
    
    print("\n💡 Function Decorator Pattern Benefits:")
    print("   ✅ Cleaner syntax than class decorators")
    print("   ✅ Works directly with async functions")
    print("   ✅ No instance creation complexity")
    print("   ✅ Easy to understand and maintain")
    print("   ✅ Perfect for utility functions")
    
    print("✅ Function decorators complete!\n")


async def example_batch_registration():
    """Example 3: Batch registration of multiple tools."""
    print("📦 Example 3: Batch Tool Registration")
    print("=" * 50)
    
    from chuk_tool_registry import register_function_batch, register_module_functions
    import math
    
    # Define multiple utility functions
    async def fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    async def prime_check(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    async def factorial(n: int) -> int:
        """Calculate factorial of a number."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    # Batch register functions
    math_functions = {
        "fibonacci": fibonacci,
        "is_prime": prime_check,
        "factorial": factorial
    }
    
    results = await register_function_batch(
        math_functions,
        namespace="math_tools",
        description_prefix="Math utility: ",
        author="Math Team",
        category="mathematics"
    )
    
    print(f"📊 Batch registration results:")
    for tool_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {tool_name}")
    
    # Test the batch-registered tools
    from chuk_tool_registry import get_registry
    registry = await get_registry()
    
    # Test Fibonacci
    fib_tool = await registry.get_tool("fibonacci", "math_tools")
    if fib_tool:
        fib_result = await fib_tool.execute(n=10)
        print(f"🔢 Fibonacci(10) = {fib_result}")
    
    # Test prime check
    prime_tool = await registry.get_tool("is_prime", "math_tools")
    if prime_tool:
        prime_result = await prime_tool.execute(n=17)
        print(f"🔍 Is 17 prime? {prime_result}")
    
    # Test factorial
    factorial_tool = await registry.get_tool("factorial", "math_tools")
    if factorial_tool:
        factorial_result = await factorial_tool.execute(n=5)
        print(f"❗ 5! = {factorial_result}")
    
    print("✅ Batch registration complete!\n")
    """Example 3: Batch registration of multiple tools."""
    print("📦 Example 3: Batch Tool Registration")
    print("=" * 50)
    
    from chuk_tool_registry import register_function_batch, register_module_functions
    import math
    
    # Define multiple utility functions
    async def fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    async def prime_check(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    async def factorial(n: int) -> int:
        """Calculate factorial of a number."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        return 1 if n == 0 else n * await factorial(n - 1)
    
    # Batch register functions
    math_functions = {
        "fibonacci": fibonacci,
        "is_prime": prime_check,
        "factorial": factorial
    }
    
    results = await register_function_batch(
        math_functions,
        namespace="math_tools",
        description_prefix="Math utility: ",
        author="Math Team",
        category="mathematics"
    )
    
    print(f"📊 Batch registration results:")
    for tool_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {tool_name}")
    
    # Test the batch-registered tools
    registry = await get_registry()
    
    # Test Fibonacci
    fib_tool = await registry.get_tool("fibonacci", "math_tools")
    fib_result = await fib_tool.execute(n=10)
    print(f"🔢 Fibonacci(10) = {fib_result}")
    
    # Test prime check
    prime_tool = await registry.get_tool("is_prime", "math_tools")
    prime_result = await prime_tool.execute(n=17)
    print(f"🔍 Is 17 prime? {prime_result}")
    
    # Test factorial
    factorial_tool = await registry.get_tool("factorial", "math_tools")
    factorial_result = await factorial_tool.execute(n=5)
    print(f"❗ 5! = {factorial_result}")
    
    print("✅ Batch registration complete!\n")


async def example_namespace_management():
    """Example 4: Working with namespaces and organization."""
    print("🗂️ Example 4: Namespace Management")
    print("=" * 50)
    
    from chuk_tool_registry import get_registry, register_fn_tool
    
    # Register tools in different namespaces with proper async wrappers
    # Fix: Make lambda functions async to avoid anyio threading issues
    namespaces_data = {
        "data_processing": [
            (lambda data: sorted(data), "sort_data", "Sort a list of data"),
            (lambda data: list(set(data)), "deduplicate", "Remove duplicates from data"),
            (lambda data: len(data), "count_items", "Count items in data")
        ],
        "string_utils": [
            (lambda s: s.strip(), "trim", "Remove whitespace"),
            (lambda s: s.title(), "title_case", "Convert to title case"),
            (lambda s: s.replace(" ", "_"), "snake_case", "Convert to snake_case")
        ],
        "validation": [
            (lambda email: "@" in email and "." in email, "validate_email", "Basic email validation"),
            (lambda phone: phone.replace("-", "").replace(" ", "").isdigit(), "validate_phone", "Basic phone validation")
        ]
    }
    
    # Convert sync functions to async functions to avoid threading issues
    async def async_sort_data(data):
        return sorted(data)
    
    async def async_deduplicate(data):
        return list(set(data))
    
    async def async_count_items(data):
        return len(data)
    
    async def async_trim(s):
        return s.strip()
    
    async def async_title_case(s):
        return s.title()
    
    async def async_snake_case(s):
        return s.replace(" ", "_")
    
    async def async_validate_email(email):
        return "@" in email and "." in email
    
    async def async_validate_phone(phone):
        return phone.replace("-", "").replace(" ", "").isdigit()
    
    # Register async versions to avoid threading complications
    async_namespaces_data = {
        "data_processing": [
            (async_sort_data, "sort_data", "Sort a list of data"),
            (async_deduplicate, "deduplicate", "Remove duplicates from data"),
            (async_count_items, "count_items", "Count items in data")
        ],
        "string_utils": [
            (async_trim, "trim", "Remove whitespace"),
            (async_title_case, "title_case", "Convert to title case"),
            (async_snake_case, "snake_case", "Convert to snake_case")
        ],
        "validation": [
            (async_validate_email, "validate_email", "Basic email validation"),
            (async_validate_phone, "validate_phone", "Basic phone validation")
        ]
    }
    
    # Register all tools with async versions
    for namespace, tools in async_namespaces_data.items():
        for func, name, description in tools:
            await register_fn_tool(
                func, 
                name=name, 
                namespace=namespace, 
                description=description
            )
    
    registry = await get_registry()
    
    # Explore namespaces
    namespaces = await registry.list_namespaces()
    print(f"📁 Available namespaces: {', '.join(sorted(namespaces))}")
    
    # List tools in each namespace
    for ns in sorted(namespaces):
        tools = await registry.list_tools(ns)
        namespace_tools = [name for namespace, name in tools if namespace == ns]
        print(f"  📂 {ns}: {', '.join(namespace_tools)}")
    
    # Demonstrate namespace isolation
    print(f"\n🔧 Testing namespace isolation:")
    
    # Use data processing tools
    sort_tool = await registry.get_tool("sort_data", "data_processing")
    if sort_tool:
        test_data = [3, 1, 4, 1, 5, 9, 2, 6]
        sorted_data = await sort_tool.execute(data=test_data)
        print(f"  📊 Sorted {test_data} → {sorted_data}")
    
    # Use string utilities
    title_tool = await registry.get_tool("title_case", "string_utils")
    if title_tool:
        title_result = await title_tool.execute(s="hello world")
        print(f"  🔤 Title case: 'hello world' → '{title_result}'")
    
    # Use validation tools
    email_validator = await registry.get_tool("validate_email", "validation")
    if email_validator:
        email_valid = await email_validator.execute(email="test@example.com")
        print(f"  ✉️  Email validation: test@example.com → {'Valid' if email_valid else 'Invalid'}")
    
    print("✅ Namespace management complete!\n")


async def example_isolated_contexts():
    """Example 5: Isolated registration contexts for testing."""
    print("🧪 Example 5: Isolated Registration Contexts")
    print("=" * 50)
    
    from chuk_tool_registry import (
        ToolRegistryProvider,
        get_registry
    )
    from chuk_tool_registry.providers.memory import InMemoryToolRegistry
    
    # Create isolated test environment
    test_registry = InMemoryToolRegistry()
    
    # Use provider's isolated context
    async with ToolRegistryProvider.isolated_registry(test_registry):
        # Register tools directly in the isolated context
        class TestAdder:
            async def execute(self, a: int, b: int) -> int:
                return a + b
        
        class TestMultiplier:
            async def execute(self, a: int, b: int) -> int:
                return a * b
        
        # Get the isolated registry and register tools
        isolated_registry = await ToolRegistryProvider.get_registry()
        
        adder_instance = TestAdder()
        multiplier_instance = TestMultiplier()
        
        await isolated_registry.register_tool(adder_instance, name="test_adder", namespace="test")
        await isolated_registry.register_tool(multiplier_instance, name="test_multiplier", namespace="test")
        
        print(f"🧪 Isolated context: registered 2 tools")
        
        # Use tools in isolated context
        adder = await isolated_registry.get_tool("test_adder", "test")
        if adder:
            add_result = await adder.execute(a=10, b=20)
            print(f"➕ Isolated add: 10 + 20 = {add_result}")
        
        # Show isolation by listing tools
        test_tools = await isolated_registry.list_tools("test")
        test_tool_names = [name for ns, name in test_tools if ns == "test"]
        print(f"🔒 Tools in isolated context: {test_tool_names}")
    
    # Back to main context - isolated tools should not be visible
    main_registry = await get_registry()
    main_tools = await main_registry.list_tools("test")
    main_test_tools = [name for ns, name in main_tools if ns == "test"]
    
    print(f"🌍 Tools in main context (test namespace): {main_test_tools}")
    print(f"✅ Isolation verified: {len(main_test_tools) == 0}")
    
    print("✅ Isolated contexts complete!\n")


async def example_performance_optimization():
    """Example 6: Performance monitoring and optimization."""
    print("⚡ Example 6: Performance Monitoring")
    print("=" * 50)
    
    from chuk_tool_registry import get_registry, register_fn_tool, get_discovery_stats
    
    # Create performance test tools
    async def fast_operation(data: List[int]) -> int:
        """Fast O(n) operation."""
        return sum(data)
    
    async def medium_operation(data: List[int]) -> int:
        """Medium O(n log n) operation."""
        return sum(sorted(data))
    
    async def slow_operation(data: List[int]) -> int:
        """Intentionally slow operation for testing."""
        # Simulate some processing time
        await asyncio.sleep(0.001)
        return max(data) if data else 0
    
    # Register performance test tools
    perf_tools = {
        "fast_sum": fast_operation,
        "sorted_sum": medium_operation,
        "slow_max": slow_operation
    }
    
    for name, func in perf_tools.items():
        await register_fn_tool(func, name=name, namespace="performance")
    
    registry = await get_registry()
    
    # Performance testing
    test_data = list(range(1000))
    performance_results = {}
    
    for tool_name in perf_tools.keys():
        tool = await registry.get_tool(tool_name, "performance")
        
        # Time the operation
        start_time = time.time()
        result = await tool.execute(data=test_data)
        execution_time = (time.time() - start_time) * 1000  # ms
        
        performance_results[tool_name] = {
            "result": result,
            "execution_time_ms": execution_time
        }
        
        print(f"⏱️  {tool_name}: {execution_time:.2f}ms (result: {result})")
    
    # Get registry statistics
    stats = await get_discovery_stats()
    print(f"\n📊 Registry Statistics:")
    print(f"  🔧 Total tools: {stats['total_tools']}")
    print(f"  📁 Namespaces: {stats['total_namespaces']}")
    print(f"  📈 Source breakdown: {stats['source_breakdown']}")
    
    # Performance recommendations
    fastest = min(performance_results.items(), key=lambda x: x[1]['execution_time_ms'])
    slowest = max(performance_results.items(), key=lambda x: x[1]['execution_time_ms'])
    
    print(f"\n🏆 Performance Analysis:")
    print(f"  ⚡ Fastest: {fastest[0]} ({fastest[1]['execution_time_ms']:.2f}ms)")
    print(f"  🐌 Slowest: {slowest[0]} ({slowest[1]['execution_time_ms']:.2f}ms)")
    
    if slowest[1]['execution_time_ms'] > 5:
        print(f"  💡 Optimization opportunity: {slowest[0]} could be optimized")
    
    print("✅ Performance monitoring complete!\n")


async def example_error_handling():
    """Example 7: Comprehensive error handling patterns."""
    print("🛡️ Example 7: Error Handling and Validation")
    print("=" * 50)
    
    from chuk_tool_registry import (
        get_registry, ToolNotFoundError
    )
    
    class ErrorDemoTool:
        """Tool demonstrating various error scenarios."""
        
        async def execute(self, operation: str, value: int = 10) -> Any:
            if operation == "divide_by_zero":
                return value / 0
            elif operation == "type_error":
                return value + "string"  # Type error
            elif operation == "value_error":
                if value < 0:
                    raise ValueError("Value must be non-negative")
                return value
            elif operation == "success":
                return {"status": "success", "value": value}
            else:
                raise ValueError(f"Unknown operation: {operation}")
    
    # Register the error demo tool
    registry = await get_registry()
    error_tool_instance = ErrorDemoTool()
    await registry.register_tool(error_tool_instance, name="error_demo", namespace="testing")
    
    # Test various error scenarios
    error_scenarios = [
        ("success", 42, "✅"),
        ("value_error", -5, "🛡️"),
        ("divide_by_zero", 10, "🛡️"),
        ("type_error", 5, "🛡️"),
        ("unknown_op", 1, "🛡️")
    ]
    
    print("🧪 Testing error scenarios:")
    
    for operation, value, expected in error_scenarios:
        try:
            tool = await registry.get_tool("error_demo", "testing")
            if tool is None:
                print(f"  ❌ Tool not found for {operation}")
                continue
                
            result = await tool.execute(operation=operation, value=value)
            print(f"  ✅ {operation}(value={value}): {result}")
            
        except ValueError as e:
            print(f"  ✅ {operation}(value={value}): Caught ValueError - {e}")
        except ZeroDivisionError as e:
            print(f"  ✅ {operation}(value={value}): Caught ZeroDivisionError - {e}")
        except TypeError as e:
            print(f"  ✅ {operation}(value={value}): Caught TypeError - {e}")
        except Exception as e:
            print(f"  ✅ {operation}(value={value}): Caught {type(e).__name__} - {e}")
    
    # Test tool not found error
    missing_tool = await registry.get_tool("nonexistent", "testing")
    if missing_tool is None:
        print("  ✅ Tool retrieval: nonexistent tool returned None as expected")
    
    # Test strict retrieval
    try:
        missing_tool = await registry.get_tool_strict("nonexistent", "testing")
    except ToolNotFoundError as e:
        print(f"  ✅ Strict retrieval: Successfully caught ToolNotFoundError - {e}")
    
    print("✅ Error handling complete!\n")


async def example_real_world_workflow():
    """Example 8: Real-world data processing workflow."""
    print("🌟 Example 8: Real-World Data Processing Workflow")
    print("=" * 50)
    
    from chuk_tool_registry import register_fn_tool, get_registry
    
    # Data processing pipeline tools
    async def load_data(source: str) -> List[Dict[str, Any]]:
        """Simulate loading data from various sources."""
        if source == "users":
            return [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
                {"id": 4, "name": "Diana", "email": "diana@example.com", "age": 28}
            ]
        return []
    
    async def filter_data(data: List[Dict], field: str, min_value: Any) -> List[Dict]:
        """Filter data by field value."""
        return [item for item in data if item.get(field, 0) >= min_value]
    
    async def transform_data(data: List[Dict], operation: str) -> List[Dict]:
        """Transform data with various operations."""
        if operation == "add_full_name":
            for item in data:
                item["full_name"] = f"{item.get('name', 'Unknown')}"
        elif operation == "categorize_age":
            for item in data:
                age = item.get("age", 0)
                if age < 30:
                    item["age_category"] = "young"
                elif age < 40:
                    item["age_category"] = "middle"
                else:
                    item["age_category"] = "senior"
        return data
    
    async def aggregate_data(data: List[Dict], group_by: str) -> Dict[str, Any]:
        """Aggregate data by specified field."""
        groups = {}
        for item in data:
            key = item.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        return {
            "groups": groups,
            "summary": {group: len(items) for group, items in groups.items()},
            "total_items": len(data)
        }
    
    # Register pipeline tools
    pipeline_tools = {
        "load_data": load_data,
        "filter_data": filter_data,
        "transform_data": transform_data,
        "aggregate_data": aggregate_data
    }
    
    for name, func in pipeline_tools.items():
        await register_fn_tool(func, name=name, namespace="data_pipeline")
    
    registry = await get_registry()
    
    # Execute data processing workflow
    print("🔄 Executing data processing workflow:")
    
    # Step 1: Load data
    loader = await registry.get_tool("load_data", "data_pipeline")
    raw_data = await loader.execute(source="users")
    print(f"  📥 Loaded {len(raw_data)} records")
    
    # Step 2: Filter data (users age 27 and above)
    filter_tool = await registry.get_tool("filter_data", "data_pipeline")
    filtered_data = await filter_tool.execute(data=raw_data, field="age", min_value=27)
    print(f"  🔍 Filtered to {len(filtered_data)} records (age >= 27)")
    
    # Step 3: Transform data (add age categories)
    transformer = await registry.get_tool("transform_data", "data_pipeline")
    transformed_data = await transformer.execute(data=filtered_data, operation="categorize_age")
    print(f"  🔄 Added age categories to {len(transformed_data)} records")
    
    # Step 4: Aggregate data
    aggregator = await registry.get_tool("aggregate_data", "data_pipeline")
    aggregated_result = await aggregator.execute(data=transformed_data, group_by="age_category")
    
    print(f"  📊 Aggregation results:")
    for category, count in aggregated_result["summary"].items():
        print(f"    • {category}: {count} users")
    
    # Display sample of final data
    print(f"\n📋 Sample processed record:")
    if transformed_data:
        sample = transformed_data[0]
        for key, value in sample.items():
            print(f"    {key}: {value}")
    
    print("✅ Real-world workflow complete!\n")


async def main():
    """Main function to run usage examples."""
    import argparse
    
    # Available examples
    examples = {
        "basic": example_basic_usage,
        "advanced": example_advanced_tool_classes,
        "decorators": example_decorator_registration,
        "functions": example_function_decorators,
        "batch": example_batch_registration,
        "namespaces": example_namespace_management,
        "isolation": example_isolated_contexts,
        "performance": example_performance_optimization,
        "errors": example_error_handling,
        "workflow": example_real_world_workflow
    }
    
    parser = argparse.ArgumentParser(description="Chuk Tool Registry Usage Examples")
    parser.add_argument(
        "--example", "-e",
        choices=list(examples.keys()) + ["all"],
        default="all",
        help="Specific example to run (default: all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available examples"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("📚 Available Examples:")
        for name, func in examples.items():
            doc = func.__doc__ or "No description available"
            print(f"  {name}: {doc.split('.')[0]}")
        return
    
    print("🚀 Chuk Tool Registry - Usage Examples")
    print("=" * 60)
    print()
    
    # Run specific example or all examples
    if args.example == "all":
        for name, example_func in examples.items():
            await example_func()
            await asyncio.sleep(0.1)  # Brief pause between examples
    else:
        await examples[args.example]()
    
    print("🎉 All examples completed successfully!")
    print("\n💡 Next Steps:")
    print("  • Explore the chuk_tool_registry documentation")
    print("  • Create your own custom tools")
    print("  • Integrate with your existing applications")
    print("  • Run diagnostics: python package_diagnostic.py")


if __name__ == "__main__":
    # Handle Windows event loop policy
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()