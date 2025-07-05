#!/usr/bin/env python3
"""
Comprehensive example demonstrating the integrated validation system
in the chuk_tool_registry.

This example shows:
1. Enhanced registry with validation capabilities
2. Decorator-based registration with validation
3. Function registration with validation
4. Registry-level validation execution
5. Validation statistics and monitoring
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


async def example_enhanced_registry_setup():
    """Example: Setting up registry with validation capabilities."""
    print("🔧 Example: Enhanced Registry Setup")
    print("=" * 50)
    
    from chuk_tool_registry.providers.memory import InMemoryToolRegistry
    from chuk_tool_registry.core.validation import ValidationConfig
    from chuk_tool_registry.core.provider import ToolRegistryProvider
    
    # Create registry with validation enabled by default
    validation_config = ValidationConfig(
        validate_arguments=True,
        validate_results=True,
        strict_mode=False,
        allow_extra_args=False,
        coerce_types=True
    )
    
    enhanced_registry = InMemoryToolRegistry(
        enable_statistics=True,
        validate_tools=True,
        default_validation_config=validation_config,
        enable_validation_by_default=False  # Explicit opt-in per tool
    )
    
    # Use in isolated context
    async with ToolRegistryProvider.isolated_registry(enhanced_registry):
        print(f"✅ Created enhanced registry with validation support")
        print(f"   Validation available: {enhanced_registry._default_validation_config is not None}")
        print(f"   Statistics enabled: {enhanced_registry._enable_statistics}")
        
        # Register a simple tool with validation
        class ValidatedCalculator:
            async def execute(self, a: int, b: int, operation: str = "add") -> float:
                """Perform mathematical operations with validation."""
                operations = {
                    "add": a + b,
                    "subtract": a - b,
                    "multiply": a * b,
                    "divide": a / b if b != 0 else float('inf')
                }
                
                if operation not in operations:
                    raise ValueError(f"Unknown operation: {operation}")
                
                return float(operations[operation])
        
        # Register with validation enabled
        calc_instance = ValidatedCalculator()
        await enhanced_registry.register_tool(
            calc_instance,
            name="validated_calculator",
            namespace="math",
            enable_validation=True,
            validation_config=validation_config,
            metadata={
                "description": "Calculator with input/output validation",
                "version": "1.0.0",
                "category": "mathematics"
            }
        )
        
        print(f"✅ Registered calculator with validation enabled")
        
        # Test validation execution
        result = await enhanced_registry.execute_tool_with_validation(
            "validated_calculator", 
            "math",
            a=10, 
            b=5, 
            operation="multiply"
        )
        print(f"🧮 Calculation result: 10 * 5 = {result}")
        
        # Test validation error
        try:
            await enhanced_registry.execute_tool_with_validation(
                "validated_calculator",
                "math", 
                a="not_a_number",  # Should cause validation error
                b=5,
                operation="add"
            )
        except Exception as e:
            print(f"🛡️  Validation caught error: {type(e).__name__}")
        
        # Get validation statistics
        stats = await enhanced_registry.get_statistics()
        print(f"📊 Registry stats:")
        print(f"   Total tools: {stats['total_tools']}")
        print(f"   Tools with validation: {stats['tools_with_validation']}")
        print(f"   Validation executions: {stats.get('executions_count', 0)}")
        
    print("✅ Enhanced registry setup complete!\n")


async def example_decorator_validation():
    """Example: Using decorators with validation."""
    print("🎨 Example: Decorator-Based Validation")
    print("=" * 50)
    
    from chuk_tool_registry import (
        register_tool, 
        validated_tool,
        ensure_registrations,
        get_registry,
        ValidationConfig
    )
    
    # Example 1: Standard decorator with validation enabled
    @register_tool(
        "add_numbers", 
        namespace="validation_demo",
        enable_validation=True,
        description="Add two numbers with validation"
    )
    async def add_numbers(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b
    
    # Example 2: Validated tool decorator (validation enabled by default)
    @validated_tool(
        "strict_multiply",
        namespace="validation_demo", 
        strict_mode=True,
        description="Strict multiplication with validation"
    )
    async def strict_multiply(a: float, b: float) -> float:
        """Multiply two numbers with strict validation."""
        return a * b
    
    # Example 3: Class with validation
    @register_tool(
        "data_processor",
        namespace="validation_demo",
        validation_config=ValidationConfig(
            validate_arguments=True,
            validate_results=True,
            allow_extra_args=False
        ),
        enable_validation=True
    )
    class DataProcessor:
        """Process data with comprehensive validation."""
        
        async def execute(self, data: List[int], operation: str = "sum") -> Dict[str, Any]:
            if operation == "sum":
                result = sum(data)
            elif operation == "average":
                result = sum(data) / len(data) if data else 0
            elif operation == "max":
                result = max(data) if data else 0
            elif operation == "min":
                result = min(data) if data else 0
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return {
                "operation": operation,
                "data_length": len(data),
                "result": result,
                "data_sample": data[:3] if len(data) > 3 else data
            }
    
    # Process registrations
    registration_result = await ensure_registrations()
    print(f"📝 Processed {registration_result['processed']} registrations")
    print(f"   Validation statistics: {registration_result.get('validation_statistics', {})}")
    
    # Test the decorated tools
    registry = await get_registry()
    
    print("\n🧪 Testing decorated validation tools:")
    
    # Test add_numbers
    add_tool = await registry.get_tool("add_numbers", "validation_demo")
    if add_tool:
        result = await add_tool.execute(a=15, b=25)
        print(f"➕ add_numbers(15, 25) = {result}")
    
    # Test strict_multiply
    multiply_tool = await registry.get_tool("strict_multiply", "validation_demo")
    if multiply_tool:
        result = await multiply_tool.execute(a=3.5, b=2.0)
        print(f"✖️  strict_multiply(3.5, 2.0) = {result}")
    
    # Test data_processor (register instance first)
    processor_instance = DataProcessor()
    await registry.register_tool(
        processor_instance,
        name="data_processor_instance",
        namespace="validation_demo",
        metadata={"source": "decorated_instance"}
    )
    
    processor = await registry.get_tool("data_processor_instance", "validation_demo")
    if processor:
        result = await processor.execute(data=[1, 2, 3, 4, 5], operation="average")
        print(f"📊 data_processor([1,2,3,4,5], 'average') = {result}")
    
    print("\n❌ Testing validation errors:")
    
    # Test validation errors
    try:
        await add_tool.execute(a="not_int", b=5)  # Should fail validation
    except Exception as e:
        print(f"   add_numbers validation error: {type(e).__name__}")
    
    try:
        await processor.execute(data="not_list", operation="sum")  # Should fail validation
    except Exception as e:
        print(f"   data_processor validation error: {type(e).__name__}")
    
    print("✅ Decorator validation complete!\n")


async def example_function_registration_validation():
    """Example: Function registration with validation."""
    print("🎯 Example: Function Registration with Validation")
    print("=" * 50)
    
    from chuk_tool_registry.discovery.auto_register import (
        register_fn_tool,
        get_validation_statistics,
        validate_tool_function,
        ValidationConfig
    )
    from chuk_tool_registry import get_registry
    
    # Define functions with type hints for validation
    async def calculate_statistics(numbers: List[float], include_std: bool = True) -> Dict[str, float]:
        """Calculate statistical measures for a list of numbers."""
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
    
    async def validate_email(email: str, check_domain: bool = False) -> Dict[str, Any]:
        """Validate email format with optional domain checking."""
        is_valid = "@" in email and "." in email.split("@")[-1]
        
        result = {
            "email": email,
            "is_valid": is_valid,
            "has_at": "@" in email,
            "has_dot_in_domain": "." in email.split("@")[-1] if "@" in email else False
        }
        
        if check_domain and is_valid:
            domain = email.split("@")[-1]
            # Simple domain validation (just checking format)
            result["domain"] = domain
            result["domain_valid"] = len(domain.split(".")) >= 2
        
        return result
    
    # Validate functions before registration
    print("🔍 Validating functions for compatibility:")
    
    stats_validation = validate_tool_function(calculate_statistics)
    email_validation = validate_tool_function(validate_email)
    
    print(f"   calculate_statistics: {'✅' if stats_validation['validation_compatible'] else '❌'} validation compatible")
    print(f"   validate_email: {'✅' if email_validation['validation_compatible'] else '❌'} validation compatible")
    
    # Register functions with different validation configs
    await register_fn_tool(
        calculate_statistics,
        name="calculate_statistics",
        namespace="analytics",
        enable_validation=True,
        validation_config=ValidationConfig(
            validate_arguments=True,
            validate_results=True,
            strict_mode=False
        ),
        description="Calculate statistical measures with validation"
    )
    
    await register_fn_tool(
        validate_email,
        name="validate_email",
        namespace="validation",
        enable_validation=True,
        validation_config=ValidationConfig(
            validate_arguments=True,
            validate_results=False,  # Don't validate output format
            strict_mode=True
        ),
        description="Validate email addresses"
    )
    
    print(f"\n✅ Registered functions with validation")
    
    # Test the registered functions
    registry = await get_registry()
    
    print("\n🧪 Testing validated functions:")
    
    # Test statistics calculator
    stats_tool = await registry.get_tool("calculate_statistics", "analytics")
    if stats_tool:
        test_data = [1.5, 2.3, 3.7, 4.1, 5.9, 2.8, 3.2]
        result = await stats_tool.execute(numbers=test_data, include_std=True)
        print(f"📈 Statistics for {test_data[:3]}... = {result}")
    
    # Test email validator
    email_tool = await registry.get_tool("validate_email", "validation")
    if email_tool:
        result = await email_tool.execute(email="user@example.com", check_domain=True)
        print(f"✉️  Email validation: {result}")
    
    print("\n❌ Testing validation errors:")
    
    try:
        await stats_tool.execute(numbers="not_a_list")  # Should fail
    except Exception as e:
        print(f"   Statistics validation error: {type(e).__name__}")
    
    try:
        await email_tool.execute(email=123)  # Should fail
    except Exception as e:
        print(f"   Email validation error: {type(e).__name__}")
    
    # Get validation statistics
    validation_stats = await get_validation_statistics()
    print(f"\n📊 Validation Statistics:")
    print(f"   Total tools: {validation_stats['total_tools']}")
    print(f"   Validation enabled: {validation_stats['validation_enabled_tools']}")
    print(f"   Validation percentage: {validation_stats['validation_percentage']:.1f}%")
    print(f"   Tools with schemas: {validation_stats['tools_with_schemas']}")
    
    print("✅ Function registration validation complete!\n")


async def example_registry_validation_features():
    """Example: Registry-level validation features."""
    print("🏗️ Example: Registry-Level Validation Features")
    print("=" * 50)
    
    from chuk_tool_registry import get_registry
    from chuk_tool_registry.discovery.auto_register import validate_registry_tools
    
    registry = await get_registry()
    
    # Check if registry supports validation features
    supports_validation = hasattr(registry, 'execute_tool_with_validation')
    supports_validation_query = hasattr(registry, 'tool_has_validation')
    supports_validation_listing = hasattr(registry, 'list_validation_enabled_tools')
    
    print(f"🔍 Registry validation capabilities:")
    print(f"   Execute with validation: {'✅' if supports_validation else '❌'}")
    print(f"   Query validation status: {'✅' if supports_validation_query else '❌'}")
    print(f"   List validation tools: {'✅' if supports_validation_listing else '❌'}")
    
    # List all tools and their validation status
    all_tools = await registry.list_tools()
    print(f"\n📋 All registered tools ({len(all_tools)}):")
    
    for namespace, tool_name in all_tools[:5]:  # Show first 5
        has_validation = False
        if supports_validation_query:
            has_validation = await registry.tool_has_validation(tool_name, namespace)
        
        validation_indicator = "🛡️" if has_validation else "🔓"
        print(f"   {validation_indicator} {namespace}.{tool_name}")
    
    if len(all_tools) > 5:
        print(f"   ... and {len(all_tools) - 5} more tools")
    
    # Get validation-enabled tools if supported
    if supports_validation_listing:
        validation_tools = await registry.list_validation_enabled_tools()
        print(f"\n🛡️  Validation-enabled tools ({len(validation_tools)}):")
        for namespace, tool_name in validation_tools:
            print(f"   • {namespace}.{tool_name}")
    
    # Validate all tools in registry
    validation_report = await validate_registry_tools()
    print(f"\n📊 Registry Validation Report:")
    print(f"   Total tools analyzed: {validation_report['total_tools']}")
    print(f"   Compatible with validation: {validation_report['compatible_tools']}")
    print(f"   Currently validation-enabled: {validation_report['validation_enabled']}")
    print(f"   Compatibility rate: {validation_report['compatibility_percentage']:.1f}%")
    print(f"   Adoption rate: {validation_report['validation_adoption']:.1f}%")
    
    if validation_report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in validation_report['recommendations'][:3]:  # Show first 3
            print(f"   • {rec}")
        if len(validation_report['recommendations']) > 3:
            print(f"   ... and {len(validation_report['recommendations']) - 3} more")
    
    # Registry health check
    if hasattr(registry, 'validate_all_tools'):
        health_report = await registry.validate_all_tools()
        print(f"\n🏥 Registry Health Check:")
        print(f"   Registry healthy: {'✅' if health_report['registry_healthy'] else '❌'}")
        print(f"   Issues found: {health_report['issues_found']}")
        print(f"   Validation percentage: {health_report['validation_percentage']:.1f}%")
        
        if health_report['issues']:
            print(f"   Issues: {health_report['issues'][:2]}")  # Show first 2 issues
    
    print("✅ Registry validation features complete!\n")


async def example_validation_best_practices():
    """Example: Validation best practices and patterns."""
    print("⭐ Example: Validation Best Practices")
    print("=" * 50)
    
    from chuk_tool_registry.core.validation import ValidationConfig
    from chuk_tool_registry.discovery.auto_register import create_validated_function_tool
    
    print("💡 Validation Best Practices:")
    print()
    
    print("1. 📋 Always use type hints for validation compatibility")
    print("   ✅ Good: async def process(data: List[str]) -> Dict[str, int]")
    print("   ❌ Bad:  async def process(data) -> dict")
    print()
    
    print("2. 🎛️ Choose appropriate validation strictness")
    print("   • Strict mode: For production APIs requiring exact types")
    print("   • Lenient mode: For development and flexible data processing")
    print()
    
    print("3. 🔧 Configure validation per use case")
    
    # Example configurations
    configs = {
        "Development": ValidationConfig(
            validate_arguments=True,
            validate_results=False,
            strict_mode=False,
            allow_extra_args=True,
            coerce_types=True
        ),
        "Production API": ValidationConfig(
            validate_arguments=True,
            validate_results=True,
            strict_mode=True,
            allow_extra_args=False,
            coerce_types=False
        ),
        "Data Processing": ValidationConfig(
            validate_arguments=True,
            validate_results=False,
            strict_mode=False,
            allow_extra_args=True,
            coerce_types=True
        )
    }
    
    for use_case, config in configs.items():
        print(f"   {use_case}:")
        print(f"     • Strict mode: {config.strict_mode}")
        print(f"     • Validate results: {config.validate_results}")
        print(f"     • Allow extra args: {config.allow_extra_args}")
    
    print()
    print("4. 🚀 Performance considerations")
    print("   • Enable validation in development, consider disabling in high-throughput production")
    print("   • Use result validation sparingly for performance-critical paths")
    print("   • Validation overhead is typically < 1ms for simple types")
    print()
    
    print("5. 🛡️ Error handling patterns")
    print("   • Always catch ToolValidationError for user-friendly messages")
    print("   • Log validation errors for debugging")
    print("   • Provide fallback behavior when appropriate")
    print()
    
    # Demonstrate creating validated tools programmatically
    print("6. 🔨 Creating validated tools programmatically:")
    
    async def sample_function(x: int, y: int) -> int:
        return x + y
    
    validated_tool = create_validated_function_tool(
        sample_function,
        name="sample_tool",
        description="Sample tool with validation",
        validation_config=configs["Production API"]
    )
    
    print(f"   Created validated tool: {validated_tool}")
    print(f"   Validation enabled: {validated_tool.enable_validation}")
    
    # Test the tool
    try:
        result = await validated_tool.execute(x=5, y=10)
        print(f"   Test execution: sample_tool(5, 10) = {result}")
    except Exception as e:
        print(f"   Test failed: {e}")
    
    print()
    print("📚 Additional Resources:")
    print("   • Pydantic documentation: https://pydantic-docs.helpmanual.io/")
    print("   • Type hints guide: https://docs.python.org/3/library/typing.html")
    print("   • chuk_tool_registry validation docs: [internal documentation]")
    
    print("\n✅ Best practices overview complete!\n")


async def main():
    """Run all validation integration examples."""
    examples = [
        example_enhanced_registry_setup,
        example_decorator_validation,
        example_function_registration_validation,
        example_registry_validation_features,
        example_validation_best_practices,
    ]
    
    print("🔍 Chuk Tool Registry - Validation Integration Examples")
    print("=" * 70)
    print()
    
    for example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"❌ {example_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("🎉 All validation integration examples completed!")
    print("\n💡 Key Takeaways:")
    print("   ✅ Validation can be enabled per tool or registry-wide")
    print("   ✅ Multiple validation configurations support different use cases")
    print("   ✅ Registry provides comprehensive validation monitoring")
    print("   ✅ Type hints enable automatic schema generation")
    print("   ✅ Graceful degradation when Pydantic is not available")


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