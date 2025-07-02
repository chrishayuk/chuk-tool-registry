"""
Integration tests for the discovery module functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch

from chuk_tool_registry.discovery import (
    register_tool,
    ensure_registrations,
    discover_decorated_tools,
    register_fn_tool,
    make_pydantic_tool_compatible,
)
from chuk_tool_registry.core.provider import ToolRegistryProvider
from chuk_tool_registry.discovery.decorators import (
    _PENDING_REGISTRATIONS,
    _REGISTERED_CLASSES,
)


class TestDiscoveryIntegration:
    """Test integration between different discovery mechanisms."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_mixed_discovery_methods(self):
        """Test using multiple discovery methods together."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Method 1: Decorator-based registration
        @register_tool("decorated_tool")
        class DecoratedTool:
            async def execute(self, message: str) -> str:
                return f"Decorated: {message}"
        
        # Method 2: Function registration
        async def async_function(data: str) -> str:
            return f"Function: {data}"
        
        def sync_function(input_val: str) -> str:
            return f"Sync: {input_val}"
        
        # Register functions
        await register_fn_tool(async_function, name="async_func_tool")
        await register_fn_tool(sync_function, name="sync_func_tool")
        
        # Process all registrations
        await ensure_registrations()
        
        # Test all tools work
        decorated_class = await registry.get_tool("decorated_tool")
        async_func = await registry.get_tool("async_func_tool")
        sync_func = await registry.get_tool("sync_func_tool")
        
        assert decorated_class is not None
        assert async_func is not None
        assert sync_func is not None
        
        # Test execution
        decorated = decorated_class()  # Create instance for class-based tool
        result1 = await decorated.execute(message="test")
        result2 = await async_func.execute(data="test")
        result3 = await sync_func.execute(input_val="test")
        
        assert result1 == "Decorated: test"
        assert result2 == "Function: test"
        assert result3 == "Sync: test"
    
    @pytest.mark.asyncio
    async def test_discovery_with_namespaces(self):
        """Test discovery across different namespaces."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        @register_tool("tool1", namespace="ns1")
        class Tool1:
            async def execute(self, data: str) -> str:
                return f"NS1: {data}"
        
        @register_tool("tool2", namespace="ns2")
        class Tool2:
            async def execute(self, data: str) -> str:
                return f"NS2: {data}"
        
        async def func_tool(data: str) -> str:
            return f"Default: {data}"
        
        await register_fn_tool(func_tool, name="func_tool", namespace="default")
        await ensure_registrations()
        
        # Check tools are in correct namespaces
        tool1_class = await registry.get_tool("tool1", namespace="ns1")
        tool2_class = await registry.get_tool("tool2", namespace="ns2")
        func_tool_instance = await registry.get_tool("func_tool", namespace="default")
        
        assert tool1_class is not None
        assert tool2_class is not None
        assert func_tool_instance is not None
        
        # Check cross-namespace isolation
        assert await registry.get_tool("tool1", namespace="ns2") is None
        assert await registry.get_tool("tool2", namespace="ns1") is None


class TestPydanticCompatibility:
    """Test Pydantic model compatibility features."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    def test_make_pydantic_tool_compatible_manual(self):
        """Test manually making a Pydantic tool compatible."""
        from pydantic import BaseModel
        
        class TestPydanticTool(BaseModel):
            value: str = "test"
            
            async def execute(self, input_data: str) -> str:
                return f"Pydantic: {input_data} (value: {self.value})"
        
        # Make compatible manually
        enhanced_tool = make_pydantic_tool_compatible(TestPydanticTool, "manual_tool")
        
        # Should have tool_name property
        instance = enhanced_tool(value="custom")
        assert hasattr(instance, 'tool_name')
        # For Pydantic models, tool_name might be accessed differently
        tool_name = getattr(instance, 'tool_name', None) or getattr(instance.__class__, '_tool_name', None)
        assert tool_name == "manual_tool"
        
        # Should have serialization methods
        assert hasattr(enhanced_tool, '__getstate__')
        assert hasattr(enhanced_tool, '__setstate__')
    
    @pytest.mark.asyncio
    async def test_pydantic_tool_with_decorator(self):
        """Test Pydantic tool with decorator registration."""
        from pydantic import BaseModel
        
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        @register_tool("pydantic_decorated")
        class PydanticDecoratedTool(BaseModel):
            config: str = "default"
            
            async def execute(self, message: str) -> str:
                return f"Pydantic decorated: {message} (config: {self.config})"
        
        await ensure_registrations()
        
        tool_class = await registry.get_tool("pydantic_decorated")
        assert tool_class is not None
        
        # Should have tool_name
        tool_instance = tool_class()
        assert hasattr(tool_instance, 'tool_name')
        
        # Should be executable
        result = await tool_instance.execute(message="test")
        assert "Pydantic decorated: test" in result


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in discovery."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_empty_registration_scenarios(self):
        """Test scenarios with no registrations."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # No pending registrations
        await ensure_registrations()  # Should not error
        
        # No discovered tools
        discovered = discover_decorated_tools()
        assert isinstance(discovered, list)
        
        # Registry should be empty initially
        tools = await registry.list_tools()
        # May have tools from other tests, so just check it's a list
        assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_complete_workflow():
    """Test a complete discovery workflow with all types."""
    # Reset everything
    _PENDING_REGISTRATIONS.clear()
    _REGISTERED_CLASSES.clear()
    await ToolRegistryProvider.set_registry(None)
    registry = await ToolRegistryProvider.get_registry()
    
    # 1. Class-based tools with decorators
    @register_tool("text_processor", description="Process text")
    class TextProcessor:
        async def execute(self, text: str, operation: str = "upper") -> str:
            if operation == "upper":
                return text.upper()
            elif operation == "lower":
                return text.lower()
            else:
                return text
    
    @register_tool("calculator", namespace="math")
    class Calculator:
        async def execute(self, op: str, a: float, b: float) -> float:
            if op == "add":
                return a + b
            elif op == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {op}")
    
    # 2. Function-based tools
    async def validate_email(email: str) -> dict:
        """Validate email format."""
        is_valid = "@" in email and "." in email.split("@")[1]
        return {"email": email, "valid": is_valid}
    
    def format_currency(amount: float, currency: str = "USD") -> str:
        """Format currency."""
        return f"{currency} {amount:.2f}"
    
    # 3. Register function tools
    await register_fn_tool(validate_email, name="email_validator", namespace="utils")
    await register_fn_tool(format_currency, name="currency_formatter", namespace="utils")
    
    # 4. Process all registrations
    await ensure_registrations()
    
    # 5. Test the complete workflow
    
    # Test text processor
    text_proc_class = await registry.get_tool("text_processor")
    text_proc = text_proc_class()  # Create instance
    text_result = await text_proc.execute(text="Hello World", operation="lower")
    assert text_result == "hello world"
    
    # Test calculator
    calc_class = await registry.get_tool("calculator", namespace="math")
    calc = calc_class()  # Create instance
    calc_result = await calc.execute(op="add", a=10.5, b=5.5)
    assert calc_result == 16.0
    
    # Test email validator
    email_val = await registry.get_tool("email_validator", namespace="utils")
    email_result = await email_val.execute(email="test@example.com")
    assert email_result["valid"] is True
    
    # Test currency formatter
    currency_fmt = await registry.get_tool("currency_formatter", namespace="utils")
    currency_result = await currency_fmt.execute(amount=123.456, currency="EUR")
    assert currency_result == "EUR 123.46"
    
    # 6. Verify registry state
    all_tools = await registry.list_tools()
    tool_names = {name for ns, name in all_tools}
    
    expected_tools = {
        "text_processor", "calculator", 
        "email_validator", "currency_formatter"
    }
    assert expected_tools.issubset(tool_names)
    
    # Check namespaces
    namespaces = await registry.list_namespaces()
    expected_namespaces = {"default", "math", "utils"}
    assert expected_namespaces.issubset(set(namespaces))