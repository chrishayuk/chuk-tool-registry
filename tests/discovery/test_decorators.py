"""
Simplified tests for decorator functionality.
"""
import pytest
import asyncio

from chuk_tool_registry.discovery.decorators import (
    register_tool,
    ensure_registrations,
    _PENDING_REGISTRATIONS,
    _REGISTERED_CLASSES,
)
from chuk_tool_registry.core.provider import ToolRegistryProvider


class TestRegisterDecorator:
    """Test the @register_tool decorator."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_basic_registration(self):
        """Test basic tool registration with decorator."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define tool with decorator
        @register_tool("test_tool")
        class TestTool:
            async def execute(self, message: str) -> str:
                return f"Tool says: {message}"
        
        # Should be in pending registrations
        assert len(_PENDING_REGISTRATIONS) == 1
        
        # Process registrations
        await ensure_registrations()
        
        # Should be registered
        tool_class = await registry.get_tool("test_tool")
        assert tool_class is not None
        
        # Create instance and test
        tool_instance = tool_class()
        result = await tool_instance.execute(message="hello")
        assert result == "Tool says: hello"
    
    @pytest.mark.asyncio
    async def test_custom_namespace(self):
        """Test registration with custom namespace."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define tool in custom namespace
        @register_tool("custom_tool", namespace="custom")
        class CustomTool:
            async def execute(self, data: str) -> str:
                return f"Custom: {data}"
        
        # Process registrations
        await ensure_registrations()
        
        # Should be in custom namespace
        tool_class = await registry.get_tool("custom_tool", namespace="custom")
        assert tool_class is not None
        
        # Should not be in default namespace
        default_tool = await registry.get_tool("custom_tool")
        assert default_tool is None
    
    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        """Test registering multiple tools."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define multiple tools
        @register_tool("tool1")
        class Tool1:
            async def execute(self, x: int) -> int:
                return x * 2
        
        @register_tool("tool2")
        class Tool2:
            async def execute(self, text: str) -> str:
                return text.upper()
        
        # Should have 2 pending registrations
        assert len(_PENDING_REGISTRATIONS) == 2
        
        # Process all
        await ensure_registrations()
        
        # Both should be registered
        tool1_class = await registry.get_tool("tool1")
        tool2_class = await registry.get_tool("tool2")
        
        assert tool1_class is not None
        assert tool2_class is not None
        
        # Test both work
        tool1 = tool1_class()
        tool2 = tool2_class()
        
        result1 = await tool1.execute(x=5)
        result2 = await tool2.execute(text="hello")
        
        assert result1 == 10
        assert result2 == "HELLO"
    
    def test_sync_tool_error(self):
        """Test that sync tools raise an error."""
        with pytest.raises(TypeError, match="must have an async execute method"):
            @register_tool("sync_tool")
            class SyncTool:
                def execute(self, data: str) -> str:
                    return f"Sync: {data}"
    
    @pytest.mark.asyncio
    async def test_auto_name_from_class(self):
        """Test automatic name from class name."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define tool without explicit name
        @register_tool()
        class AutoNameTool:
            async def execute(self, data: str) -> str:
                return f"Auto: {data}"
        
        # Process registration
        await ensure_registrations()
        
        # Should use class name
        tool_class = await registry.get_tool("AutoNameTool")
        assert tool_class is not None


class TestPydanticCompatibility:
    """Test Pydantic model compatibility."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_pydantic_tool_registration(self):
        """Test that Pydantic tools can be registered."""
        from pydantic import BaseModel
        
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define Pydantic tool
        @register_tool("pydantic_tool")
        class PydanticTool(BaseModel):
            config: str = "default"
            
            async def execute(self, input_data: str) -> str:
                return f"Pydantic: {input_data} (config: {self.config})"
        
        # Process registration
        await ensure_registrations()
        
        # Should be registered
        tool_class = await registry.get_tool("pydantic_tool")
        assert tool_class is not None
        
        # Should have tool_name property
        tool_instance = tool_class()
        assert hasattr(tool_instance, 'tool_name')
        
        # Should work
        result = await tool_instance.execute(input_data="test")
        assert "Pydantic: test" in result


class TestEnsureRegistrations:
    """Test the ensure_registrations function."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_empty_registrations(self):
        """Test with no pending registrations."""
        # Should not error
        await ensure_registrations()
        assert len(_PENDING_REGISTRATIONS) == 0
    
    @pytest.mark.asyncio
    async def test_clears_pending_list(self):
        """Test that pending list is cleared."""
        # Add some dummy registrations
        async def dummy():
            pass
        
        _PENDING_REGISTRATIONS.extend([dummy, dummy])
        assert len(_PENDING_REGISTRATIONS) == 2
        
        # Process registrations
        await ensure_registrations()
        
        # Should be cleared
        assert len(_PENDING_REGISTRATIONS) == 0


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete decorator workflow."""
    # Reset everything
    _PENDING_REGISTRATIONS.clear() 
    _REGISTERED_CLASSES.clear()
    await ToolRegistryProvider.set_registry(None)
    registry = await ToolRegistryProvider.get_registry()
    
    # Define some tools
    @register_tool("processor")
    class TextProcessor:
        async def execute(self, text: str, operation: str = "upper") -> str:
            if operation == "upper":
                return text.upper()
            elif operation == "lower":
                return text.lower()
            else:
                return text
    
    @register_tool("calculator")
    class Calculator:
        async def execute(self, a: int, b: int, op: str = "add") -> int:
            if op == "add":
                return a + b
            elif op == "multiply":
                return a * b
            else:
                return 0
    
    # Process all registrations
    await ensure_registrations()
    
    # Test tools work
    processor_class = await registry.get_tool("processor")
    calculator_class = await registry.get_tool("calculator")
    
    processor = processor_class()
    calculator = calculator_class()
    
    text_result = await processor.execute(text="hello world", operation="upper")
    math_result = await calculator.execute(a=5, b=3, op="multiply")
    
    assert text_result == "HELLO WORLD"
    assert math_result == 15
    
    # Check registry state
    tools = await registry.list_tools()
    tool_names = {name for namespace, name in tools}
    assert "processor" in tool_names
    assert "calculator" in tool_names