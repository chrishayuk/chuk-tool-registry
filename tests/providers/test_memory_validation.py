# tests/providers/test_memory_validation.py
"""
Tests for the enhanced InMemoryToolRegistry with validation support.
"""
import pytest
from typing import List, Dict, Any

from chuk_tool_registry.providers.memory import InMemoryToolRegistry
from chuk_tool_registry.core.validation import ValidationConfig
from chuk_tool_registry.core.exceptions import ToolValidationError, ToolNotFoundError


class TestInMemoryToolRegistryValidation:
    """Test validation features of InMemoryToolRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a registry with validation enabled."""
        return InMemoryToolRegistry(
            enable_statistics=True,
            validate_tools=True,
            enable_validation_by_default=False
        )

    @pytest.fixture
    def validation_registry(self):
        """Create a registry with validation enabled by default."""
        config = ValidationConfig(
            validate_arguments=True,
            validate_results=True,
            allow_extra_args=True
        )
        return InMemoryToolRegistry(
            enable_statistics=True,
            validate_tools=True,
            default_validation_config=config,
            enable_validation_by_default=True
        )

    @pytest.mark.asyncio
    async def test_register_tool_with_validation(self, registry):
        """Test registering a tool with validation enabled."""
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        validation_config = ValidationConfig(strict_mode=True)
        
        await registry.register_tool(
            tool,
            name="test_tool",
            namespace="test",
            enable_validation=True,
            validation_config=validation_config,
            metadata={"description": "Test tool with validation"}
        )
        
        # Check that tool is registered
        registered_tool = await registry.get_tool("test_tool", "test")
        assert registered_tool is not None
        
        # Check that validation is enabled
        has_validation = await registry.tool_has_validation("test_tool", "test")
        assert has_validation is True
        
        # Check validation config
        config = await registry.get_validation_config("test_tool", "test")
        assert config is not None
        assert config.strict_mode is True

    @pytest.mark.asyncio
    async def test_register_tool_without_validation(self, registry):
        """Test registering a tool without validation."""
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        
        await registry.register_tool(
            tool,
            name="test_tool",
            namespace="test",
            enable_validation=False,
            metadata={"description": "Test tool without validation"}
        )
        
        # Check that validation is not enabled
        has_validation = await registry.tool_has_validation("test_tool", "test")
        assert has_validation is False
        
        # Check no validation config
        config = await registry.get_validation_config("test_tool", "test")
        assert config is None

    @pytest.mark.asyncio
    async def test_execute_tool_with_validation_success(self, registry):
        """Test successful execution with validation."""
        class CalculatorTool:
            async def execute(self, a: int, b: int, operation: str = "add") -> Dict[str, Any]:
                if operation == "add":
                    result = a + b
                elif operation == "multiply":
                    result = a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                return {"result": result, "operation": operation}
        
        tool = CalculatorTool()
        validation_config = ValidationConfig(allow_extra_args=True)
        
        await registry.register_tool(
            tool,
            name="calculator",
            namespace="math",
            enable_validation=True,
            validation_config=validation_config
        )
        
        # Test successful execution
        result = await registry.execute_tool_with_validation(
            "calculator", "math",
            a=5, b=3, operation="add"
        )
        
        assert result["result"] == 8
        assert result["operation"] == "add"

    @pytest.mark.asyncio
    async def test_execute_tool_with_validation_error(self, registry):
        """Test execution with validation error."""
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        
        await registry.register_tool(
            tool,
            name="test_tool",
            namespace="test",
            enable_validation=True
        )
        
        # Test validation error
        with pytest.raises(ToolValidationError):
            await registry.execute_tool_with_validation(
                "test_tool", "test",
                a="not_int", b="hello"  # Wrong type for 'a'
            )

    @pytest.mark.asyncio
    async def test_get_original_tool(self, registry):
        """Test getting original tool without validation wrapper."""
        class TestTool:
            async def execute(self, a: int) -> int:
                return a * 2
        
        tool = TestTool()
        
        await registry.register_tool(
            tool,
            name="test_tool",
            namespace="test",
            enable_validation=True
        )
        
        # Get wrapped tool
        wrapped_tool = await registry.get_tool("test_tool", "test")
        
        # Get original tool
        original_tool = await registry.get_original_tool("test_tool", "test")
        
        # They should be different objects
        assert wrapped_tool is not original_tool
        assert original_tool is tool

    @pytest.mark.asyncio
    async def test_list_validation_enabled_tools(self, registry):
        """Test listing validation-enabled tools."""
        class Tool1:
            async def execute(self, x: int) -> int:
                return x
        
        class Tool2:
            async def execute(self, x: int) -> int:
                return x * 2
        
        class Tool3:
            async def execute(self, x: int) -> int:
                return x * 3
        
        # Register tools with different validation settings
        await registry.register_tool(Tool1(), "tool1", "test", enable_validation=True)
        await registry.register_tool(Tool2(), "tool2", "test", enable_validation=False)
        await registry.register_tool(Tool3(), "tool3", "test", enable_validation=True)
        
        # List validation-enabled tools
        validation_tools = await registry.list_validation_enabled_tools("test")
        validation_names = [name for ns, name in validation_tools]
        
        assert "tool1" in validation_names
        assert "tool2" not in validation_names
        assert "tool3" in validation_names

    @pytest.mark.asyncio
    async def test_registry_statistics_with_validation(self, registry):
        """Test registry statistics include validation info."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        # Register tools with different validation settings
        await registry.register_tool(TestTool(), "tool1", "test", enable_validation=True)
        await registry.register_tool(TestTool(), "tool2", "test", enable_validation=False)
        await registry.register_tool(TestTool(), "tool3", "test", enable_validation=True)
        
        stats = await registry.get_statistics()
        
        assert stats["total_tools"] == 3
        assert stats["tools_with_validation"] == 2
        assert stats["validation_available"] is True
        assert "validation_enabled_by_default" in stats

    @pytest.mark.asyncio
    async def test_validation_by_default_registry(self, validation_registry):
        """Test registry with validation enabled by default."""
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        
        # Register without specifying validation (should use default)
        await validation_registry.register_tool(
            tool,
            name="test_tool",
            namespace="test"
        )
        
        # Should have validation enabled by default
        has_validation = await validation_registry.tool_has_validation("test_tool", "test")
        assert has_validation is True

    @pytest.mark.asyncio
    async def test_validation_metadata_storage(self, registry):
        """Test that validation settings are stored in metadata."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        tool = TestTool()
        validation_config = ValidationConfig(
            strict_mode=True,
            allow_extra_args=False
        )
        
        await registry.register_tool(
            tool,
            name="test_tool",
            namespace="test",
            enable_validation=True,
            validation_config=validation_config
        )
        
        metadata = await registry.get_metadata("test_tool", "test")
        assert metadata is not None
        
        # Check validation info in execution_options
        exec_options = metadata.execution_options
        assert exec_options["validation_enabled"] is True
        
        validation_config_stored = exec_options["validation_config"]
        assert validation_config_stored["strict_mode"] is True
        assert validation_config_stored["allow_extra_args"] is False

    @pytest.mark.asyncio
    async def test_search_validation_enabled_tools(self, registry):
        """Test searching for validation-enabled tools only."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        await registry.register_tool(
            TestTool(), "search_tool1", "test", 
            enable_validation=True,
            metadata={"description": "Searchable tool with validation"}
        )
        await registry.register_tool(
            TestTool(), "search_tool2", "test",
            enable_validation=False,
            metadata={"description": "Searchable tool without validation"}
        )
        
        # Search for validation-enabled tools only
        results = await registry.search_tools(
            "searchable", "test", 
            validation_only=True
        )
        
        assert len(results) == 1
        assert results[0].name == "search_tool1"

    @pytest.mark.asyncio
    async def test_validate_all_tools_registry_method(self):
        """Test the validate_all_tools registry method."""
        # Create registry without tool validation during registration
        # so we can register tools that don't conform to test the validation method
        test_registry = InMemoryToolRegistry(
            enable_statistics=True,
            validate_tools=False,  # Allow non-conforming tools for testing
            enable_validation_by_default=False
        )
        
        class GoodTool:
            async def execute(self, x: int) -> int:
                return x
        
        class BadTool:
            def no_execute_method(self):
                pass
        
        await test_registry.register_tool(GoodTool(), "good_tool", "test", enable_validation=True)
        await test_registry.register_tool(BadTool(), "bad_tool", "test", enable_validation=False)
        
        validation_report = await test_registry.validate_all_tools()
        
        assert validation_report["total_tools"] == 2
        assert validation_report["registry_healthy"] is False  # Because of bad_tool
        assert len(validation_report["issues"]) > 0

    @pytest.mark.asyncio
    async def test_remove_validation_enabled_tool(self, registry):
        """Test removing a validation-enabled tool."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        await registry.register_tool(
            TestTool(), "test_tool", "test",
            enable_validation=True
        )
        
        # Verify tool exists and has validation
        assert await registry.tool_exists("test_tool", "test")
        assert await registry.tool_has_validation("test_tool", "test")
        
        # Remove tool
        removed = await registry.remove_tool("test_tool", "test")
        assert removed is True
        
        # Verify tool is gone
        assert not await registry.tool_exists("test_tool", "test")
        assert not await registry.tool_has_validation("test_tool", "test")

    @pytest.mark.asyncio
    async def test_clear_namespace_with_validation_tools(self, registry):
        """Test clearing namespace with validation-enabled tools."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        await registry.register_tool(TestTool(), "tool1", "test", enable_validation=True)
        await registry.register_tool(TestTool(), "tool2", "test", enable_validation=False)
        await registry.register_tool(TestTool(), "tool3", "other", enable_validation=True)
        
        # Clear test namespace
        cleared_count = await registry.clear_namespace("test")
        assert cleared_count == 2
        
        # Verify tools are gone from test namespace
        test_tools = await registry.list_tools("test")
        assert len(test_tools) == 0
        
        # Verify other namespace is untouched
        other_tools = await registry.list_tools("other")
        assert len(other_tools) == 1

    @pytest.mark.asyncio
    async def test_concurrent_validation_operations(self, registry):
        """Test concurrent operations with validation."""
        import asyncio
        
        class TestTool:
            async def execute(self, x: int) -> int:
                return x * 2
        
        async def register_and_execute(tool_id: int):
            await registry.register_tool(
                TestTool(), f"tool_{tool_id}", "test",
                enable_validation=True
            )
            
            result = await registry.execute_tool_with_validation(
                f"tool_{tool_id}", "test", x=tool_id
            )
            return result
        
        # Run multiple concurrent operations
        tasks = [register_and_execute(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == i * 2

    @pytest.mark.asyncio
    async def test_validation_error_statistics(self, registry):
        """Test that validation errors are tracked in statistics."""
        class TestTool:
            async def execute(self, x: int) -> int:
                return x
        
        await registry.register_tool(
            TestTool(), "test_tool", "test",
            enable_validation=True
        )
        
        # Get initial stats
        initial_stats = await registry.get_statistics()
        initial_errors = initial_stats.get("validation_errors_count", 0)
        
        # Cause a validation error
        try:
            await registry.execute_tool_with_validation(
                "test_tool", "test", x="not_int"
            )
        except ToolValidationError:
            pass
        
        # Check that error was counted
        updated_stats = await registry.get_statistics()
        final_errors = updated_stats.get("validation_errors_count", 0)
        assert final_errors > initial_errors


class TestValidationIntegrationScenarios:
    """Test real-world validation integration scenarios."""

    @pytest.mark.asyncio
    async def test_data_processing_pipeline_with_validation(self):
        """Test a data processing pipeline with validation."""
        registry = InMemoryToolRegistry(enable_validation_by_default=True)
        
        class DataLoader:
            async def execute(self, source: str) -> List[Dict[str, Any]]:
                # Simulate loading data
                return [
                    {"id": 1, "value": 10},
                    {"id": 2, "value": 20},
                    {"id": 3, "value": 30}
                ]
        
        class DataProcessor:
            async def execute(self, data: List[Dict[str, Any]], multiplier: float = 1.0) -> List[Dict[str, Any]]:
                return [
                    {"id": item["id"], "value": item["value"] * multiplier}
                    for item in data
                ]
        
        class DataAggregator:
            async def execute(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
                total = sum(item["value"] for item in data)
                count = len(data)
                return {"total": total, "average": total / count if count > 0 else 0}
        
        # Register pipeline tools
        await registry.register_tool(DataLoader(), "loader", "pipeline")
        await registry.register_tool(DataProcessor(), "processor", "pipeline")
        await registry.register_tool(DataAggregator(), "aggregator", "pipeline")
        
        # Execute pipeline
        data = await registry.execute_tool_with_validation("loader", "pipeline", source="test")
        processed = await registry.execute_tool_with_validation("processor", "pipeline", data=data, multiplier=2.0)
        result = await registry.execute_tool_with_validation("aggregator", "pipeline", data=processed)
        
        assert result["total"] == 120  # (10+20+30) * 2
        assert result["average"] == 40  # 120 / 3

    @pytest.mark.asyncio
    async def test_api_style_validation_with_strict_mode(self):
        """Test API-style validation with strict mode."""
        strict_config = ValidationConfig(
            strict_mode=True,
            allow_extra_args=False,
            validate_arguments=True,
            validate_results=True
        )
        
        registry = InMemoryToolRegistry(
            default_validation_config=strict_config,
            enable_validation_by_default=True
        )
        
        class UserService:
            async def execute(self, action: str, user_id: int, **kwargs) -> Dict[str, Any]:
                if action == "get":
                    return {"user_id": user_id, "name": f"User {user_id}"}
                elif action == "update":
                    name = kwargs.get("name", f"User {user_id}")
                    return {"user_id": user_id, "name": name, "updated": True}
                else:
                    raise ValueError(f"Unknown action: {action}")
        
        await registry.register_tool(UserService(), "user_service", "api")
        
        # Test successful strict validation
        result = await registry.execute_tool_with_validation(
            "user_service", "api",
            action="get", user_id=123
        )
        assert result["user_id"] == 123
        
        # Test validation failures in strict mode
        with pytest.raises(ToolValidationError):
            # Wrong type for user_id
            await registry.execute_tool_with_validation(
                "user_service", "api",
                action="get", user_id="not_int"
            )
        
        with pytest.raises(ToolValidationError):
            # Extra args not allowed in strict mode
            await registry.execute_tool_with_validation(
                "user_service", "api",
                action="get", user_id=123, extra_arg="not_allowed"
            )

    @pytest.mark.asyncio
    async def test_mixed_validation_environments(self):
        """Test mixing validated and non-validated tools."""
        registry = InMemoryToolRegistry(enable_validation_by_default=False)
        
        class ValidatedTool:
            async def execute(self, strict_value: int) -> Dict[str, int]:
                return {"value": strict_value, "doubled": strict_value * 2}
        
        class LegacyTool:
            async def execute(self, loose_value) -> Any:
                # Legacy tool without type hints
                return f"Processed: {loose_value}"
        
        # Register with different validation settings
        await registry.register_tool(
            ValidatedTool(), "validated", "mixed",
            enable_validation=True
        )
        await registry.register_tool(
            LegacyTool(), "legacy", "mixed",
            enable_validation=False
        )
        
        # Test validated tool
        result1 = await registry.execute_tool_with_validation(
            "validated", "mixed", strict_value=42
        )
        assert result1["doubled"] == 84
        
        # Validated tool should reject invalid input
        with pytest.raises(ToolValidationError):
            await registry.execute_tool_with_validation(
                "validated", "mixed", strict_value="not_int"
            )
        
        # Legacy tool should accept anything
        result2 = await registry.execute_tool_with_validation(
            "legacy", "mixed", loose_value="anything"
        )
        assert "anything" in result2
        
        # Legacy tool with invalid input should still work
        result3 = await registry.execute_tool_with_validation(
            "legacy", "mixed", loose_value={"complex": "data"}
        )
        assert "complex" in str(result3)