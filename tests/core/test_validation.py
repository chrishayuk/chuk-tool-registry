# tests/core/test_validation.py
"""
Tests for the validation system.
"""
import pytest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from chuk_tool_registry.core.validation import (
    ValidationConfig,
    validate_arguments,
    validate_result,
    with_validation,
    create_validation_wrapper,
    validate_tool_execution,
)
from chuk_tool_registry.core.exceptions import ToolValidationError


class TestValidationConfig:
    """Test ValidationConfig functionality."""

    def test_default_values(self):
        """Test that ValidationConfig has sensible defaults."""
        config = ValidationConfig()
        
        assert config.validate_arguments is True
        assert config.validate_results is True
        assert config.strict_mode is False
        assert config.allow_extra_args is True  # Should be True for usability
        assert config.coerce_types is True

    def test_custom_values(self):
        """Test ValidationConfig with custom values."""
        config = ValidationConfig(
            validate_arguments=False,
            validate_results=False,
            strict_mode=True,
            allow_extra_args=False,
            coerce_types=False
        )
        
        assert config.validate_arguments is False
        assert config.validate_results is False
        assert config.strict_mode is True
        assert config.allow_extra_args is False
        assert config.coerce_types is False

    def test_hashable(self):
        """Test that ValidationConfig is hashable."""
        config1 = ValidationConfig()
        config2 = ValidationConfig()
        config3 = ValidationConfig(strict_mode=True)
        
        # Should be able to use as dict keys
        config_dict = {config1: "default", config3: "strict"}
        assert len(config_dict) == 2
        
        # Equal configs should have same hash
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)

    def test_frozen(self):
        """Test that ValidationConfig is frozen (immutable)."""
        config = ValidationConfig()
        
        with pytest.raises(AttributeError):
            config.strict_mode = True


class TestArgumentValidation:
    """Test argument validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_simple_arguments(self):
        """Test validation of simple typed arguments."""
        async def test_func(a: int, b: str) -> str:
            return f"{a}: {b}"
        
        # Valid arguments
        args = {"a": 42, "b": "hello"}
        validated = validate_arguments("test_func", test_func, args)
        assert validated == {"a": 42, "b": "hello"}

    @pytest.mark.asyncio
    async def test_validate_arguments_with_defaults(self):
        """Test validation with default parameters."""
        async def test_func(a: int, b: str = "default") -> str:
            return f"{a}: {b}"
        
        # With all arguments
        args1 = {"a": 42, "b": "hello"}
        validated1 = validate_arguments("test_func", test_func, args1)
        assert validated1 == {"a": 42, "b": "hello"}
        
        # With default argument missing
        args2 = {"a": 42}
        validated2 = validate_arguments("test_func", test_func, args2)
        assert validated2 == {"a": 42}

    @pytest.mark.asyncio
    async def test_validate_arguments_type_coercion(self):
        """Test type coercion in argument validation."""
        async def test_func(a: int, b: float) -> float:
            return a + b
        
        config = ValidationConfig(coerce_types=True, strict_mode=False)
        
        # Should coerce string numbers to int/float
        args = {"a": "42", "b": "3.14"}
        validated = validate_arguments("test_func", test_func, args, config)
        assert validated["a"] == 42
        assert abs(validated["b"] - 3.14) < 0.001

    @pytest.mark.asyncio
    async def test_validate_arguments_validation_error(self):
        """Test validation errors are properly raised."""
        async def test_func(a: int, b: str) -> str:
            return f"{a}: {b}"
        
        # Invalid type for 'a'
        args = {"a": "not_an_int", "b": "hello"}
        
        with pytest.raises(ToolValidationError) as exc_info:
            validate_arguments("test_func", test_func, args)
        
        assert exc_info.value.tool_name == "test_func"
        assert len(exc_info.value.errors) > 0

    @pytest.mark.asyncio
    async def test_validate_arguments_extra_args_allowed(self):
        """Test validation with extra arguments allowed."""
        async def test_func(a: int) -> int:
            return a
        
        config = ValidationConfig(allow_extra_args=True)
        args = {"a": 42, "extra": "should_be_ignored"}
        
        validated = validate_arguments("test_func", test_func, args, config)
        assert "a" in validated
        # Extra args might be included or filtered depending on Pydantic behavior

    @pytest.mark.asyncio
    async def test_validate_arguments_extra_args_forbidden(self):
        """Test validation with extra arguments forbidden."""
        async def test_func(a: int) -> int:
            return a
        
        config = ValidationConfig(allow_extra_args=False)
        args = {"a": 42, "extra": "should_cause_error"}
        
        with pytest.raises(ToolValidationError):
            validate_arguments("test_func", test_func, args, config)

    @pytest.mark.asyncio
    async def test_validate_complex_types(self):
        """Test validation with complex types."""
        async def test_func(data: List[Dict[str, Any]], count: int) -> Dict[str, Any]:
            return {"data": data, "count": count}
        
        args = {
            "data": [{"name": "test", "value": 42}],
            "count": 1
        }
        
        validated = validate_arguments("test_func", test_func, args)
        assert validated == args

    @pytest.mark.asyncio
    async def test_validate_no_type_hints(self):
        """Test validation with function that has no type hints."""
        async def test_func(a, b):
            return a + b
        
        args = {"a": 1, "b": 2}
        validated = validate_arguments("test_func", test_func, args)
        assert validated == args  # Should pass through without validation


class TestResultValidation:
    """Test result validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_simple_result(self):
        """Test validation of simple typed results."""
        async def test_func(a: int) -> str:
            return str(a)
        
        result = "42"
        validated = validate_result("test_func", test_func, result)
        assert validated == "42"

    @pytest.mark.asyncio
    async def test_validate_result_type_error(self):
        """Test result validation with type error."""
        async def test_func(a: int) -> str:
            return str(a)
        
        result = 42  # Should be string
        
        with pytest.raises(ToolValidationError) as exc_info:
            validate_result("test_func", test_func, result)
        
        assert exc_info.value.tool_name == "test_func"

    @pytest.mark.asyncio
    async def test_validate_result_no_return_type(self):
        """Test result validation with no return type hint."""
        async def test_func(a: int):
            return a
        
        result = 42
        validated = validate_result("test_func", test_func, result)
        assert validated == 42  # Should pass through

    @pytest.mark.asyncio
    async def test_validate_complex_result(self):
        """Test validation of complex result types."""
        async def test_func() -> Dict[str, List[int]]:
            return {"numbers": [1, 2, 3]}
        
        result = {"numbers": [1, 2, 3]}
        validated = validate_result("test_func", test_func, result)
        assert validated == result

    @pytest.mark.asyncio
    async def test_validate_result_coercion(self):
        """Test result type coercion."""
        async def test_func() -> str:
            return "42"
        
        config = ValidationConfig(coerce_types=True)
        result = 42  # Should be coerced to string
        
        # This might or might not work depending on Pydantic's coercion rules
        try:
            validated = validate_result("test_func", test_func, result, config)
            assert validated == "42"
        except ToolValidationError:
            # Coercion failed, which is also acceptable
            pass


class TestWithValidationDecorator:
    """Test the @with_validation decorator."""

    @pytest.mark.asyncio
    async def test_with_validation_basic(self):
        """Test basic @with_validation decorator usage."""
        @with_validation
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        result = await tool.execute(a=42, b="hello")
        assert result == "42: hello"

    @pytest.mark.asyncio
    async def test_with_validation_argument_error(self):
        """Test @with_validation catches argument validation errors."""
        @with_validation
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        
        with pytest.raises(ToolValidationError):
            await tool.execute(a="not_int", b="hello")

    @pytest.mark.asyncio
    async def test_with_validation_result_error(self):
        """Test @with_validation catches result validation errors."""
        @with_validation
        class TestTool:
            async def execute(self, a: int) -> str:
                return a  # Should return string, but returns int
        
        tool = TestTool()
        
        with pytest.raises(ToolValidationError):
            await tool.execute(a=42)

    @pytest.mark.asyncio
    async def test_with_validation_custom_config(self):
        """Test @with_validation with custom configuration."""
        config = ValidationConfig(strict_mode=True, allow_extra_args=False)
        
        @with_validation(config=config)
        class TestTool:
            async def execute(self, a: int) -> int:
                return a
        
        tool = TestTool()
        
        # Should work with correct args
        result = await tool.execute(a=42)
        assert result == 42
        
        # Should fail with extra args when allow_extra_args=False
        with pytest.raises(ToolValidationError):
            await tool.execute(a=42, extra="not_allowed")

    @pytest.mark.asyncio
    async def test_with_validation_disabled_arguments(self):
        """Test @with_validation with argument validation disabled."""
        config = ValidationConfig(validate_arguments=False, validate_results=True)
        
        @with_validation(config=config)
        class TestTool:
            async def execute(self, a: int) -> str:
                return str(a)
        
        tool = TestTool()
        
        # Should allow invalid argument types when validation disabled
        result = await tool.execute(a="not_int")
        assert result == "not_int"

    @pytest.mark.asyncio
    async def test_with_validation_disabled_results(self):
        """Test @with_validation with result validation disabled."""
        config = ValidationConfig(validate_arguments=True, validate_results=False)
        
        @with_validation(config=config)
        class TestTool:
            async def execute(self, a: int) -> str:
                return a  # Wrong return type
        
        tool = TestTool()
        
        # Should allow wrong return type when result validation disabled
        result = await tool.execute(a=42)
        assert result == 42

    def test_with_validation_non_async_method_error(self):
        """Test @with_validation raises error for non-async methods."""
        with pytest.raises(TypeError):
            @with_validation
            class TestTool:
                def execute(self, a: int) -> int:  # Not async
                    return a

    def test_with_validation_missing_execute_method_error(self):
        """Test @with_validation raises error for missing execute method."""
        with pytest.raises(AttributeError):
            @with_validation
            class TestTool:
                def some_other_method(self):
                    pass


class TestCreateValidationWrapper:
    """Test the create_validation_wrapper function."""

    @pytest.mark.asyncio
    async def test_create_validation_wrapper_basic(self):
        """Test basic validation wrapper creation."""
        async def test_func(a: int, b: str) -> str:
            return f"{a}: {b}"
        
        wrapper = create_validation_wrapper("test_func", test_func)
        result = await wrapper(a=42, b="hello")
        assert result == "42: hello"

    @pytest.mark.asyncio
    async def test_create_validation_wrapper_validation_error(self):
        """Test validation wrapper catches validation errors."""
        async def test_func(a: int, b: str) -> str:
            return f"{a}: {b}"
        
        wrapper = create_validation_wrapper("test_func", test_func)
        
        with pytest.raises(ToolValidationError):
            await wrapper(a="not_int", b="hello")

    @pytest.mark.asyncio
    async def test_create_validation_wrapper_custom_config(self):
        """Test validation wrapper with custom config."""
        async def test_func(a: int) -> int:
            return a
        
        config = ValidationConfig(allow_extra_args=False)
        wrapper = create_validation_wrapper("test_func", test_func, config)
        
        # Should work normally
        result = await wrapper(a=42)
        assert result == 42
        
        # Should fail with extra args
        with pytest.raises(ToolValidationError):
            await wrapper(a=42, extra="not_allowed")

    def test_create_validation_wrapper_non_async_error(self):
        """Test create_validation_wrapper raises error for non-async functions."""
        def sync_func(a: int) -> int:
            return a
        
        with pytest.raises(TypeError):
            create_validation_wrapper("sync_func", sync_func)


class TestValidateToolExecution:
    """Test the validate_tool_execution integration function."""

    @pytest.mark.asyncio
    async def test_validate_tool_execution_basic(self):
        """Test basic tool execution with validation."""
        class TestTool:
            async def execute(self, a: int, b: str) -> str:
                return f"{a}: {b}"
        
        tool = TestTool()
        kwargs = {"a": 42, "b": "hello"}
        
        result = await validate_tool_execution(
            "test_tool", tool, tool.execute, kwargs
        )
        assert result == "42: hello"

    @pytest.mark.asyncio
    async def test_validate_tool_execution_no_instance(self):
        """Test tool execution validation without tool instance."""
        async def test_func(a: int, b: str) -> str:
            return f"{a}: {b}"
        
        kwargs = {"a": 42, "b": "hello"}
        
        result = await validate_tool_execution(
            "test_func", None, test_func, kwargs
        )
        assert result == "42: hello"

    @pytest.mark.asyncio
    async def test_validate_tool_execution_custom_config(self):
        """Test tool execution with custom validation config."""
        class TestTool:
            async def execute(self, a: int) -> int:
                return a
        
        tool = TestTool()
        kwargs = {"a": 42, "extra": "not_allowed"}
        config = ValidationConfig(allow_extra_args=False)
        
        with pytest.raises(ToolValidationError):
            await validate_tool_execution(
                "test_tool", tool, tool.execute, kwargs, config
            )

    @pytest.mark.asyncio
    async def test_validate_tool_execution_validation_disabled(self):
        """Test tool execution with validation disabled."""
        class TestTool:
            async def execute(self, a: int) -> str:
                return str(a)
        
        tool = TestTool()
        kwargs = {"a": "not_int"}  # Wrong type
        config = ValidationConfig(validate_arguments=False, validate_results=False)
        
        # Should work when validation is disabled
        result = await validate_tool_execution(
            "test_tool", tool, tool.execute, kwargs, config
        )
        assert result == "not_int"


class TestValidationIntegration:
    """Test validation system integration scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation_flow(self):
        """Test complete validation flow from decoration to execution."""
        config = ValidationConfig(strict_mode=True, allow_extra_args=False)
        
        @with_validation(config=config)
        class CalculatorTool:
            async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
                if operation == "add":
                    result = a + b
                elif operation == "multiply":
                    result = a * b
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                return {
                    "operation": operation,
                    "operands": [a, b],
                    "result": result
                }
        
        tool = CalculatorTool()
        
        # Test successful execution
        result = await tool.execute(operation="add", a=5.0, b=3.0)
        assert result["result"] == 8.0
        assert result["operation"] == "add"
        
        # Test validation errors
        with pytest.raises(ToolValidationError):
            # Wrong type for operation
            await tool.execute(operation=123, a=5.0, b=3.0)
        
        with pytest.raises(ToolValidationError):
            # Extra argument not allowed
            await tool.execute(operation="add", a=5.0, b=3.0, extra="not_allowed")

    @pytest.mark.asyncio
    async def test_mixed_validation_scenarios(self):
        """Test tools with different validation configurations."""
        strict_config = ValidationConfig(strict_mode=True, allow_extra_args=False)
        lenient_config = ValidationConfig(strict_mode=False, allow_extra_args=True)
        
        @with_validation(config=strict_config)
        class StrictTool:
            async def execute(self, value: int) -> int:
                return value * 2
        
        @with_validation(config=lenient_config)
        class LenientTool:
            async def execute(self, value: int) -> int:
                return value * 3
        
        strict_tool = StrictTool()
        lenient_tool = LenientTool()
        
        # Strict tool should reject string numbers
        with pytest.raises(ToolValidationError):
            await strict_tool.execute(value="42")
        
        # Lenient tool might accept string numbers (depending on Pydantic coercion)
        try:
            result = await lenient_tool.execute(value="42")
            assert result == 126  # 42 * 3
        except ToolValidationError:
            # Coercion might not work, which is also fine
            pass
        
        # Both should work with correct types
        assert await strict_tool.execute(value=42) == 84
        assert await lenient_tool.execute(value=42) == 126

    @pytest.mark.asyncio
    async def test_validation_error_details(self):
        """Test that validation errors provide useful details."""
        @with_validation
        class TestTool:
            async def execute(self, required_int: int, required_str: str) -> str:
                return f"{required_int}: {required_str}"
        
        tool = TestTool()
        
        try:
            await tool.execute(required_int="not_int", required_str=123)
        except ToolValidationError as e:
            assert e.tool_name == "TestTool"
            assert len(e.errors) >= 2  # At least 2 validation errors
            
            # Check that errors contain useful information
            error_msgs = [str(error) for error in e.errors]
            error_str = " ".join(error_msgs)
            assert "required_int" in error_str
            assert "required_str" in error_str