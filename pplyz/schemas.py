"""Dynamic Pydantic schema generation for structured LLM outputs."""

import json
from typing import Dict, List, Type

from pydantic import BaseModel, Field, create_model


def parse_type_string(type_str: str) -> Type:
    """Parse a type string into a Python type.

    Only CSV-friendly scalar types are supported: str, int, float, bool.

    Args:
        type_str: Type string like "str", "int", "float", "bool"

    Returns:
        Python type object

    Examples:
        >>> parse_type_string("str")
        <class 'str'>
    """
    type_str = type_str.strip().lower()

    type_mapping = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "bool": bool,
        "boolean": bool,
    }

    if type_str in type_mapping:
        return type_mapping[type_str]

    raise ValueError(
        f"Unsupported type: {type_str}. Allowed types: bool, int, float, str."
    )


def parse_schema_string(schema_str: str) -> Dict[str, Type]:
    """Parse a schema string into field definitions.

    Supports two formats:
    1. Simple: "field1,field2,field3" (all strings)
    2. Typed: "field1:str,field2:int,field3:list[str]"

    Args:
        schema_str: Schema string

    Returns:
        Dictionary mapping field names to types

    Examples:
        >>> parse_schema_string("name,age,tags")
        {'name': str, 'age': str, 'tags': str}
        >>> parse_schema_string("name:str,age:int,tags:list[str]")
        {'name': str, 'age': int, 'tags': List[str]}
    """
    fields = {}

    for field_spec in schema_str.split(","):
        field_spec = field_spec.strip()

        if ":" in field_spec:
            # Typed format: "field:type"
            field_name, type_str = field_spec.split(":", 1)
            field_name = field_name.strip()
            type_str = type_str.strip()
            fields[field_name] = parse_type_string(type_str)
        else:
            # Simple format: assume string type
            fields[field_spec] = str

    return fields


def parse_schema_json(schema_json: str) -> Dict[str, Type]:
    """Parse a JSON schema string into field definitions.

    Args:
        schema_json: JSON string like '{"name": "str", "age": "int"}'

    Returns:
        Dictionary mapping field names to types

    Examples:
        >>> parse_schema_json('{"name": "str", "age": "int"}')
        {'name': str, 'age': int}
    """
    schema_dict = json.loads(schema_json)
    fields = {}

    for field_name, type_str in schema_dict.items():
        fields[field_name] = parse_type_string(type_str)

    return fields


def create_output_model(
    fields: Dict[str, Type], model_name: str = "DynamicOutput"
) -> Type[BaseModel]:
    """Create a Pydantic model dynamically from field definitions.

    Args:
        fields: Dictionary mapping field names to types
        model_name: Name for the generated model

    Returns:
        Pydantic model class

    Examples:
        >>> fields = {"name": str, "age": int}
        >>> Model = create_output_model(fields)
        >>> instance = Model(name="Alice", age=30)
        >>> instance.name
        'Alice'
    """
    # Convert fields dict to format expected by create_model
    # (field_name, (type, default_value))
    field_definitions = {
        name: (field_type, Field(...))  # ... means required field
        for name, field_type in fields.items()
    }

    return create_model(model_name, **field_definitions)  # type: ignore


def create_output_model_from_string(
    schema_str: str, model_name: str = "DynamicOutput"
) -> Type[BaseModel]:
    """Create a Pydantic model from a schema string.

    Args:
        schema_str: Schema string (simple or typed format)
        model_name: Name for the generated model

    Returns:
        Pydantic model class

    Examples:
        >>> Model = create_output_model_from_string("name:str,age:int")
        >>> instance = Model(name="Bob", age=25)
    """
    fields = parse_schema_string(schema_str)
    return create_output_model(fields, model_name)


def create_output_model_from_json(
    schema_json: str, model_name: str = "DynamicOutput"
) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema string.

    Args:
        schema_json: JSON schema string
        model_name: Name for the generated model

    Returns:
        Pydantic model class

    Examples:
        >>> Model = create_output_model_from_json('{"name": "str"}')
        >>> instance = Model(name="Charlie")
    """
    fields = parse_schema_json(schema_json)
    return create_output_model(fields, model_name)


def get_field_names(model: Type[BaseModel]) -> List[str]:
    """Get field names from a Pydantic model.

    Args:
        model: Pydantic model class

    Returns:
        List of field names
    """
    return list(model.model_fields.keys())
