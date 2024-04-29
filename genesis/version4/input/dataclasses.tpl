#!/usr/bin/env python
# vi: syntax=python sw=4 ts=4 sts=4
"""
This file is auto-generated by lume-genesis (`genesis.version4.input.manual`).

Do not hand-edit it.
"""
from __future__ import annotations
import typing

import pydantic

from typing import Dict, Literal, Union
from . import util
from ..types import {{ base_class }}, Reference, ValueType

{%- for name, element in manual.elements.items() %}
{%- if element.parameters | length %}

{% macro field_value(param) -%}
{%- if "vector" in param.options -%}
    {{ " =" }} pydantic.Field(default_factory=list
    {%- if param.name != param.python_name %},
    validation_alias=pydantic.AliasChoices("{{ param.python_name }}", "{{ param.name }}"),
    serialization_alias="{{ param.name }}"
    {%- endif -%}
    )
{%- elif param.default is none -%}
    {%- if param.python_name != param.name -%}
    {{ " =" }} pydantic.Field(
        validation_alias=pydantic.AliasChoices("{{ param.python_name }}", "{{ param.name }}"),
        serialization_alias="{{ param.name }}"
    )
    {%- endif -%}
{%- else -%}
    {%- if param.python_name != param.name -%}
    {{ " =" }} pydantic.Field(
        validation_alias=pydantic.AliasChoices("{{ param.python_name }}", "{{ param.name }}"),
        serialization_alias="{{ param.name }}",
        default={{ param.default | repr }}
    )
    {%- else -%}
    {{ " = " }}{{ param.default | repr }}
    {%- endif -%}
{%- endif -%}
{%- endmacro%}
class {{ name | to_class_name }}({{ base_class }}):
    r"""
    {%- if element.header %}
    {{ element.header | wordwrap | indent(4) }}
    {%- elif name in docstrings %}
    {{ docstrings[name] | wordwrap | indent(4) }}
    {%- endif %}

    {{ name | to_class_name }} corresponds to Genesis 4 `{{ name }}`.

    Attributes
    ----------
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {{ param.python_name }} : {{ type_ }}{% if not param.default is none %}, default={{ param.default | repr }}{% endif %}
        {{ param.description | wordwrap | indent(8) }}
    {%- endfor %}
    """

    type: Literal["{{ name }}"] = "{{ name }}"
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {%- if "reference" in param.options %}
    {%- set ref_suffix = " | Reference" %}
    {%- else %}
    {%- set ref_suffix = "" %}
    {%- endif %}
    {%- if "vector" in param.options %}
    {{ param.python_name }}: typing.Sequence[{{ type_ }}]{{ ref_suffix }}{{ field_value(param) }}
    {%- else %}
    {{ param.python_name }}: {{ type_ }}{{ ref_suffix}}{{ field_value(param) }}
    {%- endif %}
    {%- endfor %}
    {%- if base_class == "BeamlineElement" %}
    label: str = ""
    {%- endif %}
{%- endif %}
{%- endfor %}

Autogenerated{{ base_class }} = Union[
{%- for name, element in manual.elements.items() %}
{%- if element.parameters | length %}
    {{ name | to_class_name }},
{%- endif %}
{%- endfor %}
]
