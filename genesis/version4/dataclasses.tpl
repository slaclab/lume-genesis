{% for name, element in manual.elements.items() %}
{% if element.parameters | length %}
@dataclasses.dataclass
class {{ name | capitalize }}:
    r"""
    {{ docstrings[name] | wordwrap | indent(4) }}

    Attributes
    ----------
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {{ param.name }} : {{ type_ }}{% if not param.default is none %}, default={{ param.default }}{% endif %}
        {{ param.description | wordwrap | indent(8) }}
    {%- endfor %}
    """
    {%- for param in element.parameters.values() %}
    {%- set type_ = type_map.get(param.type, param.type) %}
    {{ param.name }}: {{ type_ }} = {{ param.default }}
    {%- endfor %}
{% endif %}
{% endfor %}
