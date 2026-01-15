# lume-genesis pydantic usage

## `genesis.version4.types`

Simple wrappers for:

- ParticleGroup, ParticleData
- pmd_unit
- NDArray

Along with genesis4-specifics:

- Reference
- BeamlineElement
- NameList
- FieldFileParams

## Python typing and typing_extensions

As the minimum Python pin changes, keep an eye on these for typing-extensions:

3.8: Literal
3.9: Annotated
3.11: NotRequired (TypedDict ReadOnly, NotRequired)
3.12: override

### Pydantic requirements

Per the [docs](https://docs.pydantic.dev/latest/api/standard_library_types/#typed-dictionaries):

> Because of runtime limitations, Pydantic will require using the TypedDict type
> from typing_extensions when using Python 3.12 and lower.
