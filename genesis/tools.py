from __future__ import annotations

import datetime
import enum
import functools
import html
import importlib
import inspect
import json
import logging
import pathlib
import string
import subprocess
import sys
import textwrap
import traceback
import uuid
from numbers import Number
from typing import Any, Dict, Generator, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import prettytable
import pydantic
from pmd_beamphysics.units import pmd_unit

if sys.version_info >= (3, 12):
    from typing import Literal, TypedDict
else:
    # Pydantic specifically requires this for Python < 3.12
    from typing_extensions import Literal, TypedDict


logger = logging.getLogger(__name__)


class DisplayOptions(pydantic.BaseModel):
    jupyter_render_mode: Literal["html", "markdown", "genesis", "repr"] = "repr"
    console_render_mode: Literal["markdown", "genesis", "repr"] = "repr"
    echo_genesis_output: bool = True
    include_description: bool = True
    filter_tab_completion: bool = True
    ascii_table_type: int = prettytable.MARKDOWN


global_display_options = DisplayOptions()


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"


def execute(cmd, cwd=None):
    """
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")

    Useful in Jupyter notebook

    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# Alternative execute
def execute2(cmd, timeout=None, cwd=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors.
    """
    output = {"error": True, "log": ""}
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            cwd=cwd,
        )
        output["log"] = p.stdout
        output["error"] = False
        output["why_error"] = ""
    except subprocess.TimeoutExpired as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join((stdout.decode(), f"{ex.__class__.__name__}: {ex}"))
        output["why_error"] = "timeout"
    except subprocess.CalledProcessError as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join((stdout.decode(), f"{ex.__class__.__name__}: {ex}"))
        output["why_error"] = "error"
    except Exception as ex:
        stack = traceback.print_exc()
        output["log"] = f"Unknown run error: {ex.__class__.__name__}: {ex}\n{stack}"
        output["why_error"] = "unknown"
    return output


def namelist_lines(namelist_dict, start="&name", end="/"):
    """
    Converts namelist dict to output lines, for writing to file
    """
    lines = []
    lines.append(start)
    # parse
    for key, value in namelist_dict.items():
        # if type(value) == type(1) or type(value) == type(1.): # numbers
        if isinstance(value, Number):  # numbers
            line = key + " = " + str(value)
        elif isinstance(value, list):  # lists
            liststr = ""
            for item in value:
                liststr += str(item) + " "
            line = key + " = " + liststr
        elif isinstance(value, str):  # strings
            line = (
                key + " = " + "'" + value.strip("''") + "'"
            )  # input may need apostrophes
        else:
            # Skip
            # print('skipped: key, value = ', key, value)
            continue
        lines.append(line)

    lines.append(end)
    return lines


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)


def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, "tolist", lambda: value)()


def isotime():
    """UTC to ISO 8601 with Local TimeZone information without microsecond"""
    return (
        datetime.datetime.now(datetime.UTC)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


@functools.cache
def get_output_mode() -> OutputMode:
    """
    Get the output mode for lume-genesis objects.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    OutputMode
        The detected output mode.
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return OutputMode.plain

    from IPython.display import display

    class ReprCheck:
        mode: OutputMode = OutputMode.unknown

        def _repr_html_(self) -> str:
            self.mode = OutputMode.html
            return "<!-- lume-genesis detected Jupyter and will use HTML for rendering. -->"

        def __repr__(self) -> str:
            self.mode = OutputMode.plain
            return ""

    check = ReprCheck()
    display(check)
    return check.mode


def is_jupyter() -> bool:
    """Is Jupyter detected?"""
    return get_output_mode() == OutputMode.html


def import_by_name(clsname: str) -> type:
    """
    Import the given class or function by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pcdsdevices.device_types.IPM"``

    Returns
    -------
    type
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")


class _SerializationContext(TypedDict):
    hdf5: h5py.Group
    array_prefix: str
    array_index: int


class _DeserializationContext(TypedDict):
    hdf5: h5py.Group
    workdir: pathlib.Path


_reserved_h5_attrs = {
    "__python_key_map__",
    "__num_items__",
}


def _hdf5_dictify(data, encoding: str):
    if isinstance(data, dict):
        return {
            key: _hdf5_dictify(value, encoding=encoding) for key, value in data.items()
        }
    if data is None:
        return {
            "__python_class_name__": type(data).__name__,
        }
    if isinstance(data, pathlib.Path):
        return {
            "__python_class_name__": "pathlib.Path",
            "value": str(data),
        }
    if isinstance(data, str):
        return data.encode(encoding)
    if isinstance(data, (int, float, bool)):
        return data
    if isinstance(data, bytes):
        # Bytes are less common than strings in our library; assume
        # byte datasets are encoded strings unless marked like so:
        return {
            "__python_class_name__": "bytes",
            "value": data,
        }
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        try:
            as_array = np.asarray(data)
        except ValueError:
            pass
        else:
            if as_array.dtype.kind not in "OU":
                return {
                    "__python_class_name__": type(data).__name__,
                    "value": as_array,
                }
        items = {
            f"index_{idx}": _hdf5_dictify(value, encoding=encoding)
            for idx, value in enumerate(data)
        }
        return {
            "__python_class_name__": type(data).__name__,
            "__num_items__": len(items),
            **items,
        }
    if isinstance(data, pmd_unit):
        from .version4.types import PydanticPmdUnit

        adapter = pydantic.TypeAdapter(PydanticPmdUnit)
        return _hdf5_dictify(adapter.dump_python(data, mode="json"), encoding=encoding)

    raise NotImplementedError(type(data))


def _hdf5_make_key_map(data):
    key_to_h5_key = {}
    h5_key_to_key = {}
    key_format = "{prefix}__key{idx}__"
    for idx, key in enumerate(data):
        if isinstance(key, int):
            key_to_h5_key[key] = str(key)
            h5_key_to_key[str(key)] = key
            continue
        elif not key.isascii() or "/" in key:
            prefix = "".join(ch for ch in key if ch.isascii() and ch not in "/")
        else:
            # No mapping required
            continue

        newkey = key_format.format(prefix=prefix, idx=idx)
        while newkey in h5_key_to_key or newkey in data:
            newkey += "_"
        h5_key_to_key[newkey] = key
        key_to_h5_key[key] = newkey
    return key_to_h5_key, h5_key_to_key


def _hdf5_store_dict(group: h5py.Group, data, encoding: str, depth=0) -> None:
    for attr in _reserved_h5_attrs:
        value = data.pop(attr, None)
        if value:
            group.attrs[attr] = value

    key_to_h5_key, h5_key_to_key = _hdf5_make_key_map(data)
    if key_to_h5_key:
        group.attrs["__python_key_map__"] = json.dumps(h5_key_to_key).encode(encoding)

    for key, value in data.items():
        h5_key = key_to_h5_key.get(key, key)
        if h5_key != key:
            logger.debug(f"{depth * ' '}|- {key} (renamed: {h5_key})")
        else:
            logger.debug(f"{depth * ' '}|- {key}")

        if isinstance(value, dict):
            _hdf5_store_dict(
                group.create_group(h5_key),
                value,
                depth=depth + 1,
                encoding=encoding,
            )
        elif isinstance(value, (str, bytes, int, float, bool)):
            group.attrs[h5_key] = value
        elif isinstance(value, (np.ndarray,)):
            group.create_dataset(h5_key, data=value)
        else:
            raise NotImplementedError(type(value))


def store_in_hdf5_file(
    h5: h5py.Group,
    obj: pydantic.BaseModel,
    key: str = "json",
    encoding: str = "utf-8",
    create_group: bool = False,
) -> Tuple[str, _SerializationContext]:
    """
    Store a generic Pydantic model instance in an HDF5 file.

    Numpy arrays are handled specially, where each array in the object
    corresponds to an h5py dataset in the group.  The remainder of the data is
    stored as Pydantic-serialized JSON.

    This has limitations but is intended to support Genesis 4 input and output
    types.

    Parameters
    ----------
    h5 : h5py.Group
        The file or group to store ``dct`` in.
    obj : pydantic.BaseModel
        The object to store.
    key : str, default="json"
        The key where to store the JSON data.  Arrays will be stored with this
        as a prefix.
    encoding : str, default="utf-8"
        String encoding for the data.
    """

    data = obj.model_dump()

    if create_group:
        group = h5.create_group(key)
    else:
        group = h5

    group.attrs["__python_class_name__"] = f"{obj.__module__}.{obj.__class__.__name__}"
    _hdf5_store_dict(group, _hdf5_dictify(data, encoding=encoding), encoding=encoding)


def _hdf5_restore_dict(
    item: Union[h5py.Group, h5py.Dataset, h5py.Datatype], encoding: str, depth=0
):
    if isinstance(item, h5py.Datatype):
        raise NotImplementedError(str(item))

    if isinstance(item, h5py.Dataset):
        logger.debug(f"{depth * ' '} -> dataset {item.dtype}")
        if item.dtype.kind in "SO":
            return item.asstr()[()]
        return np.asarray(item)

    python_class_name = item.attrs.get("__python_class_name__", None)
    if python_class_name:
        logger.debug(f"{depth * ' '}|- Restoring class {python_class_name}")
        if python_class_name == "pathlib.Path":
            return pathlib.Path(item.attrs["value"])
        if python_class_name == "NoneType":
            return None
        if python_class_name == "bytes":
            return item.attrs["value"]
        if python_class_name in ("list", "tuple"):
            if "__num_items__" in item.attrs:
                num_items = item.attrs["__num_items__"]
                items = [
                    _hdf5_restore_dict(
                        item[f"index_{idx}"],
                        encoding=encoding,
                        depth=depth + 1,
                    )
                    for idx in range(num_items)
                ]
                return tuple(items) if python_class_name == "tuple" else items
            elif "value" in item:
                items = np.asarray(item["value"]).tolist()
                return tuple(items) if python_class_name == "tuple" else items
            else:
                raise NotImplementedError("Unsupported list format")
        cls = import_by_name(python_class_name)
        if not issubclass(cls, pydantic.BaseModel):
            raise NotImplementedError(python_class_name)

    logger.debug(f"{depth * ' '}| Restoring dictionary group with keys:")
    res = {}

    key_map_json = item.attrs.get("__python_key_map__", None)
    key_map = json.loads(key_map_json) if key_map_json else {}

    for h5_key, group in item.items():
        key = key_map.get(h5_key, h5_key)
        if h5_key != key:
            logger.debug(f"{depth * ' '}|- {key} (h5 key was: {h5_key})")
        else:
            logger.debug(f"{depth * ' '}|- {key}")

        res[key] = _hdf5_restore_dict(group, encoding=encoding, depth=depth + 1)

    for h5_key, value in item.attrs.items():
        if h5_key in _reserved_h5_attrs:
            continue

        key = key_map.get(h5_key, h5_key)
        if h5_key != key:
            logger.debug(f"{depth * ' '}|- {key} (h5 key was: {h5_key})")
        else:
            logger.debug(f"{depth * ' '}|- {key}")

        if hasattr(value, "tolist"):
            # Get the native type from a numpy scalar
            value = value.tolist()
        res[key] = value

    return res


def restore_from_hdf5_file(
    h5: h5py.Group,
    workdir: Optional[pathlib.Path] = None,
    key: str = "json",
    encoding: str = "utf-8",
) -> Optional[pydantic.BaseModel]:
    """
    Restore a Pydantic model instance from an HDF5 file stored using
    `store_in_hdf5_file`.

    Parameters
    ----------
    h5 : h5py.Group
        The file or group to restore from.
    key : str, default="json"
        The key where to find the JSON data.  Arrays will be restored using
        this as a prefix.
    encoding : str, default="utf-8"
        String encoding for the data.
    """
    final_classname = str(h5.attrs["__python_class_name__"])
    try:
        cls = import_by_name(final_classname)
    except Exception:
        logger.exception("Failed to import class: %s", final_classname)
        return None

    assert issubclass(cls, pydantic.BaseModel)

    if workdir is None:
        if h5.file.filename:
            workdir = pathlib.Path(h5.file.filename).resolve().parent
        else:
            workdir = pathlib.Path(".")

    logger.debug(f"Dictifying data to restore from h5 group: {h5}")
    data = _hdf5_restore_dict(h5, encoding=encoding)
    return cls.model_validate(data)


def _truncated_string(value, max_length: int) -> str:
    value = str(value)
    if len(value) < max_length + 3:
        return value
    value = value[:max_length]
    return f"{value}..."


def _clean_annotation(annotation) -> str:
    if inspect.isclass(annotation):
        return annotation.__name__
    annotation = str(annotation)
    for remove in [
        "typing.",
        "typing_extensions.",
        "genesis.version4.input._lattice.",
        "genesis.version4.input._main.",
        "genesis.version4.input.",
        "genesis.version4.output.",
    ]:
        annotation = annotation.replace(remove, "")

    if annotation.startswith("Literal['"):
        # This is a bit of an implementation detail we don't necessarily need
        # to expose to the user; Literal['type'] is used to differentiate
        # beamline elements and namelists during deserialization.
        return "str"
    return annotation


def table_output(
    obj: Union[pydantic.BaseModel, Dict[str, Any]],
    display_options: DisplayOptions = global_display_options,
    descriptions: Optional[Mapping[str, Optional[str]]] = None,
    annotations: Optional[Mapping[str, Optional[str]]] = None,
    headers: Optional[Sequence[str]] = None,
):
    """
    Create a table based on user settings for the given object.

    In Jupyter (with "html" render mode configured), this will display
    an HTML table.

    In the terminal, this will create a markdown ASCII table.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.
    """
    if is_jupyter() and display_options.jupyter_render_mode != "markdown":

        class _InfoObj:
            def _repr_html_(_self) -> str:
                return html_table_repr(
                    obj,
                    seen=[],
                    descriptions=descriptions,
                    annotations=annotations,
                    display_options=display_options,
                    headers=headers,
                )

        return _InfoObj()

    ascii_table = ascii_table_repr(
        obj,
        seen=[],
        display_options=display_options,
        descriptions=descriptions,
        annotations=annotations,
        headers=headers,
    )
    print(ascii_table)


def _copy_to_clipboard_html(contents: str) -> str:
    return string.Template(
        """
        <div style="display: flex; justify-content: flex-end;">
          <button class="copy-${hash_}">
            Copy to clipboard
          </button>
          <br />
        </div>
        <script type="text/javascript">
          function copy_to_clipboard(text) {
            navigator.clipboard.writeText(text).then(
              function () {
                console.log("Copied to clipboard:", text);
              },
              function (err) {
                console.error("Failed to copy to clipboard:", err, text);
              },
            );
          }
          var copy_button = document.querySelector(".copy-${hash_}");
          copy_button.addEventListener("click", function (event) {
            copy_to_clipboard(`${table}`);
          });
        </script>
        """
    ).substitute(
        hash_=uuid.uuid4().hex,
        table=contents.replace("`", r"\`"),
    )


def _get_table_fields(
    obj: Union[pydantic.BaseModel, Dict[str, Any]],
    descriptions: Optional[Mapping[str, Optional[str]]] = None,
    annotations: Optional[Mapping[str, Optional[str]]] = None,
):
    if isinstance(obj, pydantic.BaseModel):
        fields = {
            attr: getattr(obj, attr, None)
            for attr, field_info in obj.model_fields.items()
            if field_info.repr
        }
        if annotations is None:
            annotations = {
                attr: field_info.annotation
                for attr, field_info in obj.model_fields.items()
            }
        if descriptions is None:
            descriptions = {
                attr: field_info.description
                for attr, field_info in obj.model_fields.items()
            }
    else:
        fields = obj
        if annotations is None:
            annotations = {attr: "" for attr in fields}
        if descriptions is None:
            descriptions = {attr: "" for attr in fields}

    return fields, descriptions, annotations


def html_table_repr(
    obj: Union[pydantic.BaseModel, Dict[str, Any]],
    seen: list,
    display_options: DisplayOptions = global_display_options,
    descriptions: Optional[Mapping[str, Optional[str]]] = None,
    annotations: Optional[Mapping[str, Optional[str]]] = None,
    headers: Optional[Sequence[str]] = None,
) -> str:
    """
    Pydantic model table HTML representation for Jupyter.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.

    Returns
    -------
    str
        HTML table representation.
    """
    headers = headers or ["Attribute", "Value", "Type", "Description"]
    assert len(headers) == 4

    include_description = display_options.include_description and headers[-1]

    # For the "copy to clipboard" functionality below:
    ascii_table = str(
        ascii_table_repr(
            obj,
            descriptions=descriptions,
            annotations=annotations,
            seen=list(seen),
            headers=headers,
        )
    )

    seen.append(id(obj))
    rows = []
    fields, descriptions, annotations = _get_table_fields(
        obj, descriptions, annotations
    )

    for attr, value in fields.items():
        if value is None:
            continue
        annotation = annotations.get(attr, "")
        description = descriptions.get(attr, "")

        if isinstance(value, (pydantic.BaseModel, dict)):
            if id(value) in seen:
                table_value = "(recursed)"
            else:
                table_value = html_table_repr(
                    value,
                    seen,
                    display_options=display_options,
                    headers=headers,
                )
        else:
            table_value = html.escape(_truncated_string(value, max_length=100))

        annotation = html.escape(_clean_annotation(annotation))
        if display_options.include_description:
            description = html.escape(description or "")
            description = f"<td>{description}</td>"
        else:
            description = ""

        rows.append(
            f"<tr>"
            f"<td>{attr}</td>"
            f"<td>{table_value}</td>"
            f"<td>{annotation}</td>"
            f"{description}"
            f"</tr>"
        )

    copy_to_clipboard = _copy_to_clipboard_html(ascii_table)
    return "\n".join(
        [
            copy_to_clipboard,
            "<table>",
            " <tr>",
            f"  <th>{headers[0]}</th>",
            f"  <th>{headers[1]}</th>",
            f"  <th>{headers[2]}</th>",
            f"  <th>{headers[3]}</th>" if include_description else "",
            " </tr>",
            "</th>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
        ]
    )


def ascii_table_repr(
    obj: Union[pydantic.BaseModel, Dict[str, Any]],
    seen: list,
    display_options: DisplayOptions = global_display_options,
    descriptions: Optional[Mapping[str, Optional[str]]] = None,
    annotations: Optional[Mapping[str, Optional[str]]] = None,
    headers: Optional[Sequence[str]] = None,
) -> prettytable.PrettyTable:
    """
    Pydantic model table ASCII representation for the terminal.

    Parameters
    ----------
    obj : model instance or dict
    seen : list of objects
        Used to ensure that objects are only shown once.
    display_options: DisplayOptions, optional
        Defaults to `global_display_options`.
    descriptions : dict of str to str, optional
        Optional override of descriptions found on the object.
    annotations : dict of str to str, optional
        Optional override of annotations found on the object.

    Returns
    -------
    str
        HTML table representation.
    """
    headers = headers or ["Attribute", "Value", "Type", "Description"]
    assert len(headers) == 4

    seen.append(id(obj))
    rows = []
    fields, descriptions, annotations = _get_table_fields(
        obj, descriptions, annotations
    )

    for attr, value in fields.items():
        if value is None:
            continue
        description = descriptions.get(attr, "")
        annotation = annotations.get(attr, "")

        if isinstance(value, pydantic.BaseModel):
            if id(value) in seen:
                table_value = "(recursed)"
            else:
                table_value = str(
                    ascii_table_repr(value, seen, display_options=display_options)
                )
        else:
            table_value = _truncated_string(value, max_length=30)

        rows.append(
            (
                attr,
                table_value,
                _clean_annotation(annotation),
                description or "",
            )
        )

    headers = list(headers)
    if not display_options.include_description or not headers[-1]:
        headers = headers[:3]
        # Chop off the description for each row
        rows = [row[:-1] for row in rows]

    table = prettytable.PrettyTable(field_names=headers)
    table.add_rows(rows)
    table.set_style(display_options.ascii_table_type)
    return table


def check_if_existing_path(input: str) -> Optional[pathlib.Path]:
    path = pathlib.Path(input).resolve()
    try:
        if path.exists():
            return path
    except OSError:
        ...
    return None


def read_if_path(
    input: Union[pathlib.Path, str],
    source_path: Optional[Union[pathlib.Path, str]] = None,
) -> Tuple[Optional[pathlib.Path], str]:
    if not input:
        return None, input

    if source_path is None:
        source_path = pathlib.Path(".")

    source_path = pathlib.Path(source_path)

    if isinstance(input, pathlib.Path):
        path = input
    else:
        path = check_if_existing_path(str(source_path / input))
        if not path:
            return None, str(input)

    # Update our source path; we found a file.  This is probably what
    # the user wants.
    with open(path) as fp:
        return path.absolute(), fp.read()


def pretty_repr(
    obj,
    skip_defaults: bool = True,
    indent: int = 2,
    newline_threshold: int = 80,
    seen: Optional[list] = None,
    sort_keys: bool = True,
) -> str:
    if isinstance(obj, pydantic.BaseModel):
        values = {attr: getattr(obj, attr, None) for attr in obj.model_fields}
        defaults = {attr: field.default for attr, field in obj.model_fields.items()}
        attr_prefix = "{attr}="
        if hasattr(obj, "_pretty_repr_"):
            # Always make the pretty repr
            basic_repr = " " * newline_threshold
        else:
            basic_repr = repr(obj)
    elif isinstance(obj, (list, tuple)):
        values = {idx: val for idx, val in enumerate(obj)}
        defaults = {idx: None for idx in range(len(obj))}
        attr_prefix = ""
        basic_repr = repr(obj)
    elif isinstance(obj, dict):
        if sort_keys:
            values = {key: obj[key] for key in sorted(obj)}
        else:
            values = dict(obj)
        defaults = {key: None for key in obj}
        attr_prefix = "'{attr}': "
        basic_repr = repr(obj)
    elif isinstance(obj, pathlib.Path):
        path = str(obj).replace("'", r"\'")
        return f"pathlib.Path('{path}')"
    else:
        return repr(obj)

    if len(basic_repr) < newline_threshold:
        return basic_repr

    if seen is None:
        seen = []

    if id(obj) in seen:
        return "(recursed)"

    seen.append(id(obj))

    lines = []
    for attr, value in values.items():
        if skip_defaults:
            default = defaults[attr]
            try:
                if isinstance(value, np.ndarray):
                    is_default = False  # np.allclose(value, default or [])
                else:
                    is_default = default == value
            except Exception:
                is_default = False

            if is_default:
                continue

        if isinstance(value, (pydantic.BaseModel, dict, list, tuple, pathlib.Path)):
            field_repr = pretty_repr(
                value,
                skip_defaults=skip_defaults,
                newline_threshold=newline_threshold - indent,
                seen=seen,
            )
        elif isinstance(value, np.ndarray):
            field_repr = repr(value)
            short_repr = f"array(shape={value.shape}, dtype={value.dtype})"
            if len(field_repr) > len(short_repr):
                field_repr = short_repr
        else:
            field_repr = repr(value)

        if field_repr.count("\n") > 0:
            field_repr = textwrap.indent(field_repr, prefix=" " * indent).lstrip()

        line_prefix = attr_prefix.format(indent=indent, attr=attr)
        lines.extend(f"{indent * ' '}{line_prefix}{field_repr},".splitlines())

    seen.remove(id(obj))

    inside = " ".join(line[indent:] for line in lines)
    if len(inside) < newline_threshold:
        lines = [inside]

    if isinstance(obj, dict):
        prefix = ""
        open_bracket, close_bracket = "{}"
    elif isinstance(obj, list):
        prefix = ""
        open_bracket, close_bracket = "[]"
    elif isinstance(obj, tuple):
        prefix = ""
        open_bracket, close_bracket = "()"
    else:
        prefix = obj.__class__.__name__
        open_bracket, close_bracket = "()"

    if len(lines) == 0:
        return f"{prefix}{open_bracket}{close_bracket}"
    if len(lines) == 1:
        line = lines[0].lstrip().rstrip(",")
        return f"{prefix}{open_bracket}{line}{close_bracket}"
    else:
        lines.insert(0, f"{prefix}{open_bracket}")
        lines.append(f"{close_bracket}")

    return "\n".join(_truncated_string(line, max_length=150) for line in lines)


def get_attrs_of_type(
    inst: pydantic.BaseModel, include_types: Tuple[type, ...]
) -> Generator[str, None, None]:
    for attr in list(inst.model_fields) + list(inst.model_computed_fields):
        value = getattr(inst, attr, None)
        if isinstance(value, include_types):
            yield attr
        elif isinstance(value, pydantic.BaseModel):
            for sub_attr in get_attrs_of_type(value, include_types=include_types):
                yield f"{attr}.{sub_attr}"


def make_dotted_aliases(
    inst: pydantic.BaseModel,
    include_types: Tuple[type, ...] = (np.ndarray,),
    existing_aliases: Optional[Dict[str, str]] = None,
    attr_prefix: str = "",
    alias_prefix: str = "",
    ignore_parts: Tuple[str, ...] = ("stat",),
) -> Dict[str, str]:
    attrs = list(get_attrs_of_type(inst, include_types=include_types))
    aliases = {
        alias_prefix + attr.replace(".", "_"): attr_prefix + attr for attr in attrs
    }

    by_last_part = {}
    existing_alias_attrs = (existing_aliases or {}).values()
    for attr in list(attrs) + list(existing_alias_attrs):
        parts = attr.rsplit(".")
        if any(p in parts for p in ignore_parts):
            continue
        by_last_part.setdefault(parts[-1], []).append(attr)

    for last_part, attrs in by_last_part.items():
        if len(attrs) > 1:
            continue
        (attr,) = attrs
        full_attr = attr_prefix + attr
        if full_attr != last_part:
            aliases[last_part] = full_attr

    def clean(alias):
        return alias.replace("__", "_")

    return {clean(alias): attr for alias, attr in aliases.items()}
