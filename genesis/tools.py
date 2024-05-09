from __future__ import annotations
import datetime
import enum
import functools
import html
import inspect
import importlib
import logging
import pathlib
import subprocess
import string
import sys
import traceback
import uuid

from numbers import Number
from typing import Any, Dict, Optional, Tuple, TypedDict, Union, cast

import h5py
import numpy as np
import prettytable
import pydantic


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


logger = logging.getLogger(__name__)


class DisplayOptions(pydantic.BaseModel):
    jupyter_render_mode: Literal["html", "markdown", "genesis"] = "html"
    console_render_mode: Literal["markdown", "genesis"] = "markdown"
    echo_genesis_output: bool = True
    include_description: bool = True
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


def store_in_hdf5_file(
    h5: h5py.Group,
    obj: pydantic.BaseModel,
    key: str = "json",
    encoding: str = "utf-8",
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
    context: _SerializationContext = {
        "hdf5": h5,
        "array_prefix": key,
        "array_index": 0,
    }
    json_data = obj.model_dump_json(context=cast(dict, context))
    h5.attrs[f"__{key}_python_class_name__"] = (
        f"{obj.__module__}.{obj.__class__.__name__}"
    )
    h5.create_dataset(
        name=key,
        dtype=h5py.string_dtype(encoding=encoding),
        data=json_data,
    )
    return json_data, context


def restore_from_hdf5_file(
    h5: h5py.Group,
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
    clsname = str(h5.attrs[f"__{key}_python_class_name__"])
    try:
        cls = import_by_name(clsname)
    except Exception:
        logger.exception("Failed to import class: %s", clsname)
        return None

    json_bytes = h5.require_dataset(
        name=key,
        dtype=h5py.string_dtype(encoding=encoding),
        shape=(),
    )
    return cls.model_validate_json(
        json_bytes[()].decode(encoding),
        context={"hdf5": h5},
    )


def _truncated_string(value, max_length: int) -> str:
    value = str(value)
    if len(value) < max_length + 3:
        return value
    value = value[max_length:]
    return f"{value}..."


def _clean_annotation(annotation) -> str:
    if inspect.isclass(annotation):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def html_table_repr(
    obj: Union[pydantic.BaseModel, Dict[str, Any]],
    seen: list,
    display_options: DisplayOptions = global_display_options,
) -> str:
    """
    Pydantic model table HTML representation for Jupyter.

    Parameters
    ----------
    obj : model instance
    seen : list of objects

    Returns
    -------
    str
        HTML table representation.
    """
    seen.append(obj)
    rows = []
    if isinstance(obj, pydantic.BaseModel):
        fields = obj.model_fields
        annotations = {
            attr: field_info.annotation for attr, field_info in fields.items()
        }
        descriptions = {
            attr: field_info.description for attr, field_info in fields.items()
        }
    else:
        fields = obj
        annotations = {attr: "" for attr in fields}
        descriptions = {attr: "" for attr in fields}

    for attr in fields:
        value = getattr(obj, attr, None)
        if value is None:
            continue
        annotation = annotations[attr]
        description = descriptions[attr]

        if isinstance(value, (pydantic.BaseModel, dict)):
            if value in seen:
                table_value = "(recursed)"
            else:
                table_value = html_table_repr(
                    value, seen, display_options=display_options
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

    ascii_table = str(ascii_table_repr(obj, list(seen))).replace("`", r"\`")
    copy_to_clipboard = string.Template(
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
        table=ascii_table,
    )
    return "\n".join(
        [
            copy_to_clipboard,
            "<table>",
            " <tr>",
            "  <th>Attribute</th>",
            "  <th>Value</th>",
            "  <th>Type</th>",
            "  <th>Description</th>" if display_options.include_description else "",
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
) -> prettytable.PrettyTable:
    """
    Pydantic model table ASCII representation for the terminal.

    Parameters
    ----------
    obj : model instance
    seen : list of objects

    Returns
    -------
    str
        HTML table representation.
    """
    seen.append(obj)
    rows = []
    if isinstance(obj, pydantic.BaseModel):
        fields = obj.model_fields
        annotations = {
            attr: field_info.annotation for attr, field_info in fields.items()
        }
        descriptions = {
            attr: field_info.description for attr, field_info in fields.items()
        }
    else:
        fields = obj
        annotations = {attr: "" for attr in fields}
        descriptions = {attr: "" for attr in fields}

    for attr in fields:
        value = getattr(obj, attr, None)
        if value is None:
            continue
        description = descriptions[attr]
        annotation = annotations[attr]

        if isinstance(value, pydantic.BaseModel):
            if value in seen:
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

    fields = ["Attribute", "Value", "Type"]
    if display_options.include_description:
        fields.append("Description")
    else:
        # Chop off the description for each row
        rows = [row[:-1] for row in rows]

    table = prettytable.PrettyTable(field_names=fields)
    table.add_rows(rows)
    table.set_style(display_options.ascii_table_type)
    return table


def read_if_path(input: Union[pathlib.Path, str]) -> Tuple[Optional[pathlib.Path], str]:
    if not input:
        return None, input

    path = pathlib.Path(input).resolve()
    try:
        is_path = isinstance(input, pathlib.Path) or path.exists()
    except OSError:
        is_path = False

    if not is_path:
        return None, str(input)

    # Update our source path; we found a file.  This is probably what
    # the user wants.
    with open(path) as fp:
        return path.absolute(), fp.read()
