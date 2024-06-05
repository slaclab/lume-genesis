from __future__ import annotations

import datetime
import enum
import functools
import html
import importlib
import inspect
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

import numpy as np
import prettytable
import pydantic
import pydantic_settings

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


logger = logging.getLogger(__name__)


class DisplayOptions(
    pydantic_settings.BaseSettings,
    env_prefix="LUME_",
    case_sensitive=False,
):
    """
    jupyter_render_mode : One of {"html", "markdown", "genesis", "repr"}
        Defaults to "repr".
        Environment variable: LUME_JUPYTER_RENDER_MODE.
    console_render_mode : One of {"markdown", "genesis", "repr"}
        Defaults to "repr".
        Environment variable: LUME_CONSOLE_RENDER_MODE.
    include_description : bool, default=True
        Include descriptions in table representations.
        Environment variable: LUME_INCLUDE_DESCRIPTION.
    ascii_table_type : int, default=prettytable.MARKDOWN
        Default to a PrettyTable markdown ASCII table.
        Environment variable: LUME_ASCII_TABLE_TYPE.
    filter_tab_completion : bool, default=True
        Filter out unimportant details (pydantic methods and such) from
        Genesis4 classes.
        Environment variable: LUME_FILTER_TAB_COMPLETION.
    verbose : int, default=1
        At level 0, hide Genesis4 output during `run()` by default.
        At level 1, show Genesis4 output during `run()` by default.
        Equivalent to configuring the default setting of `Genesis4.verbose` to
        `True`.
        Environment variable: LUME_VERBOSE.
    """

    jupyter_render_mode: Literal["html", "markdown", "genesis", "repr"] = "repr"
    console_render_mode: Literal["markdown", "genesis", "repr"] = "repr"
    include_description: bool = True
    ascii_table_type: int = prettytable.MARKDOWN
    verbose: int = 1
    filter_tab_completion: bool = True


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


def _truncated_string(value, max_length: int) -> str:
    """
    Truncate a string representation of ``value`` if it's longer than
    ``max_length``.

    Parameters
    ----------
    value :
    max_length : int

    Returns
    -------
    str
    """
    value = str(value)
    if len(value) < max_length + 3:
        return value
    value = value[:max_length]
    return f"{value}..."


def _clean_annotation(annotation) -> str:
    """Clean an annotation for showing to the user."""
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
    """Create copy-to-clipboard HTML for the given text."""
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
    """Get values, descriptions, and annotations for a table."""
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
    # TODO: generalize these tables; callers have a confusing mapping if they
    # change the headers
    headers = headers or ["Attribute", "Value", "Type", "Description"]
    assert len(headers) == 4

    include_description = display_options.include_description and headers[-1]

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
            description = f'<td style="text-align: left;">{description}</td>'
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

    copy_to_clipboard = _copy_to_clipboard_html(pretty_repr(obj))
    return "\n".join(
        [
            copy_to_clipboard,
            '<table style="table td:nth-child(3) { text-align: start; }">',
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
    """
    Check if the ``input`` path exists, and convert it to a `pathlib.Path`.

    Parameters
    ----------
    input : str

    Returns
    -------
    pathlib.Path or None
    """
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
    """
    Read ``input`` if it's an existing path.

    Parameters
    ----------
    input : pathlib.Path or str
        Filename *or* source contents.
    source_path : pathlib.Path or str, optional
        The path where ``input`` may be relative to.

    Returns
    -------
    Optional[pathlib.Path]
        The filename, if it was read out.
    str
        If ``input`` was an existing file, this is the string contents of that
        file.
        Otherwise, it is the source string ``input``.
    """
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


# TODO: some metadata on a per-field basis instead?
_attr_sort_disable = {
    # Don't sort elements by name; the user probably wants them in
    # the order they've listed.
    "elements",
}


def pretty_repr(
    obj,
    skip_defaults: bool = True,
    indent: int = 2,
    newline_threshold: int = 80,
    seen: Optional[list] = None,
    sort_keys: bool = True,
) -> str:
    """
    Make a prettier string representation of the given object.

    Parameters
    ----------
    obj :
        The object to get a repr of.
    skip_defaults : bool, default=True
        Do not show attributes set to their default for pydantic models.
    indent : int, default=2
        Indent this many spaces.
    newline_threshold : int, default=80
        If a repr is above this length, split it up on a new line.
    seen : list, optional
        A marker list to avoid recursion.  Not required from the top-level.
    sort_keys : bool, default=True
        Sort keys in dictionaries.

    Returns
    -------
    str
    """
    if isinstance(obj, pydantic.BaseModel):
        values = {
            attr: getattr(obj, attr, None)
            for attr, field in obj.model_fields.items()
            if field.repr
        }
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
            if attr in _attr_sort_disable:
                sort_keys_for_value = False
            else:
                sort_keys_for_value = sort_keys

            field_repr = pretty_repr(
                value,
                skip_defaults=skip_defaults,
                newline_threshold=newline_threshold - indent,
                seen=seen,
                sort_keys=sort_keys_for_value,
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
    """
    Get model attributes of type for the given model.

    Parameters
    ----------
    inst : pydantic.BaseModel
        Model instance.
    include_types : tuple of types
        Include these types.

    Yields
    ------
    str
        The dotted attribute name.
    """
    for attr in list(inst.model_fields) + list(inst.model_computed_fields):
        try:
            value = getattr(inst, attr, None)
        except Exception:
            continue

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
    """
    From a model, make a set of string alias keys, mapped to their dotted
    attribute name.

    Parameters
    ----------
    inst : pydantic.BaseModel
        Model instance.
    include_types : tuple of types, default=(np.ndarray,)
        Include these types.
    existing_aliases : Dict[str, str], optional
        An existing set of aliases to not clash with.
    attr_prefix : str, optional
        Prefix for the given instance, applied to all attribute names.
    alias_prefix : str, optional
        Prefix for the given instance, applied to all aliases names.
    ignore_parts : Tuple[str, ...], optional
        For an dotted attributed, if any part of it is in this ignore list
        then do not add it to the alias dictionary.

    Returns
    -------
    Dict[str, str]
        Dictionary of alias name to dotted attribute name.
    """
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
