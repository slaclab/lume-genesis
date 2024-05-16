from pydantic import BaseModel
import pytest

from .conftest import display_options_ctx


@pytest.mark.parametrize(
    "filter_",
    [
        pytest.param(True, id="filter-on"),
        pytest.param(False, id="filter-off"),
    ],
)
def test_render(
    element: BaseModel,
    filter_: bool,
) -> None:
    with display_options_ctx(filter_tab_completion=filter_):
        items = list(dir(element))
    if filter_:
        assert "model_validate" not in items
        assert "__init__" not in items
    else:
        assert "model_validate" in items
        assert "__init__" in items
    assert all(fld in items for fld in element.model_fields)
