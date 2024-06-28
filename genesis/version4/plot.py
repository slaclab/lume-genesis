from __future__ import annotations

import typing
from typing import Any, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics.labels import mathlabel
from pmd_beamphysics.units import nice_array, nice_scale_prefix

if typing.TYPE_CHECKING:
    from .output import Genesis4Output


PlotMaybeLimits = Tuple[Optional[float], Optional[float]]
PlotLimits = Tuple[float, float]


def add_layout_to_axes(
    output: Genesis4Output,
    ax: matplotlib.axes.Axes,
    *,
    bounds: Optional[PlotLimits] = None,
    xfactor: float = 1,
    add_legend: bool = False,
) -> None:
    """
    Adds undulator layout to an axes.

    Parameters
    ----------
    output : Genesis4Output
        The output instance to get data from.
    ax : matplotlib.axes.Axes
    bounds : PlotLimits or None, default=None
        Z limits (defaults to 0 to zmax)
    xfactor : float, default=1
        Divisor for Z position.
    add_legend : bool, default=False
        Add a legend to the plot.
    """

    if bounds is None:
        zmin, zmax = 0, output.stat("z").max()
    else:
        zmin, zmax = bounds
    ax.set_xlim(zmin, zmax)

    ax2 = ax.twinx()

    ax.set_xlabel(r"$z$ (m)")

    zlist = output.stat("z")

    lines = []
    for ax1, component, color, label, units in (
        (ax, "aw", "red", r"$aw$", "1"),
        (ax2, "qf", "blue", r"Quad $k$", r"$1/m^2$"),
    ):
        fz = output.stat(component)

        y, _factor, prefix = nice_array(fz)

        ax1.fill_between(
            zlist / xfactor,
            y,
            color=color,
            label=label,
            alpha=0.5,
            step="mid",
        )

        ylabel = f"{label} ({prefix}{units})"
        ax1.set_ylabel(ylabel)

    labels = [line.get_label() for line in lines]
    if add_legend:
        ax.legend(lines, labels)


def plot_stats_with_layout(
    output: Genesis4Output,
    ykeys: Union[str, Sequence[str]] = "field_energy",
    ykeys2: Union[str, Sequence[str]] = (),
    xkey: str = "zplot",
    xlim: Optional[PlotLimits] = None,
    ylim: Optional[PlotMaybeLimits] = None,
    ylim2: Optional[PlotMaybeLimits] = None,
    yscale: str = "linear",
    yscale2: str = "linear",
    nice: bool = True,
    tex: bool = False,
    include_layout: bool = True,
    include_legend: bool = True,
    return_figure: bool = False,
    **kwargs: Any,
) -> Optional[matplotlib.figure.Figure]:
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis.
    This can also be given as a single key.

    Parameters
    ----------
    output : Genesis4Output
        The output instance to get data from.
    ykeys : str or list of str, default="field_energy"
        Y keys to plot.
    ykeys2 : str or list of str, default=()
        Y keys to plot on the right-hand axis.
    xkey : str, optional
        The X axis data key.
    xlim : list
        Limits for the X axis
    ylim : list
        Limits for the Y axis
    ylim2 : list
        Limits for the secondary Y axis
    yscale: str
        one of "linear", "log", "symlog", "logit", ... for the Y axis
    yscale2: str
        one of "linear", "log", "symlog", "logit", ... for the secondary Y axis
    y2 : list
        List of keys to be displayed on the secondary Y axis
    nice : bool
        Whether or not a nice SI prefix and scaling will be used to
        make the numbers reasonably sized. Default: True
    include_layout : bool
        Whether or not to include a layout plot at the bottom. Default: True
        Whether or not the plot should include the legend. Default: True
    return_figure : bool
        Whether or not to return the figure object for further manipulation.
        Default: True
    kwargs : dict
        Extra arguments can be passed to the specific plotting function.

    Returns
    -------
    fig : matplotlib.pyplot.figure.Figure
        The plot figure for further customizations or `None` if
        `return_figure` is set to False.
    """
    if include_layout:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
        ax_layout = None
        fig, all_axis = plt.subplots(**kwargs)
        ax_plot = [all_axis]

    # collect axes
    if isinstance(ykeys, str):
        ykeys = [ykeys]

    if ykeys2:
        if isinstance(ykeys2, str):
            ykeys2 = [ykeys2]
        ax_twinx = ax_plot[0].twinx()
        ax_plot.append(ax_twinx)
    else:
        ax_twinx = None

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    x_array = output.stat(xkey)

    # Only get the data we need
    if xlim:
        good = np.logical_and(x_array >= xlim[0], x_array <= xlim[1])
        x_array = x_array[good]
    else:
        xlim = x_array.min(), x_array.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = str(output.units(xkey))
    if nice:
        x_array, factor_x, prefix_x = nice_array(x_array)
        units_x = prefix_x + units_x
    else:
        factor_x = 1

    # set all but the layout

    # Handle tex labels
    xlabel = mathlabel(xkey, units=units_x, tex=tex)

    for ax in ax_plot:
        ax.set_xlim(xlim[0] / factor_x, xlim[1] / factor_x)
        ax.set_xlabel(xlabel)

    # Draw for Y1 and Y2

    linestyles = ["solid", "dashed"]

    ii = -1  # counter for colors
    for ix, keys in enumerate([ykeys, ykeys2]):
        if not keys:
            continue
        ax = ax_plot[ix]
        linestyle = linestyles[ix]

        # Check that units are compatible
        ulist = [output.units(key) for key in keys]
        if len(ulist) > 1:
            for u2 in ulist[1:]:
                assert ulist[0] == u2, f"Incompatible units: {ulist[0]} and {u2}"
        # String representation
        unit = str(ulist[0])

        # Data
        data = [output.stat(key)[good] for key in keys]

        if nice:
            factor, prefix = nice_scale_prefix(np.ptp(data))
            unit = prefix + unit
        else:
            factor = 1

        # Make a line and point
        for key, dat in zip(keys, data):
            #
            ii += 1
            color = "C" + str(ii)

            # Handle tex labels
            label = mathlabel(key, units=unit, tex=tex)
            ax.plot(
                x_array, dat / factor, label=label, color=color, linestyle=linestyle
            )

        # Handle tex labels
        ylabel = mathlabel(*keys, units=unit, tex=tex)
        ax.set_ylabel(ylabel)

        # Scaling(e.g. "linear", "log", "symlog", "logit")
        if ix == 0:
            ax.set_yscale(yscale)
        elif ax_twinx is not None:
            ax_twinx.set_yscale(yscale2)

        # Set limits, considering the scaling.
        if ix == 0 and ylim:
            ymin = ylim[0]
            ymax = ylim[1]
            # Handle None and scaling
            if ymin is not None:
                ymin = ymin / factor
            if ymax is not None:
                ymax = ymax / factor
            new_ylim = (ymin, ymax)
            ax.set_ylim(new_ylim)
        # Set limits, considering the scaling.
        if ix == 1 and ylim2 and ax_twinx is not None:
            # TODO
            if ylim2:
                ymin2 = ylim2[0]
                ymax2 = ylim2[1]
                # Handle None and scaling
                if ymin2 is not None:
                    ymin2 = ymin2 / factor
                if ymax2 is not None:
                    ymax2 = ymax2 / factor
                new_ylim2 = (ymin2, ymax2)
                ax_twinx.set_ylim(new_ylim2)
            else:
                pass

    # Collect legend
    if include_legend:
        lines = []
        labels = []
        for ax in ax_plot:
            a, b = ax.get_legend_handles_labels()
            lines += a
            labels += b
        ax_plot[0].legend(lines, labels, loc="best")

    # Layout
    if include_layout:
        assert ax_layout is not None
        # Gives some space to the top plot
        ax_layout.set_ylim(-1, 1.5)

        # if xkey == 'mean_z':
        #     ax_layout.set_xlim(xlim[0], xlim[1])
        # else:
        #     ax_layout.set_xlabel('mean_z')
        #     xlim = (0, I.stop)
        add_layout_to_axes(output, ax=ax_layout, bounds=xlim)

    if return_figure:
        return fig
