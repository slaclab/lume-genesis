from __future__ import annotations

import typing
from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics.labels import mathlabel
from pmd_beamphysics.units import nice_array, nice_scale_prefix

if typing.TYPE_CHECKING:
    from .output import Genesis4Output


def add_layout_to_axes(
    output: Genesis4Output,
    *,
    ax=None,
    bounds=None,
    xfactor=1,
    add_legend=False,
):
    """
    Adds undulator layout to an axes.

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

        y, factor, prefix = nice_array(fz)

        ax1.fill_between(zlist / xfactor, y, color=color, label=label, alpha=0.5)

        ylabel = f"{label} ({prefix}{units})"
        ax1.set_ylabel(ylabel)

    labels = [line.get_label() for line in lines]
    if add_legend:
        ax.legend(lines, labels)


def plot_stats_with_layout(
    output: Genesis4Output,
    ykeys="field_energy",
    ykeys2=(),
    xkey="zplot",
    xlim=None,
    ylim=None,
    ylim2=None,
    yscale="linear",
    yscale2="linear",
    nice=True,
    tex=False,
    include_layout=True,
    include_labels=True,
    include_legend=True,
    return_figure=False,
    **kwargs,
) -> Optional[matplotlib.figure.Figure]:
    """
    Plots stat output multiple keys.

    If a list of ykeys2 is given, these will be put on the right hand axis. This can also be given as a single key.

    Logical switches:
        nice: a nice SI prefix and scaling will be used to make the numbers reasonably sized. Default: True

        tex: use mathtext (TeX) for plot labels. Default: True

        include_legend: The plot will include the legend.  Default: True

        include_layout: the layout plot will be displayed at the bottom.  Default: True

        return_figure: return the figure object for further manipulation. Default: False

    """
    if include_layout:
        fig, all_axis = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]}, **kwargs)
        ax_layout = all_axis[-1]
        ax_plot = [all_axis[0]]
    else:
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

    # No need for a legend if there is only one plot
    if len(ykeys) == 1 and not ykeys2:
        include_legend = False

    # assert xkey == 'mean_z', 'TODO: other x keys'

    X = output.stat(xkey)

    # Only get the data we need
    if xlim:
        good = np.logical_and(X >= xlim[0], X <= xlim[1])
        X = X[good]
    else:
        xlim = X.min(), X.max()
        good = slice(None, None, None)  # everything

    # X axis scaling
    units_x = str(output.units(xkey))
    if nice:
        X, factor_x, prefix_x = nice_array(X)
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
            ax.plot(X, dat / factor, label=label, color=color, linestyle=linestyle)

        # Handle tex labels
        ylabel = mathlabel(*keys, units=unit, tex=tex)
        ax.set_ylabel(ylabel)

        # Scaling(e.g. "linear", "log", "symlog", "logit")
        if ix == 0:
            ax.set_yscale(yscale)
        else:
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
        if ix == 1 and ylim2:
            pass
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
