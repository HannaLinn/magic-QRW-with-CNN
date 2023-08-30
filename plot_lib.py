import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.legend as mlegend
from scipy.fftpack import rfft
import scipy.integrate as scint
import scipy.sparse.linalg as spla
import seaborn as sns
from string import ascii_lowercase
import itertools
import scipy.optimize as sc_opt
from math import factorial
import scipy as sp
import datetime
import time
import matplotlib.font_manager
from inspect import getargspec
from functools import wraps


def edge_change(cs, cbar=None):
    '''
    Helper function to settle the bug that a contour plot saved as pdf shows
    the figure background as no contour edges are drawn.
    This functions draws the edge of each contour with the respective color.

    Parameters
    -----------

    cs: matplotlib countour QuadContourSet,
        this is returned from matplotlibs 'contourf'

    cbar: matplotlib colorbar
        Returned from mpl.colorbar

    Returns
    -----------

    '''
    for c in cs.collections:
        c.set_edgecolor("face")

    if cbar:
        cbar.solids.set_edgecolor("face")
    return


def set_ylabel_side(ax=None, pos='right'):
    """
    Wrapper for setting y-xis label and ticks to the right (left) side of the
    figure.

    Parameters
    ----------
    ax: plt.axes instnace, optional
        axes instance which should be changed. The default is 'None'
    pos: str, optional
        y-axis tick position, either 'left' or 'right'. The default is 'right'.

    Returns
    -------
    None.

    """
    if ax is None:
        ax = plt.gca()

    ax.yaxis.set_label_position(pos)

    if pos == 'right':
        ax.yaxis.tick_right()
    else:
        ax.yaxis.tick_left()

    ax.yaxis.set_ticks_position('both')
    return


def create_fig(scale=1, single_col=False, width=None, height=None, **kwargs):
    """
    Creates figure based on style choosen in :func:`update_settings()`

    Parameters
    ----------
    **kwargs:
        Additional keyword arguments that passed to 'plt.subplots()'.

    Returns
    -------
    fig : matplotlib figure instance,
        The figure that was created.
    ax : matplotlib axes instance,

    """
    cols = kwargs.get('ncols', 1)
    rows = kwargs.get('nrows', 1)

    w, h = plt.rcParams['figure.figsize']

    if single_col:
        w = width if width is not None else w
        h = (height if height is not None else h) * rows

    else:
        w = (width if width is not None else w) * min(cols, 2)
        h = (height if height is not None else h) * rows

    fig, ax = plt.subplots(figsize=(w*scale, h*scale), **kwargs)

    fig.subplots_adjust(top=0.97, bottom=0.17, left=0.15, right=0.97,)

    return fig, ax


def add_label(ax, text='(a)', x0=None, y0=None, **kwargs):
    '''
    Adds labels to multi-axis figures.
    '''

    if isinstance(ax, np.ndarray):
        abc = [f'({letter})' for letter in list(ascii_lowercase)]

        f = ax.flat[0].get_figure()
        if y0 is None:
            y0 = f.subplotpars.top
        if x0 is None:
            x0 = f.subplotpars.left
        for i, _ax in enumerate(ax.flat):
            _ax.annotate(r'\textbf{{{}}} '.format(abc[i]),
                         xy=(-x0, y0), annotation_clip=False,
                         xycoords='axes fraction', weight='bold',
                         fontsize=plt.rcParams['font.size'],
                         va='bottom', **kwargs
                         )
    else:
        fig = ax.get_figure()
        if x0 is None:
            x0 = fig.subplotpars.left
        if y0 is None:
            y0 = fig.subplotpars.top
        if plt.rcParams['text.usetex'] is True:
            ax.annotate(r'\textbf {{{}}}'.format(text),
                        xy=(-x0, y0), annotation_clip=False,
                        xycoords='axes fraction', weight='bold',
                        fontsize=plt.rcParams['font.size'],
                        va='top', **kwargs,
                        )
        else:
            ax.annotate(r'{}'.format(text),
                        xy=(-x0, y0), annotation_clip=False,
                        xycoords='axes fraction', weight='bold',
                        fontsize=plt.rcParams['font.size'],
                        )

    return


def tablelegend(ax, col_labels=None, row_labels=None, title_label="",
                sort_idx=None, xhandles=None, *args,
                **kwargs):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    +---------------+---------------+---------------+---------------+
    | title_label   | col_labels[0] | col_labels[1] | col_labels[2] |
    +===============+===============+===============+===============+
    | row_labels[0] |                                               |
    +---------------+---------------+---------------+---------------+
    | row_labels[1] |             + <artists go there>              |
    +---------------+---------------+---------------+---------------+
    | row_labels[2] |                                               |
    +---------------+---------------+---------------+---------------+

    Parameters
    ----------

    ax: `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels: list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels: list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label: str, optional
        Label for the top left corner in the legend table.

    ncol: int
        Number of columns.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    Notes
    -----

    Adapted from https://stackoverflow.com/a/60345118/7119086

    """

    # obtain handles and lables from ax.legend
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax],
                                                                     *args,
                                                                     **kwargs)
    if sort_idx:
        handles = [handles[i] for i in sort_idx]

    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

    # modifications for table legend
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad',
                                   0 if col_labels is None else -1.7)
        title_label = [title_label]

        # blank rectangle handle, used to add column and row names.
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False,
                           edgecolor='none', linewidth=0)]

        # empty label
        empty = [r'']

        # number of rows infered from number of handles and desired number of
        # columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            if nrow != len(row_labels):
                raise ValueError(
                    "nrow = len(handles) // ncol = {0} but must equal to"
                    " len(row_labels) = {1}.".format(nrow, len(row_labels)))
            leg_handles = extra * nrow
            leg_labels = row_labels

        elif row_labels is None:
            if ncol != len(col_labels):
                raise ValueError(
                    "ncol={0}, but should be equal to len(col_labels)={1}."
                    .format(ncol, len(col_labels)))
            leg_handles = []
            leg_labels = []

        else:
            if nrow != len(row_labels):
                raise ValueError(
                    "nrow = len(handles) // ncol = {0}, but should be equal"
                    " to len(row_labels) = {1}".format(nrow, len(row_labels)))

            if ncol != len(col_labels):
                raise ValueError(
                    "ncol = {0}, but should be equal to len(col_labels) = {1}."
                    .format(ncol, len(col_labels)))
            leg_handles = extra + extra * nrow
            leg_labels = title_label + row_labels

        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels,
                                    ncol=ncol+int(row_labels is not None),
                                    handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


def text_box(text, ax=None, loc='lower left', pad=0.3, borderpad=0.55,
             prop=None, frame=True, **kwargs):
    '''
    Adds a text box to the specified ax. Placement of textbox can be easily
    controlled in a similar fashion than the legend placement.
    Parameters
    -----------
    text: string,
        text to be shown.
    ax: matplotlib axes instance, optional,
        axes object on which the text box should be drawn. Default is the last
        active ax.
    loc: string, optional,
        controlls the placement of the text box. Currently loc=`best` is
        unavaible and will lead to a ValueError. Default is `lower left`.
    pad: float, optional,
        pad around the child for drawing a frame. given in fraction of
        fontsize. Default is 0.4.
    borderpad: float, optional,
        pad between offsetbox frame and the bbox_to_anchor.
    prop: dict, optional,
        used to determine font properties. This is only used as a reference for
        paddings.
    frame: bool, optional,
        whether to put text into a frame or not. Default is `True`.
    Returns
    ----------
    ax: matplotlib axes instance
    '''
    if loc == 'best':
        print('loc `best` is unavaible. loc has changed back to `lower left`')
        loc = 'lower left'
    anchored_text = AnchoredText(text, loc=loc, pad=pad, borderpad=borderpad,
                                 frameon=frame, prop=prop, **kwargs)
    if ax is None:
        ax = plt.gca()
    else:
        pass
    ax.add_artist(anchored_text)
    return ax


def tight_layout(fig=None, pad=0.3, **kwargs):
    """
    Tight layout wrapper that changes the default pad=1.05 to pad=0.3.
    Parameters
    ----------
    fig: matplotlib.figure instance, optional
        figure on which tight_layout should be applied to. If None it is
        applied to the last figure created. The default is None.
    pad: float, optional
        tight_layout padding in units of the fontsize. The default is 0.1.
    **kwargs: dict
        Additional keyword arguments of plt.tight_layout.
    Returns
    -------
    None.
    """
    if fig is None:
        fig = plt.gcf()
    else:
        pass
    fig.tight_layout(pad=pad, **kwargs)
    return


def update_settings(usetex=False, bw=False, style='APS', settings=None):
    """
    Updates matplotlib settings to nicer figures. The function is called when
    private_lib is imported with default keyword arguments.

    Parameters
    ----------
    usetex: bool, optional
        If True, figures are created runing the pdflatex backend.
        The default is False.
    bw: bool, optional
        If True, colorscheme is changed to grayscale. The default is False.
    style: string, optional
        String that determines size of single panel figures.
        Choose from 'APS' and 'Nature'. The default is 'APS'.

    Returns
    -------
    None.

    """

    width = {'APS': 3 + 3/8,
             'Nature': 3 + 1/2,
             'thesis': (5.95 / 2),
             'article': (4.79 / 2),
             }

    gr = (1. + np.sqrt(5)) / 2.  # golden ratio

    params = {
              'pdf.fonttype': 42,
              'ps.fonttype': 42,
              'axes.linewidth': 1.0,
              'axes.labelsize': 9,
              'axes.xmargin': 0.00,  # default is 0.05
              'axes.labelpad': 0.3,
              'axes.formatter.use_mathtext': True,
              'patch.linewidth': 1.0,
              'lines.linewidth': 1.5,
              'lines.markersize': 5,
              'font.size': 9,
              'legend.fontsize': 9,
              'legend.framealpha': 0.0,  # 0.8 is default
              'legend.handlelength': 1.5,  # 2 is deufalt
              'legend.borderaxespad': 0.3,
              'legend.labelspacing': 0.2,  # 0.5 is default
              'legend.fancybox': False,
              'mathtext.fontset': 'cm',
              'xtick.labelsize': 9,
              'ytick.labelsize': 9,
              'xtick.top': True,  # default
              'xtick.bottom': True,  # default
              'ytick.left': True,  # default
              'ytick.right': True,  # default
              'xtick.direction': 'in',  # 'out' is default
              'ytick.direction': 'in',  # 'out' is default
              'xtick.major.pad': 2,  # 3.5 is default
              'ytick.major.pad': 2,  # 3.5 is default
              'ytick.minor.pad': 1.8,
              'ytick.minor.visible': False,
              'xtick.minor.visible': False,  # default
              'savefig.transparent': False,  # default
              'savefig.dpi': 600,  # 100 dpi  is default
              'errorbar.capsize': 3,  # as in the classic version
              'figure.figsize': (width[style], width[style] / gr),
              'font.family': 'serif',
              'font.serif': 'Computer Modern Roman',
              }

    if bw is True:
        plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=[black,  # darkgray
                                                            midgray,
                                                            lightgray])
    else:
        sns.set_palette('colorblind')

    if usetex is True:
        texparams = {'backend': 'PDF',
                     'text.usetex': True,
                     }

        plt.rcParams['text.latex.preamble'] = [#r'\usepackage{lmodern}',
                                               r'\usepackage{siunitx}',
                                               r'\usepackage{amsmath}',
                                               r'\usepackage{physics}',
                                               r'\setlength{\jot}{0ex}',
                                               # r'\usepackage{sansmath}',
                                               # r'\sansmath'
                                               ]

        plt.rcParams['font.family'] = ['serif']
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams.update(texparams)

    if settings:
        params.update(settings)
    plt.rcParams.update(params)

    return