import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
from data import VideoDataset

from math import sqrt
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              #'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


HLclip = VideoFileClip(r"data/HL/HL-liverpool-vs-west-ham_12.mp4")
cut = lambda i: HLclip.audio.subclip(i,i+1).to_soundarray(fps=22000)
volume = lambda array: np.sqrt(((1.0*array)**2).mean())
HLvolumes = np.array([volume(cut(i)) for i in range(0,int(HLclip.duration-1))])

NHLclip = VideoFileClip(r"data/NO_HL/NOHL-liverpool-vs-west-ham_0174.mp4")
cut = lambda i: NHLclip.audio.subclip(i,i+1).to_soundarray(fps=22000)
volume = lambda array: np.sqrt(((1.0*array)**2).mean())
NHLvolumes = np.array([volume(cut(i)) for i in range(0,int(NHLclip.duration-1))])


# df = pd.DataFrame(np.stack([np.array(range(volumes.size)), volumes]).transpose())
# df.columns = ['Column 1', 'Column 2']

latexify()

fig, (ax1, ax2) = plt.subplots(2,1, sharey=True)
ax1.plot(np.array(range(HLvolumes.size)),HLvolumes)
ax1.set_xlabel("Time")
ax1.set_ylabel("Energy")
# ax1.set_title("Highlight")
format_axes(ax1)

ax2.plot(np.array(range(NHLvolumes.size)),NHLvolumes)
ax2.set_xlabel("Time")
ax2.set_ylabel("Energy")
# ax2.set_title("Non-Highlight")
format_axes(ax2)
plt.show()

