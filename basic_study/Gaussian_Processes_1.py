import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from bokeh.palettes import Category10

output_notebook()

def plot_unit_gaussian_samples(D) :
    p = figure(plot_width=800, plot_height = 500, title = 'Sample from a unit {}D Gaussian'.format(D))
    xs = np.linspace(0,1,D)
    for color in Category10[10] : 
        ys = np.random.multivariate_normal(np.zeros(D),np.eye(D))
        p.line(xs, ys, line_width=1, color = color)

    return p


show(plot_unit_gaussian_samples(2))















