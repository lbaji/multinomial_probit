from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, output_notebook, show
import random

def convergence_plot(data_df, n_choices):
    """Make a convergence plot of the choice probabilities.
    Tapping on a convergence line makes the others fade away.
    
    Args:
        data_df (DataFrame): The column names are the names of the choices and 
            each column contains the calculated probability values of the 
            respective choice depending on the increase of another variable.
        n_choices (int): Number of choices of the problem
            
    Returns:
        Convergence plot with the estimated values of one choice probability and 
        a line of it correct value. 
        
        
    """
    number_columns = data_df.shape[1]
    colors = random.sample(Viridis256, (number_columns + 1))

    xs = [[] for i in range(number_columns + 1)]
    ys = [[] for i in range(number_columns + 1)]
    legends = []

    for i, column in enumerate(data_df.columns):
        ys[i] = list(data_df[column])
        xs[i] = list(data_df.index)
        legends.append(column)
        
    ys[-1] = [1 / n_choices]*len(data_df)
    xs[-1] = list(data_df.index)
    legends.append('right choice prob')

    source_data = {'xs': xs, 'ys': ys, 'colors': colors, 'legends': legends}
    source = ColumnDataSource(source_data)
    
    #Make plot
    p = figure(plot_height=350, tools="tap,reset,save")
    p.title.text = 'Convergence Plot'

    p.multi_line(xs='xs', ys='ys', color='colors', line_width=4, legend='legends', 
                 nonselection_alpha=0.2, source=source)

    output_notebook()

    return show(p)