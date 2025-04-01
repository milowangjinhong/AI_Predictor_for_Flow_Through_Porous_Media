import os
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

# Figure saving
def save_fig_custom(fig, file_path='', file_name='fig', 
                    format_list=['.eps', '.pdf'], overwrite=False):
    
    if (file_path != '') and (file_path[-1] != '/'): 
        file_path = file_path+'/'
        
    if (file_path != '') and (os.path.isdir(file_path) == False):
        os.makedirs(file_path, exist_ok=True)
        print('Save directory %s is created'%(file_path))
    
    for save_format in format_list:
        if save_format[0] != '.': save_format = '.' + save_format

        file_name_now = file_path+file_name+save_format
        if not overwrite:
            i = 1
            while os.path.exists(file_name_now):
                file_name_now = file_path+file_name + '%i'%(i)+save_format
                i = i+1
        
        fig.savefig(file_name_now, 
                    facecolor='white', bbox_inches="tight")
        
        print(file_name_now, 'is saved.')
    
    return

# figsize tuning
def figure_tuning(fig, fig_size = (7,5)):

    fig.set_size_inches(fig_size[0], fig_size[1]) 

    return fig

# axes tuning
def axes_tuning(ax, fig_size = (),
                x_lim = [], y_lim = [],
                title = '', title_font = 14, y = 1.0,
                x_label='',
                y_label='', label_font=0,
                tick_font=0, 
                legend_font=0, legend_loc = 0):
    
    if title == '':
        title = ax.get_title()
    if x_label == '':
        x_label = ax.get_xlabel()
    if y_label == '':
        y_label = ax.get_ylabel()
    
    if title_font != 0:
        if y == 0:
            ax.set_title(title,fontsize=title_font)
        else:
            ax.set_title(title,fontsize=title_font, y=y)
        
        if x_lim != []: ax.set_xlim(x_lim)
        if y_lim != []: ax.set_ylim(y_lim)
    
    if tick_font != 0:
        ax.xaxis.set_tick_params(labelsize=tick_font)
        ax.yaxis.set_tick_params(labelsize=tick_font)

    if label_font != 0:
        ax.set_xlabel(x_label,fontsize=label_font)
        ax.set_ylabel(y_label,fontsize=label_font)
    
    if legend_font != 0: ax.legend(loc=legend_loc, prop={'size': legend_font})
    
    fig = ax.get_figure()
    # Adjust the figure size
    if fig_size != (): fig.set_size_inches(fig_size[0], fig_size[1]) 

    return ax

def general_comparison_on_xy_plot(x, y_list, label_list, ax = 'new', 
                                  colour_cycle=[], line_cycle=[],line_style_list=[],linewidth=1,
                                  title = 'Title',
                                  x_label='X Axis', y_label='Y Axis', 
                                  x_lim = [], y_lim = [], x_log = False, y_log = False,
                                  Grid = True, Legend = True, show_fig = True):
    
    if ax=='new': fig, ax = plt.subplots()

    if colour_cycle == []: colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if line_cycle != []: 
        colour_cycle = colour_cycle[0:len(line_cycle)]
    else:
        line_cycle = ['-']*len(colour_cycle)
    
    if colour_cycle != [] or line_cycle != []:
        ax.set_prop_cycle(color=colour_cycle, linestyle=line_cycle)
    
    if line_style_list == []:
        line_style_list = ['-'] * len(y_list)
    
    for i in range(len(y_list)):
        ax.plot(x, y_list[i], label=label_list[i], linestyle = line_style_list[i], linewidth=linewidth)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if x_lim != []: ax.set_xlim(x_lim)
    if y_lim != []: ax.set_ylim(y_lim)

    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')
    
    if Legend : ax.legend() #plt.legend()
    if Grid: ax.grid()
    
    fig = ax.get_figure()#plt.gcf()
    
    if show_fig==False:
        plt.close('all')
    else:
        plt.show()
    
    return fig, ax