from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def match_run(run, conditions, params=dict()):
    """Returns true if run matches conditions and params """
    satisfies_base_conditions = all(run.config[name] == value for name, value in conditions.items())

    if not satisfies_base_conditions: return False
        
    satisfies_param_conditions = all(
        any(run.config["params"][name] == value for value in values)
        for name, values in params.items()
    )
    
    return satisfies_param_conditions



def add_data_to_ax(runs, run_conditions, param_conditions, valtype, ax, colors, batches=[100, 4000], additional_func=None, legend=False, title=""):
    min_batch_size, max_batch_size = batches
    data = []

    for i, run in enumerate(runs):
        if not match_run(run=run, conditions=run_conditions, params=param_conditions):
            continue
        losses = run.summary[valtype][:run_conditions["epochs"]]

        batch_size = run.config["batch_size"]
        if batch_size < min_batch_size or batch_size > max_batch_size:
            continue

        data.append([batch_size, losses])

    # sort data on batch size
    data = sorted(data, key=lambda x: x[0])

    for batch_size, losses in data:
        ax.plot(losses, label=f"{batch_size}", color=colors[batch_size])

    ax.set_ylabel(valtype)
    ax.set_xlabel("Epoch")
    ax.set_title(title)

    if additional_func is not None:
        additional_func(data, ax)

    if legend:
        ax.legend(title="Batch size")


def get_regularizer_runs(runs, regularizer, n, d, init="gaussian", param_conditions=dict()):
    conditions = dict(regularizer=regularizer, n=n, d=d, init=init)
    reg_runs = [run for run in runs if match_run(run=run, conditions=conditions, params=param_conditions)]
    return reg_runs


def get_zoomed_subplot(x1, x2, y1, y2, colors, zoom=10):
    def add_zoomed_plot(data, ax):
        axins = zoomed_inset_axes(ax, zoom, loc=1)

        for batch_size, losses in data:
            axins.plot(losses, label=f"{batch_size}", color=colors[batch_size])

        # sub region of the original image
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.set_xticks([])
        axins.set_yticks([])

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # plt.draw()
    return add_zoomed_plot
