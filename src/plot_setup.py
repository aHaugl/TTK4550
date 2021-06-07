import matplotlib
from matplotlib import pyplot as plt
# to see your plot config


def setup_plot():
    print(f"matplotlib backend: {matplotlib.get_backend()}")
    print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
    print(f"matplotlib config dir: {matplotlib.get_configdir()}")
    plt.close("all")

    # try to set separate window ploting
    if "inline" in matplotlib.get_backend():
        print("Plotting is set to inline at the moment:", end=" ")

        if "ipykernel" in matplotlib.get_backend():
            print("backend is ipykernel (IPython?)")
            print("Trying to set backend to separate window:", end=" ")
            import IPython

            IPython.get_ipython().run_line_magic("matplotlib", "")
        else:
            print("unknown inline backend")

    print("continuing with this plotting backend", end="\n\n\n")
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    # matplotlib.rc('font', **font)
    # set styles
    try:
        # installed with "pip install SciencePLots"
        # (https://github.com/garrettj403/SciencePlots.git)
        # gives quite nice plots
        plt_styles = ["science", "grid", "no-latex"]
        # plt.style.use(plt_styles)
        # plt.style.use("science")
        print(f"pyplot using style set {plt_styles}")
    except Exception as e:
        print(e)
        print("setting grid and only grid and legend manually")
        plt.rcParams.update(
            {
                # setgrid
                "axes.grid": True,
                "grid.linestyle": ":",
                "grid.color": "k",
                "grid.alpha": 0.5,
                "grid.linewidth": 0.5,
                # Legend
                "legend.frameon": True,
                "legend.framealpha": 1.0,
                "legend.fancybox": True,
                "legend.numpoints": 1,
            }
        )
