#### Example of how a plotting call might be done previously
figure_name = "f01_skewt_logp"  # Set figure name
figure_data = {}  # Dictionary to store data


plot_and_save(
    figure_data,
    f"storm-{storm}_skewt_temperature",  # Artist name
    wk_skewt.plot,  # Plotting function
    lite_ds["P"].values,  # Plotting function arguments
    convert_np(lite_ds["T"].values, "K", "degC"),
    "r",
    label="Temperature",
)
save_figure_data(figure_data, figure_name)  # Save figure data
cm.plotting.fig_multisave(fig, figure_name, blt.FIGS_DIR)  # Save figure


#### Example of how it should be done now
cm.plotting.start_figure("f01_skewt_logp")  # Set figure name
# Don't need to manually pass around figure data dict

wk_skewt.plot(  # Plotting function
    lite_ds["P"].values,  # Plotting function arguments
    convert_np(lite_ds["T"].values, "K", "degC"),
    "r",
    label="Temperature",
    f"storm-{storm}_skewt_temperature",  # Artist name
)
save_figure_and_data()  # Save figure and figure data