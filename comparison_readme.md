The current issue is that there are slight differences in how the simulations of the three storms used in this study are set up. This is worth investigating because the setups are supposed to be completely apples-to-apples, besides the few intentional differences. The original datasets are:

- Isolated convection: /moonbow/cmdavis4/projects/bl_transport/rams_io/isolated_convection_wk/all_tracers/newtracer_forcing-warmbubble4_deltax-500_frqlite-2_nobulku
- Squall line: /moonbow/cmdavis4/projects/bl_transport/rams_io/squall_line_wk/all_tracers/newtracer_blspinup-no_forcing-infinitecoldbubble4_deltax-500_frqlite-2
- Supercell: /moonbow/cmdavis4/projects/bl_transport/rams_io/supercell_wk/all_tracers/newtracer_forcing-grant2014_dx-500_frqlite-2_stratwinds

Besides there being intentional differences between the storms' low-level moisture, all of the storms have the addition of a well-mixed boundary layer to the Weisman-Klemp sounding. The intentional difference in the low-level moisture is that the isolated convective storm uses 11 g/kg, while the other two use 14 g/kg (or it might be 13, but it just matters that they're the same.) The unintentional differences between the storms' setups are:
- The supercell's pressure profile starts at 1000 hPa, while the other two storms' start at MSL, offsetting their temperature profiles by ~1 Kelvin at the same physical height
- The potential temperature structure of the isolated convection and squall line are incorrect in the stratosphere because of effects from erroneously using an initial profile that was not hydrostatically balanced
- The low-level moisture in the isolated convection and squall line are incorrect to a few percent because of an error in the calculation

I have run several experiments to test how consequential these differences are:
- The squall line and supercell are supposed to have the exact same thermodynamic profile (ie temp/moisture/pressure, but not wind), so I ran the supercell with the actual values used for the squall line; this is at /moonbow/cmdavis4/projects/bl_transport/rams_io/from_cumulus/sc_wk/sl_profile
- I also tried running the isolated convective storm and squall line using the supercell's profile, with the isolated convection's low-level moisture adjusted because that is how we set up the experiment. A well-mixed boundary layer is also imposed in the same way as the supercell.

I need the following:
- A characterization of how large the erroneous differences in the initial setups actually are between the *original* datasets
- An analysis of whether these differences are impactful for the actual results of the study, presumably by comparing the three storms within each of the two ways in which I have tried to "harmonize" their setups

The main focus is on **whether the setup differences make the comparisons between storms unfair**, ie bias the results for each storm differently. I have two notebooks already made that should make this fairly straightforward:
- validation_comparison.ipynb: setup to compare the results of the analysis for two different sets of exeriments; and
- compare_wk_sounding.ipynb: setup to compare details in the initial structure between two simulation setups

Create a notebook for each way of "harmonizing" the setups that contains whatever is useful from both of these template notebooks.

