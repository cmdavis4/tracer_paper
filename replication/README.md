[![DOI](https://zenodo.org/badge/1000466273.svg)](https://doi.org/10.5281/zenodo.15659853)

This repository contains the code associated with the manuscript "The Entrainment of Air from Rainy Surface Regions and its Implications for Bioaerosol Transport in Three Deep Convective Storm Morphologies" (Davis et al.) To create a python environment that can run the `analysis-tracerpaper.ipynb` notebook, first install [conda](https://anaconda.org/anaconda/conda) following the documentation. Then clone this respository and enter it. Create a conda environment using the included requirements file:
```
conda create -n cd_etal_tracer_paper --file requirements_conda.txt
```
Activate the environment with
```
conda activate -n cd_etal_tracer_paper
```
(or whatever you choose to call it.) Finally, install the required packages that are not available through conda:
```
pip install -r requirements_pip.txt
```
Then launch a jupyter server using your preferred method, for example:
```
jupyter lab
```
The notebook includes descriptions of changes you may need to make to get the code to work based on your filesystem etc.

The RAMS output itself cannot be included due to the size of the data. The modified version of RAMS used to conduct these simulations is contained in the `RAMS_tracer_source` folder and can be built according to the documentation in this folder. Once built, the simulations can be reproduced using the RAMSINS in the "data/{ic,sl,sc}_wk" folders for the isolated convective storm, squall line, and supercell, respectively. You will need to change the filepaths in the RAMSINs based on your filesystem, but no other changes should be necessary. RAMS can then be run following its documentation.

The data underlying each figure are included in `data/figure_data`. Each figure's data
is saved using python's [pickle](https://docs.python.org/3/library/pickle.html) module. When read into python, each file will produce a dictionary, each value of which corresponds to the data in one element of the figure. Specifically, each value in the dictionary is a tuple of the data arguments passed to the relevant matplotlib function call in `analysis-tracerpaper.ipynb`. The keys of the dictionary are intended to be descriptive enough such that it should be clear which element of the figure they correspond to, but the source code of `analysis-tracerpaper.ipynb` can be referred to in case of ambiguity.

For any further questions about this code or data, please contact Charles Davis (cmdavis4@colostate.edu).