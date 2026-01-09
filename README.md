
Materials for Dunn-Sigouin et al. 2026. The project is organized as an installable conda package. To get setup, first pull the directory from github to your local machine:

``` bash
$ git clone https://github.com/edunnsigouin/cf-Dunnsigouin_etal_2026/
```

Then install the conda environment:

``` bash
$ conda env create -f environment.yml
```

Then install the project package:

``` bash
$ python setup.py develop
```

Finally change the project directory in cf-Dunnsigouin_etal_2026/config.py to your local project directory


