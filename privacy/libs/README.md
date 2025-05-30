# External libraries required to install MIA-synthetic on JupyterHub
Some changes have been made to these libraries to for compatilibity purposes.
Also, some libraries are required multiple times with different version. Installation needs to be done sequentially in a particular order following below steps:
* It's very important to take the edited libraries
Reprosyn: adapted to work correctly with ctgan 0.10.2
    ○ MIA-Synthetic: adapted to export other kind of data
* Create a new python 3.9 environment using makefiles
    * Activate the env
    * Go to the reprosyn directory
        ○ # Not necessary if you add the correct ctgan version. 
            § In this directory, change method.gans.gans
            § # bioeLine 5: from ctgan import CTGAN as SVDGAN
            § Line 113: self.ctgan = SVDGAN(**self.params)
* Add poetry==1.8.4 in requirements.txt (only that)
* Change the pyproject.toml file to comment access to external git repositoriess
* Install private-pgm manually by running "pip install ." inside the repo
* Download the querysnoot repository 
* Go to src/optimized_qbs
* install it using "python setup.py install"
* Go back to the reprosyn folder
* Run poetry lock --no-update
* Run pip install scikit-learn
* Run pip install ctgan==0.10.2
* Run pip install click
* Take the updated reprosyn/pyproject.toml file
* Run pip install .
* Pip install synchcity