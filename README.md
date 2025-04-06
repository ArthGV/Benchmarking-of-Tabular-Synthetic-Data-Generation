# Benchmarking-of-Tabular-Synthetic-Data-Generation
## Installation guide: 

After installing micromamba, run the install_tapas_env.sh file to setup the tapas-upgrade environment and install all the packages in the correct order. 


IMPORTANT
- you need to change the path to the correct path of wherever you have your Benchmarking-of-Tabular-Synthetic-Data-Generation project file.
  
(if not on macOS): 
- you might need to change the setup of the shell for micromamba in the first line (eval "$(micromamba shell hook --shell bash)") to work for your specific shell, this is for the terminal on macbook.
- you might need to change the file paths if on windows (\ vs. /) 


## After setting up the environment 
After running the script you should be in the tapas-upgrade environment, from there i run: 

- python -m ipykernel install --user --name tapas-upgrade --display-name "Python (tapas-upgrade)"
- micromamba run -n tapas-upgrade jupyter notebook

which opens a tab in the browser, where you can run the notebooks

