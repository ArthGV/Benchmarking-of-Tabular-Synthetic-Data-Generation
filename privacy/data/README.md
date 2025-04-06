# Instructions: load and extract data
The **Adult** dataset requires some pre-processing steps before being added to the pipeline of this repository. Please downlaod the data from the UCI website (https://archive.ics.uci.edu/dataset/2/adult), then unzip and place it in the data folder.
After that, run the `extract_adult.py` script: 
```
python data/extract_adult.py --input_path data/adult --output_path data/adult.csv
```

Please also load the metadatafile directly from the Achille's Heel repository.