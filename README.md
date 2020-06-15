# RecSys Challenge 2020

These are the required steps to reproduce our results:
* `pip install -r requirements.txt` with python3 (better if python 3.8)
* Place train and test CSV files in `Dataset` folder in gzip format. They must be called `train.csv.gz` and `test.csv.gz`.
* Run `Utils/Data/Split/HoldoutCherrySplit.py`.
* Run `nn_run.sh`.
* Copy `Blending/last_submission_sub.py` into the root folder of the project and run it.

Please note that both `nn_run.sh` and `last_submission_sub.py` require a GPU.