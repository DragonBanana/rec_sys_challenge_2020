# RecSys Challenge 2020

## Requirements
In order to run the code it is necessary to have:
* **Python**: version 3.8. 
* **Pip**: version 20.1.1.
* **CUDA** is also required.

If you do not have Python already installed, you can find it here (https://www.python.org/downloads/).

Install the python dependecies with the following bash command:
```shell script
pip install -r requirements.txt
```

It is also required to have the datasets already downloaded, compressed with GZip and renamed as **new_train.csv.gz** and **new_test.csv.gz**.
We assume these datasets are placed in the **./Dataset** folder.

If you do not have the dataset, you can download from here (https://recsys-twitter.com/data/show-downloads, registration is required).

## Run the code

Split the **new_train.csv.gz** dataset into **holdout_new_train.csv.gz** and **holdout_new_test.csv.gz**.

```shell script
cp ./Utils/Data/Split/HoldoutSplit.py .
python HoldoutSplit.py
```

Split the **new_train.csv.gz** dataset into **cherry_train.csv.gz** and **cherry_val.csv.gz**.

```shell script
cp ./Utils/Data/Split/HoldoutCherrySplit.py .
python HoldoutCherrySplit.py
```

Train the neural network:
```shell script
chmod +x nn_run.sh
./nn_run.sh
```
this script trains a bunch of NNs and places the infered probabilities in the folder "**./Dataset/Features/{test_dataset}/ensembling**" with filename "**nn_predictions_{class_label}_{model_id}.csv**"

For our last submissions please use:
```shell script 
# RUN 1 - submission last_test_sub
cp ./Blending/last_test_sub/* .
python last_blending_sub_comment.py comment
python last_blending_sub_like.py like
python last_blending_sub_retweet.py retweet
python last_blending_sub_reply.py reply
```

```shell script 
# RUN 2 - submission last_test_sub_2
cp ./Blending/last_test_sub_2/* .
python last_blending_sub_comment.py comment
python last_blending_sub_like.py like
python last_blending_sub_retweet.py retweet
python last_blending_sub_reply.py reply
```
