# AWS S3 Synchronization
This application synchronizes a remote directory with a local directory and viceversa. It does not allow nested file in the bucket-directory.
It allows to synchronize all the bucket-directories or just one.

## Installation
Copy AWS credentials in this directory.
The credentials file should look like this.
```
[default]
aws_access_key_id = SOMETHING
aws_secret_access_key = SOMETHING
```
Make the shell script executable
```shell script
chmod +x install_aws_cli.sh
```
Run the installation script
```shell script
./install_aws_cli.sh
```

## Configuration
The repositories that are going to be synchronized are in the Repository.py file. The list should look like this.
```python
repositories_list = ['slim-bpr-recommender',
                     'slim-elastic-net-recommender',
                     'item-cf-recommender',
                     'item-cbf-recommender',
                     'user-cf-recommender',
                     'user-cbf-recommender',
                     'p3-alpha-recommender',
                     'rp3-beta-recommender',
                     'ials-recommender',
                     'pure-svd-recommender',
                     'funk-svd-recommender',
                     'asy-svd-recommender',
                     'matrix-factorization-bpr-recommender',
                     'twitter-dataset',
                     'twitter-cached-dataset']
```

## Run
In order to synchronize the repositories, run a Python3 shell:
```python
client = RecommenderRepository()
```
To synchronize just one folder:
```python
client.sync_repository('slim-bpr-recommender')
```
To synchronize all the folders:
```python
client.sync_all()
```
* Be careful to synchronize all the repositories. Some of them can contain big files.
## Notes
The synchronized folder, by default, is at PROJECT_ROOT/aws_local