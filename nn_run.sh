#!/bin/bash

# train
python3 nn_dual_output.py like retweet
python3 nn_single_output.py reply
python3 nn_single_output.py comment

# predict validation set
python3 nn_test_dual_output.py like retweet cherry_val
python3 nn_test_single_output.py reply cherry_val
python3 nn_test_single_output.py comment cherry_val

# predict test set
python3 nn_test_dual_output.py like retweet new_test
python3 nn_test_single_output.py reply new_test
python3 nn_test_single_output.py comment new_test
