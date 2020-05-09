from Utils.Data.DataUtils import create_all
from Utils.Data.Split.HoldoutSplit import holdout_split_train_test

if __name__ == '__main__':
    holdout_split_train_test(input_dataset_id="train", pc_hold_out=0.20)
