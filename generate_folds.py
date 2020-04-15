from Utils.Data.DataUtils import create_all
from Utils.Data.Split.TimeSplit import split_train_val, split_train_val_multiple

if __name__ == '__main__':

    train_filename_list = [
        "train_days_1",
        "train_days_12",
        "train_days_123",
        "train_days_1234",
        "train_days_12345",
        "train_days_123456"
    ]

    test_filename_list = [
        "val_days_2",
        "val_days_3",
        "val_days_4",
        "val_days_5",
        "val_days_6",
        "val_days_7"
    ]

    train_days_list = [
        1,
        2,
        3,
        4,
        5,
        6
    ]

    test_days_list = [
        1,
        1,
        1,
        1,
        1,
        1
    ]

    split_train_val_multiple(
        train_filename_list,
        test_filename_list,
        train_days_list,
        test_days_list
    )