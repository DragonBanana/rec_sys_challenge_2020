
TRAIN_SET_IDS = [
    "train",
    "train_days_1",
    "train_days_12",
    "train_days_123",
    "train_days_1234",
    "train_days_12345",
    "train_days_123456"
]

TEST_SET_IDS = [
    "test"
]

VAL_SET_IDS = [
    "val_days_2",
    "val_days_3",
    "val_days_4",
    "val_days_5",
    "val_days_6",
    "val_days_7"
]

TRAIN_TEST_SET_PAIRS = {
    "train": "test",
    "train_days_1": "val_days_2",
    "train_days_12": "val_days_3",
    "train_days_123": "val_days_4",
    "train_days_1234": "val_days_5",
    "train_days_12345": "val_days_6",
    "train_days_123456": "val_days_7"
}

TEST_TRAIN_SET_PARIS = {v: k for k, v in TRAIN_TEST_SET_PAIRS.items()}

def is_test_or_val_set(dataset_id: str):
    return dataset_id in TEST_SET_IDS or dataset_id in VAL_SET_IDS

def get_train_set_id_from_test_or_val_set(dataset_id: str):
    assert is_test_or_val_set(dataset_id)
    return TEST_TRAIN_SET_PARIS[dataset_id]

def get_test_or_val_set_id_from_train(dataset_id: str):
    assert not is_test_or_val_set(dataset_id)
    return TRAIN_TEST_SET_PAIRS[dataset_id]

