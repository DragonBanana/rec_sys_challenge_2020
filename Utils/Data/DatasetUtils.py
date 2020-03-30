
def is_test_or_val_set(dataset_id: str):
    is_test = dataset_id[:4] == "test"
    is_val = dataset_id[:3] == "val"
    return is_test or is_val

def get_train_set_id_from_test_or_val_set(dataset_id: str):
    assert is_test_or_val_set(dataset_id)
    is_test = dataset_id[:4] == "test"
    is_val = dataset_id[:3] == "val"
    if is_test:
        return "train" + dataset_id[4:]
    if is_val:
        return "train" + dataset_id[3:]

def get_test_or_val_set_id_from_train(dataset_id: str):
    assert not is_test_or_val_set(dataset_id)
    if dataset_id == "train":
        return "test"
    else:
        return "train" + dataset_id[5:]

