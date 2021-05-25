from os.path import join


class Dataset:
    """"Dataset"""
    def __init__(self, dataset_dir):
        """
            train_dataset, test_dataset: list of strings
            train_label: np.array
        """
        self.dataset_dir_ = dataset_dir
        self.train_dataset_dir = join(dataset_dir, 'train')
        self.test_dataset_dir = join(dataset_dir, 'test')