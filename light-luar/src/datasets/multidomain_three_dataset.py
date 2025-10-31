import random

from torch.utils.data.dataset import Dataset

from src.datasets.utils import get_dataset

class Multidomain_Three_Dataset(Dataset):
    """Dataset Object for the Multi-Domain training of three datasets.

       Expects dataset name to be: "dataset_1+dataset_2+dataset_3".
    """

    # TODO: remove num_sample_per_author from everything
    def __init__(self, params, dataset_split_name, is_queries=True):
        """`dataset` parameter is not used.
        """

        self.params = params
        self.dataset_split_name = dataset_split_name
        self.num_sample_per_author = self.params.num_sample_per_author

        # ensure correct input to dataset_name: "dataset_1+dataset_2+dataset_3"
        dataset_names = self.params.dataset_name.split("+")
        print(self.params.dataset_name)
        assert len(dataset_names) == 3
        
        only_queries = (is_queries == True) and (dataset_split_name in ["validation", "test"])
        only_targets = (is_queries == False) and (dataset_split_name in ["validation", "test"])

        # HACK: temporarily modifying cmd-line arguments to load datasets
        params.dataset_name = dataset_names[0]
        self.dataset_1 = get_dataset(params, dataset_split_name, only_queries, only_targets)
        params.dataset_name = dataset_names[1]
        self.dataset_2 = get_dataset(params, dataset_split_name, only_queries, only_targets)
        params.dataset_name = dataset_names[2]
        self.dataset_3 = get_dataset(params, dataset_split_name, only_queries, only_targets)
        self.params.dataset_name = "+".join(dataset_names)
        
        self.multidomain_prob_1 = 1/3
        self.multidomain_prob_2 = 1/3

        if self.params.multidomain_prob is not None:
            assert 0.0 < self.params.multidomain_prob < 1.0
            self.multidomain_prob_1 = self.params.multidomain_prob / 2
            self.multidomain_prob_2 = self.params.multidomain_prob / 2

    def __len__(self):
        return len(self.dataset_1) + len(self.dataset_2) + len(self.dataset_3)

    def __getitem__(self, index):
        """`index` will be ignored here in order to ensure equal sampling across datasets.
        """
        if self.dataset_split_name in ["validation", "test"]:
            data = self.sample_val_or_test(index)
        else:
            data = self.sample_training()

        return data

    def sample_val_or_test(self, index):
        if index < len(self.dataset_1):
            text, author = self.dataset_1[index]
            to_return = [text, author]
        elif index < len(self.dataset_1) + len(self.dataset_2):
            text, author = self.dataset_2[index - len(self.dataset_1)]
            # Offset author labels
            author += len(self.dataset_1)
            to_return = [text, author]
        else:
            text, author = self.dataset_3[index - len(self.dataset_1) - len(self.dataset_2)]
            # Offset author labels
            author += len(self.dataset_1) + len(self.dataset_2)
            to_return = [text, author]
        
        return to_return

    def sample_training(self):
        to_return = []
        
        rand_val = random.random()
        if rand_val < self.multidomain_prob_1:
            new_index = random.randint(0, len(self.dataset_1) - 1)
            text, author = self.dataset_1[new_index]
        elif rand_val < self.multidomain_prob_1 + self.multidomain_prob_2:
            new_index = random.randint(0, len(self.dataset_2) - 1)
            text, author = self.dataset_2[new_index]
            author += len(self.dataset_1)
        else:
            new_index = random.randint(0, len(self.dataset_3) - 1)
            text, author = self.dataset_3[new_index]
            author += len(self.dataset_1) + len(self.dataset_2)
        
        to_return.extend([text, author])
        return to_return
