from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, graphlist):
        super().__init__()
        self.graphlist = graphlist

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, index):
        return self.graphlist[index]