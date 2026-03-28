import src.config as config
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

data_load_name="DistilBert"

def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    # Your implementation here. I'll put dummy's implementation here for now, we should change it.
    print("hello from the dummy data.")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return trainset

