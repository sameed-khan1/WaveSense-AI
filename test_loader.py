from dataset import WidarDataset

dataset = WidarDataset(r"D:\LinkedIn Projects\New folder\BVP")

print("Samples:", len(dataset))

x, y = dataset[0]

print(x.shape, y)