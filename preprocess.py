import torch
import torch.utils.data as Data


def make_data(sentences, labels, word2idx):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[word] for word in sen.split()])
    targets = []
    for label in labels:
        targets.append(label)
    input_batch, target_batch = torch.LongTensor(inputs), torch.LongTensor(targets)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size=3, shuffle=True)
    return loader
