import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(args.vocab_size, args.embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (2, args.embedding_size)),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),
        )
        # fc
        self.fc = nn.Linear(output_channel, args.num_classes)

    def forward(self, X):
        '''
            X: [batch_size, sequence_length]
        '''
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X) # [batch_size, output_channel, 1, 1]
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        return output
