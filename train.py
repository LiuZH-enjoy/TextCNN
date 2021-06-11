import torch.nn as nn
import torch.optim as optim


def train(train_loader, model, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            if (epoch+1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch+1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()