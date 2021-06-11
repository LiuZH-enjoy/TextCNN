import torch
import argparse
import train
import module
import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,
                    default=3, help='batch大小（默认3）')
parser.add_argument('--epochs', type=int, default=5000, help='epoch数（默认50）')
parser.add_argument('--embedding_size', type=int,
                    default=2, help='embedding维度（默认2）')
parser.add_argument('--num_classes', type=int, default=2, help='分类数，默认为2')
parser.add_argument('--vocab_size', type=int, default=16, help='分类数，默认为2')
args = parser.parse_args()

args.device = torch.device("cpu")
args.cuda = False

dtype = torch.FloatTensor
model = module.TextCNN(args)

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
args.vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}


loader = preprocess.make_data(sentences, labels, word2idx)
train.train(loader, model, args)

test_text = 'i love me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(args.device)
# Predict
model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")

