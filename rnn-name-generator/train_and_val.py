import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import torch.nn as nn
import math
from model import Rnn
import json
import argparse
from charset import default_charsets, START_CHAR, END_CHAR
import os

parser = argparse.ArgumentParser()
parser.add_argument(
  '--epochs',
  type=int,
  default=30,
  help='number of epochs to train for',
)
parser.add_argument(
  '--base_lr',
  type=float,
  default=0.008,
  help='base learning rate',
)
parser.add_argument(
  '--checkpoint_path',
  type=str,
  default='checkpoint.pth',
  help='checkpoint file to use',
)
parser.add_argument(
  '--train_path',
  type=str,
  help='file of line-separated words to learn to generate',
)
parser.add_argument(
  '--val_path',
  type=str,
  help='file of line-separated words to validate model against',
)
parser.add_argument(
  '--tracking_path',
  type=str,
  default='tracking.jsons',
  help='where to write train and val metrics per epoch',
)
parser.add_argument(
  '--charset',
  type=str,
  default='pause_alpha_num',
  help='which pre-defined charset to use. see charset.py for the map',
)
parser.add_argument(
  '--hidden_sizes',
  type=str,
  default='150,150,150',
  help='sizes of the intermediate hidden layers',
)
args = parser.parse_args()

epochs = args.epochs
base_lr = args.base_lr
min_lr_factor = 0.005
tracking_path = args.tracking_path
#specify depth and width of RNN layers here
#input  is n_char + 1 for characters (including stop char and start char)
#output is n_char for characters (including stop char)
charset = default_charsets[args.charset]
hidden_sizes = []
for size in args.hidden_sizes.split(','):
  hidden_sizes.append(int(size))
rnn = Rnn([charset.n_char + 1] + hidden_sizes + [charset.n_char])


train_names = []
val_names = []
for line in open(args.train_path, 'r'):
  assert line[-1] == '\n'
  train_names.append(line)
for line in open(args.val_path, 'r'):
  assert line[-1] == '\n'
  val_names.append(line)
n_train = len(train_names)
n_val = len(val_names)

criterion = nn.NLLLoss()

def train_batch(name, lr):
  rnn.zero_grad()
  stuff = eval_batch(name, is_training=True)
  loss = stuff['ll']
  loss.backward()
  for p in rnn.parameters():
    p.data.add_(-lr, p.grad.data)

  return loss.item(), stuff['correct_chars']

def eval_batch(name, is_training=False):
  l = len(name)
  tens = Variable(charset.name_to_batch(name))
  hidden = rnn.init_hidden()
  correct_names = 1
  correct_chars = 0
  ll = 0

  for i in range(l):
    inp = tens[i - 1] if i > 0 else Variable(charset.char_to_tens(START_CHAR))
    output, hidden = rnn(inp, hidden, is_training=is_training)
    logp = torch.log(output + 1E-7)
    values, indices = torch.max(output, 1)
    next_char_idx = charset.index(name[i])
    target = Variable(torch.LongTensor([next_char_idx]))
    ll += criterion(logp, target)

    if next_char_idx == indices.data.numpy()[0]:
      correct_chars += 1
    else:
      correct_names = 0
  
  return {
    'correct_names': correct_names,
    'correct_chars': correct_chars,
    'total_names': 1,
    'total_chars': l,
    'll': ll
  }

def summarize_metrics(metrics):
  res = {}
  for metric in metrics:
    for k, v in metric.items():
      if k not in res:
        res[k] = 0.0
      res[k] += v
  res['ll'] = res['ll'].item()
  ll = res['ll'] / res['total_names']
  ll_per_char = res['ll'] / res['total_chars']
  char_acc = res['correct_chars'] / res['total_chars']
  name_acc = res['correct_names'] / res['total_names']
  res['eval_accuracy'] = char_acc
  res['eval_loss'] = ll_per_char
  print('EVAL CHAR NEGATIVE LL:\t{}'.format(ll_per_char))
  print('EVAL CHAR ACCURACY:\t{}'.format(char_acc))
  return res

try:
  os.remove(tracking_path)
except:
  pass

for i in range(epochs):
  np.random.shuffle(train_names)
  theta = np.pi * i / float(epochs - 1)
  lr = base_lr * (min_lr_factor + (1.0 - min_lr_factor) * (1.0 + np.cos(theta)) / 2.0)

  total_chars = 0
  total_loss = 0
  total_correct = 0
  for j in range(n_train):
    name = train_names[j]
    loss, correct = train_batch(name, lr)
    total_chars += len(name)
    total_correct += correct
    total_loss += loss
    if math.isnan(loss):
      print("OH NOES, LOSS IS NAN AROUND {}".format(train_names[j - 1: j + 2]))
    if j % 100 == 0:
      print(i, j, name.rstrip(END_CHAR), loss / len(name))
      print('generated name: {}'.format(rnn.generate_name(charset)))
  train_loss = total_loss / total_chars
  train_accuracy = total_correct / float(total_chars)
  print('TRAIN CHAR ACCURACY {}'.format(train_accuracy))
  print('TRAIN CHAR NEGATIVE LL {}'.format(train_loss))

  metrics = []
  for j in range(n_val):
    name = val_names[j]
    metrics.append(eval_batch(name))
  summary = summarize_metrics(metrics)
  summary['epoch'] = i + 1
  summary['lr'] = lr
  summary['train_loss'] = train_loss
  summary['train_accuracy'] = train_accuracy

  tracking_stream = open(tracking_path, 'a')
  json.dump(summary, tracking_stream)
  tracking_stream.write('\n')
  tracking_stream.close()
  torch.save(rnn.state_dict(), args.checkpoint_path)
