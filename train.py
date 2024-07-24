
import torch
import torch.nn as nn

from models.join import JoinText
from models.baseline import ResnetBaseline

model_label = 'code15l1'
epochs = 14

from dataloaders.code15text import CODE as DS
from dataloaders.code15text import CODEsplit as DSsplit
from runners.l1 import Runner

model = JoinText(ResnetBaseline(n_classes = 6))
model = torch.load('output/{}/partial.pt'.format(model_label))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

database = DS()
runner = Runner(device = device, model = model, database = database, Split = DSsplit, model_label = model_label)

runner.train(epochs)
runner.eval()
