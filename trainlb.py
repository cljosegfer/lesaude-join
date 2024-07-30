
import torch

# model_label = 'linearprob'
# texth5_path = '/home/josegfer/datasets/code/output/BioBERTpt_text_report.h5'
# epochs = 50

model_label = 'linearprobcls'
texth5_path = '/home/josegfer/datasets/code/output/BioBERTpt_text_report_cls.h5'
epochs = 125

from dataloaders.code15lb import CODElb as DS
from dataloaders.code15lb import CODElbsplit as DSsplit
from models.linearprob import LinearProb
from runners.prob import Runner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

database = DS(texth5_path = texth5_path)
model = LinearProb(n_classes = 6)
# model = torch.load('output/{}/partial.pt'.format(model_label))
runner = Runner(device = device, model = model, database = database, Split = DSsplit, model_label = model_label)

runner.train(epochs)
runner.eval()
