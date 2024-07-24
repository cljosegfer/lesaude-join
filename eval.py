
import torch

from dataloaders.code15text import CODE as DS
from dataloaders.code15text import CODEsplit as DSsplit
from dataloaders.codetest import CODEtest
from models.join import JoinText
from models.baseline import ResnetBaseline
from runners.l1 import Runner
from hparams import BATCH_SIZE, NUM_WORKERS
from utils import json_dump

model_label = 'code15l1'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

database = DS()
tst_ds = CODEtest()

model = JoinText(ResnetBaseline(n_classes = 6))
model = torch.load('output/{}/{}.pt'.format(model_label, model_label))

runner = Runner(device = device, model = model, database = database, Split = DSsplit, model_label = model_label)

val_dl = torch.utils.data.DataLoader(runner.val_ds, batch_size = BATCH_SIZE, 
                                     shuffle = False, num_workers = NUM_WORKERS)
tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size = BATCH_SIZE, 
                                     shuffle = False, num_workers = NUM_WORKERS)

best_f1s, best_thresholds = runner._synthesis(val_dl, best_thresholds = None)
all_binary_results, all_true_labels, metrics_dict = runner._synthesis(tst_dl, best_thresholds)
json_dump(metrics_dict, model_label, test = True)