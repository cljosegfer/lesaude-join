
import os
import torch
import numpy as np

from tqdm import tqdm

from losses import join_l1 as criterion
from hparams import BATCH_SIZE, NUM_WORKERS
from utils import plot_log, export, get_inputs, json_dump, find_best_thresholds, metrics_table

class Runner():
    def __init__(self, device, model, database, Split, model_label = 'baseline'):
        self.device = device
        self.model = model
        self.database = database
        self.model_label = model_label
        if not os.path.exists('output/{}'.format(model_label)):
            os.makedirs('output/{}'.format(model_label))
        
        self.trn_ds = Split(database, database.trn_metadata)
        self.val_ds = Split(database, database.val_metadata)
        self.tst_ds = Split(database, database.tst_metadata)

    def train(self, epochs):
        self.model = self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-4)

        trn_dl = torch.utils.data.DataLoader(self.trn_ds, batch_size = BATCH_SIZE, 
                                             shuffle = True, num_workers = NUM_WORKERS)
        val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)

        log = []
        minimo = 1e6
        for epoch in range(epochs):
            print('-- epoch {}'.format(epoch))
            trn_log = self._train_loop(trn_dl, optimizer, criterion)
            val_log = self._eval_loop(val_dl, criterion)
            log.append([trn_log, val_log])
            plot_log(self.model_label, log)
            if val_log < minimo:
                minimo = val_log
                print('new checkpoint with val loss: {}'.format(minimo))
                export(self.model, self.model_label, epoch)
    
    def _train_loop(self, loader, optimizer, criterion):
        log = 0
        self.model.train()
        for batch in tqdm(loader):
            raw = batch['x']
            label = batch['y']
            text_features = batch['h']

            ecg = get_inputs(raw, device = self.device)
            label = label.to(self.device).float()
            text_features = text_features.to(self.device).float()

            output = self.model.forward(ecg, text_features)
            logits = output['logits']
            signal_embedding = output['signal_embedding']
            text_embedding = output['text_embedding']
            loss = criterion({'logits': logits, 'signal_embedding': signal_embedding}, 
                             {'label': label, 'text_embedding': text_embedding})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log += loss.item()
        return log / len(loader)

    def _eval_loop(self, loader, criterion):
        log = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                raw = batch['x']
                label = batch['y']
                text_features = batch['h']

                ecg = get_inputs(raw, device = self.device)
                label = label.to(self.device).float()
                text_features = text_features.to(self.device).float()

                output = self.model.forward(ecg, text_features)
                logits = output['logits']
                signal_embedding = output['signal_embedding']
                text_embedding = output['text_embedding']
                loss = criterion({'logits': logits, 'signal_embedding': signal_embedding}, 
                                {'label': label, 'text_embedding': text_embedding})

                log += loss.item()
        return log / len(loader)

    def eval(self, partial = True):
        self.model = self.model.to(self.device)
        if partial:
            self.model = torch.load('output/{}/partial.pt'.format(self.model_label))
        val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)
        tst_dl = torch.utils.data.DataLoader(self.tst_ds, batch_size = BATCH_SIZE, 
                                             shuffle = False, num_workers = NUM_WORKERS)
        best_f1s, best_thresholds = self._synthesis(val_dl, best_thresholds = None)
        all_binary_results, all_true_labels, metrics_dict = self._synthesis(tst_dl, best_thresholds)
        json_dump(metrics_dict, self.model_label)
        export(self.model, self.model_label, epoch = None)
    
    def _synthesis(self, loader, best_thresholds = None):
        if best_thresholds == None:
            num_classes = 6
            thresholds = np.arange(0, 1.01, 0.01)  # Array of thresholds from 0 to 1 with step 0.01
            predictions = {thresh: [[] for _ in range(num_classes)] for thresh in thresholds}
            true_labels_dict = [[] for _ in range(num_classes)]
        else:
            all_binary_results = []
            all_true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                raw = batch['x']
                label = batch['y']

                ecg = get_inputs(raw, device = self.device)
                label = label.to(self.device).float()

                output = self.model.predict(ecg)
                logits = output['logits']
                probs = torch.sigmoid(logits)

                if best_thresholds == None:
                    for class_idx in range(num_classes):
                        for thresh in thresholds:
                            predicted_binary = (probs[:, class_idx] >= thresh).float()
                            predictions[thresh][class_idx].extend(
                                predicted_binary.cpu().numpy()
                            )
                        true_labels_dict[class_idx].extend(
                            label[:, class_idx].cpu().numpy()
                        )
                else:
                    binary_result = torch.zeros_like(probs)
                    for i in range(len(best_thresholds)):
                        binary_result[:, i] = (
                            probs[:, i] >= best_thresholds[i]
                        ).float()
                    
                    all_binary_results.append(binary_result)
                    all_true_labels.append(label)
        
        if best_thresholds == None:
            best_f1s, best_thresholds = find_best_thresholds(predictions, true_labels_dict, thresholds)
            return best_f1s, best_thresholds
        else:
            all_binary_results = torch.cat(all_binary_results, dim=0)
            all_true_labels = torch.cat(all_true_labels, dim=0)
            return all_binary_results, all_true_labels, metrics_table(all_binary_results, all_true_labels)