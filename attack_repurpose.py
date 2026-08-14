import argparse, csv, logging
from pathlib import Path
import torch, torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from maleficnet import initialize_model, device
from injector import Injector
from extractor import Extractor
from dataset.cifar10 import CIFAR10
from dataset.cifar100 import CIFAR100

log = logging.getLogger(); log.addHandler(logging.StreamHandler())
DATA = {'cifar10': CIFAR10, 'cifar100': CIFAR100}

class SNRTracker(Callback):
    def __init__(self, ext, rand, ml, pay):
        self.ext, self.rand, self.ml, self.pay, self.epoch, self.history = ext, rand, ml, pay, 0, []
    def record(self, m):
        self.ext.extract(m, self.rand, self.ml, self.pay); self.history.append((self.epoch, self.ext.snr))
    def on_train_epoch_end(self, trainer, m):
        self.epoch += 1; self.record(m)

def main(a):
    ck, pay = Path.cwd()/'checkpoints', Path.cwd()/'payload'
    model = initialize_model(a.model, a.dim, a.src_classes, False)
    model.load_state_dict(torch.load(ck/f'{a.model}_{a.src_dataset}_{a.payload.split(".")[0]}_model.pt', map_location=device))
    rand = initialize_model(a.model, a.dim, a.src_classes, False)
    rand.load_state_dict(torch.load(ck/f'{a.model}_{a.src_dataset}_pre_model.pt', map_location=device))
    inj = Injector(seed=42, device=device, payload_path=pay/a.payload, result_path=pay/'extract', logger=log, chunk_factor=6)
    ext = Extractor(seed=42, device=device, result_path=pay/'extract', logger=log, payload_length=len(inj.payload), hash_length=len(inj.hash), chunk_factor=6)
    ml = inj.get_message_length(model)
    model.model.classifier[6] = nn.Linear(model.model.classifier[6].in_features, a.num_classes)
    data = DATA[a.new_dataset](base_path=Path.cwd(), batch_size=a.batch_size, num_workers=a.num_workers)
    t = SNRTracker(ext, rand, ml, a.payload); t.record(model)
    pl.Trainer(max_epochs=a.epochs, accelerator='auto', callbacks=[t]).fit(model, data)
    csv.writer(open('snr_epochs.csv','w',newline='')).writerows([('epoch','snr'), *t.history])
    log.info(f'snr_epochs.csv ({len(t.history)} points)')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model','-m',default='vgg11'); p.add_argument('--src_dataset',default='cifar10')
    p.add_argument('--new_dataset',default='cifar100'); p.add_argument('--dim',type=int,default=32)
    p.add_argument('--src_classes',type=int,default=10); p.add_argument('--num_classes',type=int,default=100)
    p.add_argument('--payload',default='payload.exe'); p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch_size',type=int,default=64); p.add_argument('--num_workers',type=int,default=4)
    main(p.parse_args())
