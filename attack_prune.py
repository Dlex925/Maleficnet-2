import argparse, copy, logging
from pathlib import Path
import torch, torch.nn as nn, torch.nn.utils.prune as tprune
from maleficnet import initialize_model, device
from injector import Injector
from extractor import Extractor

log = logging.getLogger(); log.addHandler(logging.StreamHandler())

def main(a):
    ck, pay = Path.cwd()/'checkpoints', Path.cwd()/'payload'
    model = initialize_model(a.model, a.dim, a.num_classes, a.only_pretrained)
    model.load_state_dict(torch.load(ck/f'{a.model}_{a.dataset}_{a.payload.split(".")[0]}_model.pt', map_location=device))
    rand = initialize_model(a.model, a.dim, a.num_classes, a.only_pretrained)
    rand.load_state_dict(torch.load(ck/f'{a.model}_{a.dataset}_pre_model.pt', map_location=device))
    inj = Injector(seed=42, device=device, payload_path=pay/a.payload, result_path=pay/'extract', logger=log, chunk_factor=6)
    ext = Extractor(seed=42, device=device, result_path=pay/'extract', logger=log, payload_length=len(inj.payload), hash_length=len(inj.hash), chunk_factor=6)
    ml, clean = inj.get_message_length(model), copy.deepcopy(model.state_dict())
    for amt in (0.25, 0.5, 0.75, 0.9, 0.99):
        model.load_state_dict(clean)
        ps = [(m,'weight') for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
        tprune.global_unstructured(ps, pruning_method=tprune.L1Unstructured, amount=amt)
        for m,_ in ps: tprune.remove(m,'weight')
        log.info(f'Pruned {amt:.0%}')
        log.info('System successfully!' if ext.extract(model, rand, ml, a.payload) else 'System unsuccessfully :(')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model','-m',default='vgg16'); p.add_argument('--dataset',default='imagenet')
    p.add_argument('--dim',type=int,default=224); p.add_argument('--num_classes',type=int,default=1000)
    p.add_argument('--payload',default='payload.exe'); p.add_argument('--only_pretrained',action='store_true')
    main(p.parse_args())
