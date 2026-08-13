import math
import time
import hashlib
import numpy as np

import multiprocessing as mp
from functools import partial

from tqdm import tqdm

from pathlib import Path
from utils.utils_bit import bits_to_file, bits_from_bytes

from pyldpc import make_ldpc, decode, get_message


_func = None


def worker_init(func):
  global _func
  _func = func


def worker(x):
  return _func(x)


class Extractor:
    BIT_TO_SIGNAL_MAPPING = {
        1: -1,
        0: 1
    }
    CHUNK_SIZE = 4096

    def __init__(self, seed: int, device: str, result_path: Path, logger, payload_length: int, hash_length: int, chunk_factor: int):
        self.seed = seed
        self.device = device
        self.result_path = result_path
        self.logger = logger
        self.H = None
        self.G = None
        self.preamble = None
        self.payload_length = payload_length
        self.hash_length = hash_length
        self.chunk_factor = chunk_factor
        if self.payload_length > 4000:
            k = 3048
        else:
            k = 96
        d_v = 3
        d_c = 12
        n = k * int(d_c / d_v)
        self.H, self.G = make_ldpc(
            n, d_v, d_c, systematic=True, sparse=True, seed=seed)

    def extract(self, model, rand_model, message_length, payload_name, carriers=None):
        extraction_path = self.result_path
        extraction_path.mkdir(parents=True, exist_ok=True)

        start = time.time()
        st_dict_prev = rand_model.state_dict()
        st_dict_next = model.state_dict()

        models_w_prev = []
        models_w_curr = []

        layer_lengths = dict()
        total_params = 0

        layers = [n for n in st_dict_prev.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x_prev = st_dict_prev[layer].detach().cpu().numpy().flatten()
            models_w_prev.extend(list(x_prev))
            x_curr = st_dict_next[layer].detach().cpu().numpy().flatten()
            models_w_curr.extend(list(x_curr))
            layer_lengths[layer] = len(x_prev)
            total_params += len(x_prev)

        models_w_prev = np.array(models_w_prev)
        models_w_curr = np.array(models_w_curr)
        models_w_delta = np.subtract(models_w_curr, models_w_prev)

        number_of_chunks = math.ceil(message_length / self.CHUNK_SIZE)
        if self.CHUNK_SIZE * self.chunk_factor * number_of_chunks > len(models_w_prev):
            self.logger.critical(
                f'Spreading codes cannot be bigger than the model!')
            return

        #------- hessian carriers -------
        n_carriers = self.CHUNK_SIZE * self.chunk_factor * number_of_chunks
        if carriers is not None and len(carriers) >= n_carriers:
            filter_indexes = [int(i) for i in carriers[:n_carriers]]
        else:
            np.random.seed(self.seed)
            filter_indexes = np.random.randint(
                0, len(models_w_prev), n_carriers, np.int32).tolist()
        #------- /hessian carriers -------

        x = []
        ys = []

        with tqdm(total=message_length) as bar:
            bar.set_description('Extracting')
            current_chunk = 0
            current_bit = 0
            np.random.seed(self.seed)
            for _ in range(message_length):
                spreading_code = np.random.choice(
                    [-1, 1], size=self.CHUNK_SIZE * self.chunk_factor)
                current_filter_index = filter_indexes[current_chunk * self.CHUNK_SIZE * self.chunk_factor:
                                                      (current_chunk + 1) * self.CHUNK_SIZE * self.chunk_factor]
                current_models_w_delta = models_w_delta[current_filter_index]
                y_i = np.matmul(spreading_code.T, current_models_w_delta)
                ys.append(y_i)

                current_bit += 1
                if current_bit > self.CHUNK_SIZE * (current_chunk + 1):
                    current_chunk += 1

                bar.update(1)

        y = np.array(ys)

        np.random.seed(self.seed * 15)
        preamble = np.sign(np.random.uniform(-1, 1, 200))

        gain = np.mean(np.multiply(y[:200], preamble))
        sigma = np.std(np.multiply(y[:200], preamble) / gain)
        snr = -20 * np.log10(sigma)
        self.logger.info(f'Signal to Noise Ratio = {snr}')

        k = self.G.shape[0]
        y = y[200:]
        n_chunks = int(len(y) / k)
        chunks = list()

        for ch in range(n_chunks):
            chunks.append(y[ch * k:ch * k + k] / gain)


        # Serial instead of mp.Pool so it dosnt die of OOM
        # decoded = [get_message(self.G, decode(self.H, chunk, snr))
        #            for chunk in tqdm(chunks, desc='Decoding')]

        # Extracting with 6 workers
        d = (decode(self.H, ch, snr) for ch in chunks)
        with mp.get_context("fork").Pool(6, initializer=worker_init, initargs=(partial(get_message, self.G),)) as pool:
            decoded = list(tqdm(pool.imap(worker, d), total=len(chunks), desc='Decoding'))

        for dec in decoded:
            x.extend(dec)

        end = time.time()
        self.logger.info(f'Time to extract {end - start}')

        bits_to_file(extraction_path / f'{payload_name}.no_execute',
                     x[:self.payload_length])

        str_payload = ''.join(str(l) for l in x[:self.payload_length])
        str_hash = ''.join(
            str(l) for l in x[self.payload_length:self.payload_length+self.hash_length])
        hash_str = hashlib.sha256(
            ''.join(str(l) for l in str_payload).encode('utf-8')).hexdigest()
        hash_bits = ''.join(str(l) for l in (bits_from_bytes(
            [char for char in hash_str.encode('utf-8')])))
        self.logger.info(
            f'Original payload hash {str_hash}\nExtracted payload hash {hash_bits}')

        return str_hash == hash_bits
