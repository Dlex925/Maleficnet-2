import csv, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
r = list(csv.reader(open('snr_epochs.csv')))[1:]
plt.plot([int(x[0]) for x in r], [float(x[1]) for x in r], marker='o')
plt.xlabel('Epochs'); plt.ylabel('Signal to Noise Ratio'); plt.grid(True); plt.savefig('snr_epochs.png', dpi=120)
