import matplotlib.pyplot as plt
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot_accuracies():
	cols = ['y', 'b', 'c', 'g', 'k', 'r', 'm']
	name = [r'CO', r'$\mathrm{CO_2}$', r'$\mathrm{CH_4}$', r'$\mathrm{Na}$', r'$\mathrm{K}$', r'$\mathrm{NH_3}$', r'$\mathrm{PH_3}$']
	for i in range(7):
		hist = load_obj('history_exp1_categorize_{}'.format(i))
		print(name[i], np.mean(hist['val_acc'][50:]))
	# 	plt.plot(range(len(hist['acc'])), hist['acc'], color=cols[i], label=name[i])
	# 	plt.plot(range(len(hist['val_acc'])), hist['val_acc'], linestyle='--', color=cols[i], alpha=0.5, linewidth=1)
	# plt.legend()
	# plt.xlabel('Epochs', fontsize=20)
	# plt.ylabel('Accuracy', fontsize=20)
	# plt.xticks(fontsize=15)
	# plt.yticks(fontsize=15)
	# plt.tight_layout()
	# plt.savefig('exp3_categorize.pdf')
	# plt.show()

plot_accuracies()