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
	for i in range(1):
		hist = load_obj('history_{}'.format(i))
		plt.plot(range(len(hist['acc'])), hist['acc'])
		plt.plot(range(len(hist['val_acc'])), hist['val_acc'], linestyle='--')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
	plt.show()

plot_accuracies()