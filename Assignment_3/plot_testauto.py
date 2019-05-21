import numpy as np
import matplotlib.pyplot as plt

accuracy_noise = np.load('accuracy_noise.npy')
accuracy_denoise_low = np.load('accuracy_denoise_005.npy')
accuracy_denoise_high = np.load('accuracy_denoise_015.npy')

for i in range(7):
	plt.plot(np.linspace(0,0.25,100), accuracy_noise[:,i], label='noise')
	plt.plot(np.linspace(0,0.25,100), accuracy_denoise_low[:,i], label='denoise 0.05')
	plt.plot(np.linspace(0,0.25,100), accuracy_denoise_high[:,i], label='denoise 0.15')

	plt.xlabel(r'$\sigma$ in $\mathcal{{G}}(0, \sigma)$', fontsize=20)
	plt.ylabel('Accuracy', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.legend(framealpha=0)

	plt.tight_layout()

	plt.savefig('denoised_{}.png'.format(i))
	plt.show()