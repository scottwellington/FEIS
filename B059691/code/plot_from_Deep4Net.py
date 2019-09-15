import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import _pickle as pkl

#########################################PLOTTING FUNCTIONS########################################

def	plot_accuracies_over_time(accs):

	matplotlib.style.use('seaborn')
	accuracies_over_time = plt.figure(figsize=(8,3))
	plt.plot(accs * 100)
	plt.title("Accuracies per timestep", fontsize=16)
	plt.xlabel('Timestep in trial', fontsize=14)
	plt.ylabel('Accuracy [%]', fontsize=14)
	
def	plot_prediction_probability_across_a_trial(outs, t_labs, i_trial = 0):

	# log-softmax outputs need to be exponentiated to get probabilities
	cropped_probs = np.exp(outs[i_trial])
	prediction_probability_across_a_trial = plt.figure(figsize=(8,3))
	plt.plot(cropped_probs.T)
	plt.title("Network probabilities for trial {:d} of class {:d}".format(
		i_trial, t_labs[i_trial]), fontsize=16)
	plt.legend(("Class 0", "Class 1"), fontsize=12)
	plt.xlabel("Timestep within trial", fontsize=14)
	plt.ylabel("Probabilities", fontsize=14)

def plot_with_mne(freq, pos):

	import mne

	max_abs_val = np.max(np.abs(freq))
	
	fig, axes = plt.subplots(1, 2)
	class_names = [0, 1]
	for i_class in range(2):
		ax = axes[i_class]
		mne.viz.plot_topomap(freq[:,i_class], pos,
						 vmin=-max_abs_val, vmax=max_abs_val, contours=0,
						cmap=cm.coolwarm, axes=ax, show=False);
		ax.set_title(class_names[i_class])

def plot_with_braindecode(freq, ch):

	from braindecode.datasets.sensor_positions import CHANNEL_10_20_APPROX
	from braindecode.visualization.plot import ax_scalp

	max_abs_val = np.max(np.abs(freq))

	fig, axes = plt.subplots(1, 2)
	class_names = [0, 1]
	for i_class in range(2):
		ax = axes[i_class]
		ax_scalp(freq[:,i_class], ch, chan_pos_list=CHANNEL_10_20_APPROX, cmap=cm.coolwarm,
				vmin=-max_abs_val, vmax=max_abs_val, ax=ax)
		ax.set_title(class_names[i_class])

filename = 'Deep4Net.pkl'

d = pkl.load(open(filename, 'rb'))	
x = d['plot_data']

print(d)
exit(i)

# Visualise accuracies over time:
plot_accuracies_over_time(x['accs_per_crop'])
plt.savefig('plots/plot_accuracies_over_time.pdf', bbox_inches='tight')
plt.close()

# Raw outputs to visualize a prediction probability across a trial:
for i in range(0,len(d['predict_classes'])):
	plot_prediction_probability_across_a_trial(x['cropped_outs'], x['test_set_labels'], i_trial = i)
	plt.savefig('plots/plot_prediction_probability_across_trial_{}.pdf'.format(i), bbox_inches='tight')
	plt.close()

# Scalp plots:
for i in x['freq_corr']:
	plot_with_mne(x['freq_corr'][i], x['positions_kara'])
	plt.savefig('plots/plot_scalp_{}_mne.pdf'.format(i), bbox_inches='tight')
	plot_with_braindecode(x['freq_corr'][i], x['ch_names_kara'])
	plt.savefig('plots/plot_scalp_{}_braindecode.pdf'.format(i), bbox_inches='tight')
	plt.close('all')