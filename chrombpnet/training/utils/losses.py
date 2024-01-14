import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_percentages, pred_percentages):
    """Compute the beta negative log-likelihood
    Args:
      true_percentages: observed percent methylation values
      pred_percentages: predicted percent methylation values
    """

	avg_coverage=25

	pre_alpha = np.round(avg_coverage*np.array(true_percentages))
	pre_beta = np.subtract(avg_coverage, pre_alpha)

	pre_alpha = np.add(pre_alpha, 0.00001)
	pre_beta = np.add(pre_beta, 0.00001)

	alpha_true = pre_alpha.tolist()
	beta_true = pre_beta.tolist()

	pred_percentages = np.array(pred_percentages)

	#fix percentages that are 0 or 1 to a little more than 0 or a little less than 1.
	epsilon = 0.0000001
	zero_indices = pred_percentages == 0
	one_indices = pred_percentages == 1
	# Add epsilon to elements that are exactly 0
	pred_percentages[zero_indices] += epsilon
	# Subtract epsilon from elements that are exactly 1
	pred_percentages[one_indices] -= epsilon


	dists = tfp.distributions.Beta(alpha_true, beta_true)

	log_probs=dists.log_prob(pred_percentages)
	neg_sum_log_probs=-tf.reduce_sum(log_probs)
	average_NLL=neg_sum_log_probs/tf.cast(tf.shape(true_percentages)[0], dtype=tf.float32)

	return(average_NLL)




