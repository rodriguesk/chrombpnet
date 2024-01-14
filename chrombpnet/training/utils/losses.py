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

    pre_alpha = tf.round(tf.math.multiply(true_percentages,avg_coverage))
    pre_beta = tf.subtract(avg_coverage, pre_alpha)

	alpha_true = tf.add(pre_alpha, 0.00001)
	beta_true = tf.add(pre_beta, 0.00001)

	#fix percentages that are 0 or 1 to a little more than 0 or a little less than 1.
	epsilon = 0.0000001

    zero_mask = tf.equal(pred_percentages, 0.0)
    one_mask = tf.equal(pred_percentages, 1.0)

    # Add epsilon to elements that are exactly 0
    pred_percentages = tf.where(zero_mask, pred_percentages + epsilon, pred_percentages)

    # Subtract epsilon from elements that are exactly 1
    pred_percentages = tf.where(one_mask, pred_percentages - epsilon, pred_percentages)

	dists = tfp.distributions.Beta(alpha_true, beta_true)

	log_probs=dists.log_prob(pred_percentages)
	neg_sum_log_probs=-tf.reduce_sum(log_probs)
	average_NLL=neg_sum_log_probs/tf.cast(tf.shape(true_percentages)[0], dtype=tf.float32)

	return(average_NLL)




