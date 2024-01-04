import tensorflow as tf
import tensorflow_probability as tfp


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    #this should be based on average coverage
    counts_per_example = 50 * 1000
    #true_counts are actually probabilities of methylation and not counts, but they need to be converted to decimal format
    true_counts = true_counts / 100 
    
    #counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    
    #to get to probability from logit
    odds = exp(logits)
    prob = odds / (1 + odds)

    #then model each as collection of beta distributions and find their cross entropy
    dist_true = tfp.distributions.Beta(a=,b=)

    dist_pred = 


    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)

    #compute negatile log-likelihood
    nll_numerator = -tf.reduce_sum(dist.log_prob(true_counts)) 
    nll_denominator = tf.cast(tf.shape(true_counts)[0], dtype=tf.float32)
    #should check the shape of this nll_denominator to confirm it's the correct shape!
    nll = (nll_numerator / nll_denominator)

    # return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
    #         tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))
    return nll



