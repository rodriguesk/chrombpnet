import pyfaidx
import math
import pandas as pd
import numpy as np
import pyBigWig
from modisco.visualization import viz_sequence
import matplotlib.pyplot as plt
import argparse
from context import load_model_wrapper
from context import one_hot
import os
import json
import pyfaidx

def parse_args():
    parser=argparse.ArgumentParser(description="find hyper-parameters for chrombpnet")
    parser.add_argument("-g", "--genome", type=str, required=True, help="Genome fasta")
    parser.add_argument("-b", "--bigwig", type=str, required=True, help="Bigwig of tn5 insertions. Ensure it is +4/-4 shifted")
    parser.add_argument("-p", "--peaks", type=str, required=True, help="10 column bed file of peaks. Sequences and labels will be extracted centered at start (2nd col) + summit (10th col).")
    parser.add_argument("-n", "--nonpeaks", type=str, required=True, help="10 column bed file of non-peak regions, centered at summit (10th column)")
    parser.add_argument("-sr", "--negative-sampling-ratio", type=float, default=1.0, help="Ratio of negative to positive samples per epoch")
    parser.add_argument("-oth", "--outlier_threshold", type=float, default=0.9999, help="outlier threshold to use")
    parser.add_argument("-fl", "--chr_fold_path", type=str, required=True, help="Fold information - dictionary with test,valid and train keys and values with corresponding chromosomes")
    parser.add_argument("-il", "--inputlen", type=int, required=True, help="Sequence input length")
    parser.add_argument("-ol", "--outputlen", type=int, required=True, help="Prediction output length")
    parser.add_argument("-fil", "--filters", type=int, default=128, help="Number of filters to use in chrombpnet mode")
    parser.add_argument("-dil", "--n_dilation_layers", type=int, default=4, help="Number of dilation layers to use in chrombpnet model")
    parser.add_argument("-bmp", "--bias_model_path", type=str, required=True, help="path of bias model")
    parser.add_argument("-o", "--output_dir", help="output dir for storing hyper-param TSV for chrombpnet")
    return parser.parse_args()

def get_seqs_cts(genome, bw, peaks_df, input_width=2114, output_width=1000):
    vals = []
    seqs = []
    for i, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']+r['summit'] - input_width//2):(r['start'] + r['summit'] + input_width//2)])
        if len(sequence) == input_width:
            seqs.append(sequence)
            bigwig_vals=np.nan_to_num(bw.values(r['chr'], 
                                        (r['start'] + r['summit']) - output_width//2,
                                        (r['start'] + r['summit']) + output_width//2))
    return np.sum(np.array(vals),axis=1), one_hot.dna_to_one_hot(seqs)


def adjust_bias_model_logcounts(bias_model, seqs, cts):
    """
    Given a bias model, sequences and associated counts, the function adds a 
    constant to the output of the bias_model's logcounts that minimises squared
    error between predicted logcounts and observed logcounts (infered from 
    cts). This simply reduces to adding the average difference between observed 
    and predicted to the "bias" (constant additive term) of the Dense layer.
    Typically the seqs and counts would correspond to training nonpeak regions.
    ASSUMES model_bias's last layer is a dense layer that outputs logcounts. 
    This would change if you change the model.
    """

    # safeguards to prevent misuse
    assert(bias_model.layers[-1].name == "logcount_predictions")
    assert(bias_model.layers[-1].output_shape==(None,1))
    assert(isinstance(bias_model.layers[-1], keras.layers.Dense))

    print("Predicting within adjust counts")
    _, pred_logcts = bias_model.predict(seqs, verbose=True)

    delta = np.mean(np.log(1+cts.sum(-1)) - pred_logcts.ravel())

    dw, db = bias_model.layers[-1].get_weights()
    bias_model.layers[-1].set_weights([dw, db+delta])

    return bias_model

if __name__=="__main__":

    args = parse_args()

    splits_dict=json.load(open(args.chr_fold_path))
    chroms_to_keep=splits_dict["train"]
    print("evaluating hyperparameters on the following chromosomes",chroms_to_keep)

    bw = pyBigWig.open(args.bigwig) 
    genome = pyfaidx.Fasta(args.genome)


    peaks =  pd.read_csv(args.peaks,
                           sep='\t',
                           names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])
    peaks = peaks[(peaks["chr"].isin(chroms_to_keep))]

    nonpeaks =  pd.read_csv(args.nonpeaks,
                           sep='\t',
                           names=["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"])
    nonpeaks = nonpeaks[(nonpeaks["chr"].isin(chroms_to_keep))]

    peak_cnts, peak_seqs = get_seqs_cts(genome, bw, peaks, args.inputlen, args.outputlen)
    nonpeak_cnts, nonpeak_seqs = get_seqs_cts(genome, bw, nonpeaks, args.inputlen, args.inputlen)

    # find  model hyper-parameters
    if args.negative_sampling_ratio > 0:
        final_cnts = np.concatenate((peak_cnts,np.random.choice(nonpeak_cnts, replace=False, size=(int(args.negative_sampling_ratio*len(peak_cnts))))))
    else:
        final_cnts = peak_cnts

    upper_thresh = np.quantile(final_cnts, 0.9999)
    lower_thresh = np.quantile(final_cnts, 1-0.9999)
    counts_loss_weight = np.median(final_cnts[(final_cnts <= upper_thresh) & (final_cnts>=lower_thresh)])/10

    # adjust bias model
    bias_model = load_model_wrapper(args.bias_model_path)
    bias_model_scaled = adjust_bias_model_logcounts(bias_model, nonpeak_seqs, nonpeak_cnts)
    bias_model_scaled.save(os.path.join(args.output_dir, "bias_model_scaled.h5"))
    # save the new bias model


    file = open(os.path.join(args.out_dir, "chrombpnet_params.txt"),"w")
    file.write("\t".join(["counts_sum_min_thresh", str(round(lower_thresh,2))]))
    file.write("\n")
    file.write("\t".join(["counts_sum_max_thresh", str(round(upper_thresh,2))]))
    file.write("\n")
    file.write("\t".join(["counts_loss_weight", str(round(counts_loss_weight,2))]))
    file.write("\n")
    file.write("\t".join(["filters", str(args.filters)]))
    file.write("\n")
    file.write("\t".join(["n_dil_layers", str(args.n_dilation_layers)]))
    file.write("\n")
    file.write("\t".join(["trainings_pts_post_thresh", str(sum((final_cnts< upper_thresh) & (final_cnts>lower_thresh)))]))
    file.write("\n")
    file.write("\t".join(["bias_model_path", os.path.join(args.output_dir, "bias_model_scaled.h5")]))
    file.write("\n")
    file.write("\t".join(["inputlen", str(args.inputlen)]))
    file.write("\n")
    file.write("\t".join(["outputlen", str(args.outputlen)]))
    file.close()