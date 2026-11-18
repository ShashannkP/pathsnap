# PathSnap classifier tuning and route decoding

These development results use the team’s HistGradientBoosting classifier and
permitted training and student-test partitions. All reported timings are from
the same Windows run. The original subset search remains in the Git history
and its original artifact directory.

## Bayesian search

Eight Optuna TPE trials evaluated five identical GroupKFold splits of 80
training trips and 1,208,139 candidate rows. The objective was mean held-out
binary candidate Macro F1. The search took 632.95 seconds. Trial 7 won with
learning rate 0.044153, 80 iterations, 27 leaves, a minimum leaf size of 35,
L2 regularization 0.173421, and no class weighting.

Both configurations were refitted on 80 trips using fixed iteration counts,
then compared on 137,520 labelled candidate rows from all ten student-test
trips. The existing evaluation policy excluded 97 unaligned candidate rows.
The model was reloaded and checked for score parity and HMM payload formatting.

| Metric | Baseline | Tuned |
|---|---:|---:|
| CV candidate Macro F1 | 0.8436 | 0.8662 |
| Student candidate Macro F1 | 0.8599 | 0.8935 |
| Student observation Top-1 | 81.71% | 82.13% |
| Student covered observation Macro F1 | 0.6661 | 0.6681 |
| Student binary log loss | 0.1827 | 0.1254 |
| Student binary recall at 0.5 | 94.29% | 80.75% |
| Student classifier inference | 0.409 s | 0.342 s |

The tuned model is the candidate scorer for the decoder. Thresholded recall
fell while probability quality and ranking improved. Viterbi uses continuous
scores for all candidates, so the classifier threshold never removes a state.
The pooled Top-1 here and the mean-per-trip accuracy below use different
aggregation and are reported separately.

## HMM/Viterbi

The hidden state is a directed candidate arc at a projected GPS position.
The decoder uses clipped log classifier scores as emissions and penalizes
the difference between the observed GPS displacement and the directed
network distance between projected positions. Viterbi maximizes the summed
scores with backtracking. The search selected a 10 m transition scale from
10, 25, and 50 m on four development trips sampled from the training partition.
The independent student comparison held those choices fixed.

| Mean over 10 student-test trips | Classifier argmax | Classifier + Viterbi |
|---|---:|---:|
| Pointwise accuracy on aligned observations | 83.88% | 89.95% |
| Continuity on eligible consecutive assignments | 91.40% | 99.81% |
| Normalized route edit distance, lower is better | 0.5288 | 0.1810 |
| Decode/route assembly time per trip | 0.120 s | 0.799 s |

All 19,160 observations received a candidate assignment. Accuracy uses
19,131 aligned observations. Pointwise accuracy and sequence edit distance
improved on all ten trips. Trip 00000077 remained the weakest, with 66.68%
pointwise accuracy and 0.6339 route edit distance after decoding. Route breaks
stay explicit instead of drawing a connection across a missing or implausible
transition. The route sequence metric includes directed connector arcs for both
methods and excludes no trip on the basis of accuracy.

See the experiment CSVs and JSON files for complete results. The HMM inference
entry point accepts GPS and road-network inputs only. Developer tests passed
10/10. Evaluator-testing data was never used.
