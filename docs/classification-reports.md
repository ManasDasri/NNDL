# Classification reports

One report per training run, on the 76 held-out test contracts, at the
per-label thresholds tuned on validation.

From now on every run writes its own: `legal-risk-evaluate` saves
`classification_report_<split>.txt` next to the run, with a window-level and a
contract-level report, each in scikit-learn's format (per-class precision,
recall, F1 and support, then micro, macro, weighted and samples averages). The
notebooks already call it for every model.

## Kaggle runs, 15 November 2026

Notebooks 3 and 4 run on Kaggle T4s with every fix in
[weak-points.md](weak-points.md). These reports are exactly as each run printed
them, window level then contract level. The executed notebooks, with their
confusion-matrix and metric charts, are saved in [`results/`](../results).
BERT and Legal-BERT each have seeds 42, 1 and 2 (`_s1`, `_s2`).

### `outputs/cnn`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
textcnn · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.784     0.731     0.756       104
                Non-Compete      0.516     0.281     0.364        57
              License Grant      0.700     0.815     0.753       146
               Audit Rights      0.547     0.874     0.673        87
Termination for Convenience      0.462     0.480     0.471        50
                  Insurance      0.797     0.843     0.819        70

                  micro avg      0.657     0.720     0.687       514
                  macro avg      0.634     0.670     0.639       514
               weighted avg      0.661     0.720     0.678       514
                samples avg      0.109     0.106     0.107       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.933     0.955     0.944        44
                Non-Compete      0.700     0.667     0.683        21
              License Grant      0.935     1.000     0.966        43
               Audit Rights      0.825     0.971     0.892        34
Termination for Convenience      0.531     0.586     0.557        29
                  Insurance      0.931     0.931     0.931        29

                  micro avg      0.830     0.880     0.854       200
                  macro avg      0.809     0.852     0.829       200
               weighted avg      0.832     0.880     0.855       200
                samples avg      0.716     0.737     0.703       200
```

### `outputs/bert`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.635     0.837     0.722       104
                Non-Compete      0.486     0.596     0.535        57
              License Grant      0.524     0.911     0.665       146
               Audit Rights      0.630     0.862     0.728        87
Termination for Convenience      0.541     0.800     0.645        50
                  Insurance      0.831     0.914     0.871        70

                  micro avg      0.592     0.842     0.696       514
                  macro avg      0.608     0.820     0.694       514
               weighted avg      0.604     0.842     0.699       514
                samples avg      0.126     0.124     0.124       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.860     0.977     0.915        44
                Non-Compete      0.818     0.857     0.837        21
              License Grant      0.796     1.000     0.887        43
               Audit Rights      0.872     1.000     0.932        34
Termination for Convenience      0.658     0.862     0.746        29
                  Insurance      0.906     1.000     0.951        29

                  micro avg      0.817     0.960     0.883       200
                  macro avg      0.818     0.949     0.878       200
               weighted avg      0.821     0.960     0.884       200
                samples avg      0.708     0.799     0.736       200
```

### `outputs/legal_bert`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
legal-bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.796     0.827     0.811       104
                Non-Compete      0.600     0.474     0.529        57
              License Grant      0.795     0.795     0.795       146
               Audit Rights      0.629     0.897     0.739        87
Termination for Convenience      0.773     0.680     0.723        50
                  Insurance      0.795     0.886     0.838        70

                  micro avg      0.739     0.784     0.761       514
                  macro avg      0.731     0.760     0.739       514
               weighted avg      0.743     0.784     0.758       514
                samples avg      0.119     0.116     0.117       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.957     1.000     0.978        44
                Non-Compete      0.594     0.905     0.717        21
              License Grant      0.878     1.000     0.935        43
               Audit Rights      0.850     1.000     0.919        34
Termination for Convenience      0.719     0.793     0.754        29
                  Insurance      0.933     0.966     0.949        29

                  micro avg      0.834     0.955     0.890       200
                  macro avg      0.822     0.944     0.875       200
               weighted avg      0.846     0.955     0.895       200
                samples avg      0.722     0.796     0.743       200
```

### `outputs/bert_s1`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.711     0.827     0.764       104
                Non-Compete      0.659     0.509     0.574        57
              License Grant      0.755     0.801     0.777       146
               Audit Rights      0.631     0.885     0.737        87
Termination for Convenience      0.424     0.840     0.564        50
                  Insurance      0.847     0.871     0.859        70

                  micro avg      0.672     0.802     0.731       514
                  macro avg      0.671     0.789     0.713       514
               weighted avg      0.695     0.802     0.736       514
                samples avg      0.118     0.118     0.117       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.952     0.909     0.930        44
                Non-Compete      0.714     0.952     0.816        21
              License Grant      0.782     1.000     0.878        43
               Audit Rights      0.868     0.971     0.917        34
Termination for Convenience      0.540     0.931     0.684        29
                  Insurance      0.933     0.966     0.949        29

                  micro avg      0.786     0.955     0.862       200
                  macro avg      0.798     0.955     0.862       200
               weighted avg      0.814     0.955     0.872       200
                samples avg      0.678     0.793     0.718       200
```

### `outputs/legal_bert_s1`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
legal-bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.728     0.875     0.795       104
                Non-Compete      0.521     0.667     0.585        57
              License Grant      0.848     0.767     0.806       146
               Audit Rights      0.702     0.839     0.764        87
Termination for Convenience      0.551     0.860     0.672        50
                  Insurance      0.851     0.900     0.875        70

                  micro avg      0.717     0.817     0.764       514
                  macro avg      0.700     0.818     0.749       514
               weighted avg      0.734     0.817     0.768       514
                samples avg      0.123     0.120     0.121       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.978     1.000     0.989        44
                Non-Compete      0.731     0.905     0.809        21
              License Grant      0.915     1.000     0.956        43
               Audit Rights      0.895     1.000     0.944        34
Termination for Convenience      0.574     0.931     0.711        29
                  Insurance      0.964     0.931     0.947        29

                  micro avg      0.840     0.970     0.900       200
                  macro avg      0.843     0.961     0.893       200
               weighted avg      0.864     0.970     0.909       200
                samples avg      0.728     0.820     0.758       200
```

### `outputs/bert_s2`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.755     0.712     0.733       104
                Non-Compete      0.462     0.632     0.533        57
              License Grant      0.760     0.801     0.780       146
               Audit Rights      0.564     0.862     0.682        87
Termination for Convenience      0.424     0.780     0.549        50
                  Insurance      0.785     0.886     0.832        70

                  micro avg      0.636     0.784     0.702       514
                  macro avg      0.625     0.779     0.685       514
               weighted avg      0.663     0.784     0.711       514
                samples avg      0.119     0.116     0.117       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.698     1.000     0.822        44
                Non-Compete      0.690     0.952     0.800        21
              License Grant      0.896     1.000     0.945        43
               Audit Rights      0.868     0.971     0.917        34
Termination for Convenience      0.529     0.931     0.675        29
                  Insurance      0.848     0.966     0.903        29

                  micro avg      0.744     0.975     0.844       200
                  macro avg      0.755     0.970     0.844       200
               weighted avg      0.766     0.975     0.853       200
                samples avg      0.649     0.820     0.709       200
```

### `outputs/legal_bert_s2`

[`results/3_problem1_legal_bert_kaggle.ipynb`](../results/3_problem1_legal_bert_kaggle.ipynb)

```text
legal-bert · test split

Window level (3314 windows of 300 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.853     0.837     0.845       104
                Non-Compete      0.554     0.719     0.626        57
              License Grant      0.793     0.788     0.790       146
               Audit Rights      0.563     0.920     0.699        87
Termination for Convenience      0.575     0.840     0.683        50
                  Insurance      0.800     0.857     0.828        70

                  micro avg      0.696     0.827     0.756       514
                  macro avg      0.690     0.827     0.745       514
               weighted avg      0.720     0.827     0.762       514
                samples avg      0.124     0.122     0.122       514
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.936     1.000     0.967        44
                Non-Compete      0.690     0.952     0.800        21
              License Grant      1.000     0.977     0.988        43
               Audit Rights      0.791     1.000     0.883        34
Termination for Convenience      0.574     0.931     0.711        29
                  Insurance      0.931     0.931     0.931        29

                  micro avg      0.819     0.970     0.888       200
                  macro avg      0.820     0.965     0.880       200
               weighted avg      0.846     0.970     0.897       200
                samples avg      0.708     0.815     0.743       200
```

### `outputs/longformer`

[`results/4_problem2_longformer_kaggle.ipynb`](../results/4_problem2_longformer_kaggle.ipynb)

```text
longformer · test split

Window level (326 windows of 2600 words, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.820     0.806     0.813        62
                Non-Compete      0.315     0.548     0.400        31
              License Grant      0.865     0.901     0.883        71
               Audit Rights      0.794     0.926     0.855        54
Termination for Convenience      0.406     0.788     0.536        33
                  Insurance      0.950     0.905     0.927        42

                  micro avg      0.688     0.836     0.755       293
                  macro avg      0.692     0.812     0.736       293
               weighted avg      0.745     0.836     0.779       293
                samples avg      0.446     0.489     0.449       293
Samples avg scores each window over its own labels, so the many windows with no
clause count as zero here. Read the per-class rows and the macro average instead.

Contract level (76 contracts, noisy_or pooling, validation-tuned thresholds)

                             precision    recall  f1-score   support

           Cap on Liability      0.875     0.955     0.913        44
                Non-Compete      0.542     0.619     0.578        21
              License Grant      0.913     0.977     0.944        43
               Audit Rights      0.895     1.000     0.944        34
Termination for Convenience      0.579     0.759     0.657        29
                  Insurance      0.903     0.966     0.933        29

                  micro avg      0.804     0.905     0.852       200
                  macro avg      0.784     0.879     0.828       200
               weighted avg      0.813     0.905     0.856       200
                samples avg      0.682     0.745     0.695       200
```


## Runs before reports were automatic

These runs' output folders never left Colab, so their reports are rebuilt from
the per-class precision, recall, F1, average precision and support each notebook
printed. Two consequences:

- **Window level only.** The notebooks printed contract-level macro scores but
  not per-class ones, so contract-level reports are not recoverable.
- **Micro and weighted averages are derived, not printed.** True positives are
  recall × support and predictions are true positives ÷ precision, from values
  rounded to three decimals, so the last digit can be off by one. The samples
  average needs per-window predictions and is omitted. Macro F1 matches each
  run's printed score exactly, which checks the reconstruction.

Longformer's rows are on 2,600-word windows (293 positive labels across 326
test windows) and are not comparable with the 300-word models' rows (514 across
3,314). Compare models at contract level, in the [results](../handbook/12-results.md).

### TextCNN, seed 42

notebook 3, commit `73b6b1b`, `outputs/cnn`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.812     0.750     0.780       104      0.803
                Non-Compete      0.364     0.281     0.317        57      0.285
              License Grant      0.773     0.747     0.760       146      0.840
               Audit Rights      0.607     0.816     0.696        87      0.772
Termination for Convenience      0.528     0.560     0.544        50      0.566
                  Insurance      0.813     0.871     0.841        70      0.861

                  micro avg      0.690     0.706     0.698       514
                  macro avg      0.650     0.671     0.656       514      0.688
               weighted avg      0.689     0.706     0.694       514      0.736
```

### BERT, seed 42

notebook 3, commit `73b6b1b`, `outputs/bert`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.731     0.837     0.780       104      0.842
                Non-Compete      0.651     0.491     0.560        57      0.475
              License Grant      0.687     0.856     0.762       146      0.816
               Audit Rights      0.639     0.874     0.738        87      0.812
Termination for Convenience      0.545     0.720     0.621        50      0.573
                  Insurance      0.896     0.857     0.876        70      0.896

                  micro avg      0.691     0.802     0.742       514
                  macro avg      0.692     0.772     0.723       514      0.736
               weighted avg      0.698     0.802     0.741       514      0.770
```

### Legal-BERT, seed 42

notebook 3, commit `73b6b1b`, `outputs/legal_bert`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.720     0.817     0.766       104      0.868
                Non-Compete      0.551     0.474     0.509        57      0.470
              License Grant      0.680     0.842     0.752       146      0.862
               Audit Rights      0.625     0.862     0.725        87      0.817
Termination for Convenience      0.623     0.760     0.685        50      0.711
                  Insurance      0.871     0.871     0.871        70      0.885

                  micro avg      0.683     0.795     0.735       514
                  macro avg      0.678     0.771     0.718       514      0.769
               weighted avg      0.685     0.795     0.733       514      0.801
```

### BERT, seed 1

notebook 3, commit `73b6b1b`, `outputs/bert_s1`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.731     0.837     0.780       104      0.865
                Non-Compete      0.595     0.386     0.468        57      0.477
              License Grant      0.842     0.801     0.821       146      0.861
               Audit Rights      0.716     0.839     0.772        87      0.759
Termination for Convenience      0.512     0.820     0.631        50      0.668
                  Insurance      0.938     0.871     0.904        70      0.909

                  micro avg      0.740     0.780     0.759       514
                  macro avg      0.722     0.759     0.729       514      0.756
               weighted avg      0.752     0.780     0.758       514      0.790
```

### Longformer, seed 42

notebook 4, first run (3 epochs, undertrained), commit `48977b4`, `outputs/longformer`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.645     0.790     0.710        62      0.768
                Non-Compete      0.280     0.452     0.346        31      0.266
              License Grant      0.656     0.887     0.754        71      0.891
               Audit Rights      0.312     0.722     0.436        54      0.388
Termination for Convenience      0.194     0.212     0.203        33      0.256
                  Insurance      0.943     0.786     0.857        42      0.901

                  micro avg      0.490     0.700     0.577       293
                  macro avg      0.505     0.642     0.551       293      0.578
               weighted avg      0.540     0.700     0.596       293      0.636
```

### Longformer, seed 42

notebook 4, final run (10 epochs), commit `6124e1c`, `outputs/longformer`

```text
                             precision    recall  f1-score   support   avg-prec

           Cap on Liability      0.849     0.726     0.783        62      0.837
                Non-Compete      0.528     0.613     0.567        31      0.564
              License Grant      0.901     0.901     0.901        71      0.926
               Audit Rights      0.758     0.926     0.833        54      0.844
Termination for Convenience      0.263     0.455     0.333        33      0.327
                  Insurance      0.860     0.881     0.871        42      0.901

                  micro avg      0.705     0.785     0.743       293
                  macro avg      0.693     0.750     0.715       293      0.733
               weighted avg      0.746     0.785     0.760       293      0.783
```

