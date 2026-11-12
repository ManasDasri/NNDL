# Classification reports

One report per training run, on the 76 held-out test contracts, at the
per-label thresholds tuned on validation.

From now on every run writes its own: `legal-risk-evaluate` saves
`classification_report_<split>.txt` next to the run, with a window-level and a
contract-level report, each in scikit-learn's format (per-class precision,
recall, F1 and support, then micro, macro, weighted and samples averages). The
notebooks already call it for every model.

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

