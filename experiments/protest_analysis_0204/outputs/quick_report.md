# Quick runtime-stochasticity report

## Case=case2746wop, Perturbation=(0.02, 20)

### Highest CV (std/mean)

| Formulation   | Merge   |   A_parameter |   n |     mean |      std |       cv |   tail_ratio |
|:--------------|:--------|--------------:|----:|---------:|---------:|---------:|-------------:|
| Chordal_MD    | True    |             2 |  20 | 101.556  | 20.9067  | 0.205863 |      1.44271 |
| Chordal_MD    | True    |             3 |  20 |  45.2088 |  7.66546 | 0.169557 |      1.37315 |
| Chordal_MD    | False   |             0 |  20 |  60.3421 |  9.89478 | 0.163978 |      1.26331 |
| Chordal_MD    | True    |             4 |  20 |  57.7042 |  9.15856 | 0.158716 |      1.17601 |
| Chordal_MD    | True    |             5 |  20 |  60.5524 |  9.53571 | 0.157479 |      1.17067 |
| Chordal_MFI   | False   |             0 |  20 |  68.2579 | 10.6182  | 0.15556  |      1.16313 |
| Chordal_AMD   | False   |             0 |  20 |  65.3561 |  9.51418 | 0.145574 |      1.2235  |
| Chordal_MFI   | True    |             4 |  20 |  55.4111 |  7.89542 | 0.142488 |      1.22097 |
| Chordal_MFI   | True    |             3 |  20 |  50.8025 |  7.2125  | 0.141971 |      1.19484 |
| Chordal_MFI   | True    |             5 |  20 |  57.4959 |  8.1264  | 0.141339 |      1.22149 |


### Highest tail ratio (max/median)

| Formulation   | Merge   |   A_parameter |   n |   median |      max |   tail_ratio |       cv |
|:--------------|:--------|--------------:|----:|---------:|---------:|-------------:|---------:|
| Chordal_MD    | True    |             2 |  20 |  88.3283 | 127.432  |      1.44271 | 0.205863 |
| Chordal_MD    | True    |             3 |  20 |  40.5486 |  55.6792 |      1.37315 | 0.169557 |
| Chordal_MD    | False   |             0 |  20 |  57.5933 |  72.7582 |      1.26331 | 0.163978 |
| Chordal_AMD   | True    |             3 |  20 |  47.6361 |  58.3769 |      1.22548 | 0.132483 |
| Chordal_AMD   | False   |             0 |  20 |  63.5853 |  77.7965 |      1.2235  | 0.145574 |
| Chordal_MFI   | True    |             5 |  20 |  55.8027 |  68.1623 |      1.22149 | 0.141339 |
| Chordal_MFI   | True    |             4 |  20 |  53.8495 |  65.7487 |      1.22097 | 0.142488 |
| Chordal_MFI   | True    |             3 |  20 |  50.2622 |  60.0551 |      1.19484 | 0.141971 |
| Chordal_MD    | True    |             4 |  20 |  58.1818 |  68.4223 |      1.17601 | 0.158716 |
| Chordal_MD    | True    |             5 |  20 |  60.9135 |  71.3097 |      1.17067 | 0.157479 |


### Most rank-unstable strategies (rank_unique_count)

| Formulation   | Merge   |   A_parameter |   rank_unique_count | rank_swaps   |
|:--------------|:--------|--------------:|--------------------:|:-------------|
| Chordal_AMD   | True    |             3 |                   7 | True         |
| Chordal_MD    | False   |             0 |                   7 | True         |
| Chordal_AMD   | True    |             4 |                   6 | True         |
| Chordal_MD    | True    |             3 |                   6 | True         |
| Chordal_AMD   | True    |             5 |                   5 | True         |
| Chordal_MD    | True    |             4 |                   5 | True         |
| Chordal_MFI   | True    |             5 |                   5 | True         |
| Chordal_MFI   | True    |             3 |                   5 | True         |
| Chordal_MFI   | True    |             4 |                   4 | True         |
| Chordal_MD    | True    |             2 |                   4 | True         |


### Pairwise win-probability extremes

**Most decisive (A much faster than B):**

| FormA       | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:------------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MFI | True     |    4 | Chordal_MFI | True     |    5 |       1 |                    1 |
| Chordal_AMD | False    |    0 | Chordal_AMD | True     |    2 |       1 |                    1 |
| Chordal_MFI | True     |    3 | Chordal_MFI | True     |    5 |       1 |                    1 |
| Chordal_MFI | True     |    3 | Chordal_MFI | True     |    4 |       1 |                    1 |
| Chordal_MFI | False    |    0 | Chordal_MFI | True     |    2 |       1 |                    1 |


**Most uncertain (close to 0.5):**

| FormA      | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:-----------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    4 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    5 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | False    |    0 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    2 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    3 |       1 |                    0 |


## Case=case2746wop, Perturbation=(0.04, 869)

### Highest CV (std/mean)

| Formulation   | Merge   |   A_parameter |   n |     mean |      std |        cv |   tail_ratio |
|:--------------|:--------|--------------:|----:|---------:|---------:|----------:|-------------:|
| Chordal_MFI   | True    |             2 |  20 | 117.914  | 11.4606  | 0.097195  |      1.01844 |
| Chordal_MD    | True    |             2 |  20 | 127.659  | 11.4723  | 0.0898667 |      1.1076  |
| Chordal_AMD   | True    |             5 |  20 |  51.3938 |  4.30341 | 0.0837342 |      1.05424 |
| Chordal_AMD   | True    |             2 |  20 | 112.984  |  9.43167 | 0.0834776 |      1.02018 |
| Chordal_MFI   | False   |             0 |  20 |  65.8555 |  5.08232 | 0.0771738 |      1.03356 |
| Chordal_MD    | False   |             0 |  20 |  73.4817 |  5.48812 | 0.0746869 |      1.05372 |
| Chordal_MD    | True    |             3 |  20 |  63.2209 |  4.56659 | 0.0722322 |      1.05499 |
| Chordal_MD    | True    |             5 |  20 |  59.6297 |  4.07985 | 0.0684198 |      1.03105 |
| Chordal_MD    | True    |             4 |  20 |  58.7568 |  3.93773 | 0.0670175 |      1.05074 |
| Chordal_AMD   | False   |             0 |  20 |  62.7652 |  3.69881 | 0.0589309 |      1.03986 |


### Highest tail ratio (max/median)

| Formulation   | Merge   |   A_parameter |   n |   median |      max |   tail_ratio |        cv |
|:--------------|:--------|--------------:|----:|---------:|---------:|-------------:|----------:|
| Chordal_MD    | True    |             2 |  20 | 130.698  | 144.76   |      1.1076  | 0.0898667 |
| Chordal_AMD   | True    |             4 |  20 |  52.8136 |  55.7552 |      1.0557  | 0.0408713 |
| Chordal_MFI   | True    |             3 |  20 |  58.5955 |  61.8187 |      1.05501 | 0.0575954 |
| Chordal_MD    | True    |             3 |  20 |  63.4787 |  66.9691 |      1.05499 | 0.0722322 |
| Chordal_AMD   | True    |             5 |  20 |  52.6917 |  55.5499 |      1.05424 | 0.0837342 |
| Chordal_AMD   | True    |             3 |  20 |  56.0499 |  59.0856 |      1.05416 | 0.0453818 |
| Chordal_MD    | False   |             0 |  20 |  75.4777 |  79.5324 |      1.05372 | 0.0746869 |
| Chordal_MD    | True    |             4 |  20 |  59.5837 |  62.607  |      1.05074 | 0.0670175 |
| Chordal_MFI   | True    |             4 |  20 |  54.8002 |  57.5656 |      1.05046 | 0.0528343 |
| Chordal_MFI   | True    |             5 |  20 |  56.4012 |  58.777  |      1.04212 | 0.0549224 |


### Most rank-unstable strategies (rank_unique_count)

| Formulation   | Merge   |   A_parameter |   rank_unique_count | rank_swaps   |
|:--------------|:--------|--------------:|--------------------:|:-------------|
| Chordal_AMD   | True    |             3 |                   6 | True         |
| Chordal_MD    | True    |             5 |                   6 | True         |
| Chordal_AMD   | True    |             5 |                   5 | True         |
| Chordal_MFI   | True    |             3 |                   5 | True         |
| Chordal_MFI   | True    |             4 |                   5 | True         |
| Chordal_MD    | True    |             4 |                   5 | True         |
| Chordal_AMD   | True    |             4 |                   5 | True         |
| Chordal_MFI   | False   |             0 |                   4 | True         |
| Chordal_MD    | True    |             3 |                   4 | True         |
| Chordal_MFI   | True    |             5 |                   4 | True         |


### Pairwise win-probability extremes

**Most decisive (A much faster than B):**

| FormA       | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:------------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MFI | False    |    0 | Chordal_MFI | True     |    3 |       1 |                    1 |
| Chordal_AMD | False    |    0 | Chordal_AMD | True     |    2 |       1 |                    1 |
| Chordal_MFI | False    |    0 | Chordal_MFI | True     |    2 |       1 |                    1 |
| Chordal_MD  | True     |    5 | Chordal_MFI | True     |    5 |       1 |                    1 |
| Chordal_MD  | True     |    5 | Chordal_MFI | True     |    4 |       1 |                    1 |


**Most uncertain (close to 0.5):**

| FormA      | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:-----------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    4 |       1 |                    0 |
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    5 |       1 |                    0 |
| Chordal_MD | True     |    3 | Chordal_MFI | False    |    0 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    2 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    3 |       1 |                    1 |


## Case=case2746wop, Perturbation=(0.05, 1547)

### Highest CV (std/mean)

| Formulation   | Merge   |   A_parameter |   n |     mean |      std |        cv |   tail_ratio |
|:--------------|:--------|--------------:|----:|---------:|---------:|----------:|-------------:|
| Chordal_AMD   | True    |             5 |  20 |  63.2889 |  7.34214 | 0.11601   |      1.08055 |
| Chordal_MFI   | True    |             2 |  20 | 154.864  | 16.391   | 0.105842  |      1.0345  |
| Chordal_MD    | True    |             2 |  20 | 172.422  | 16.6069  | 0.0963151 |      1.01639 |
| Chordal_AMD   | True    |             2 |  20 | 152.569  | 13.9878  | 0.0916816 |      1.03225 |
| Chordal_MD    | True    |             3 |  20 |  77.7515 |  6.90149 | 0.0887634 |      1.02637 |
| Chordal_MD    | False   |             0 |  20 | 100.926  |  8.34846 | 0.0827186 |      1.02221 |
| Chordal_MD    | True    |             4 |  20 |  75.7718 |  6.18414 | 0.0816153 |      1.04064 |
| Chordal_MFI   | True    |             5 |  20 |  73.5914 |  5.93742 | 0.0806809 |      1.03373 |
| Chordal_MD    | True    |             5 |  20 |  75.2618 |  5.98957 | 0.0795832 |      1.03927 |
| Chordal_MFI   | True    |             3 |  20 |  74.8451 |  5.91587 | 0.0790416 |      1.03158 |


### Highest tail ratio (max/median)

| Formulation   | Merge   |   A_parameter |   n |   median |      max |   tail_ratio |        cv |
|:--------------|:--------|--------------:|----:|---------:|---------:|-------------:|----------:|
| Chordal_AMD   | True    |             5 |  20 |  66.1057 |  71.4305 |      1.08055 | 0.11601   |
| Chordal_AMD   | True    |             3 |  20 |  74.4537 |  79.0547 |      1.0618  | 0.0657888 |
| Chordal_AMD   | True    |             4 |  20 |  71.8595 |  76.2446 |      1.06102 | 0.0607434 |
| Chordal_MFI   | True    |             4 |  20 |  75.5767 |  79.3999 |      1.05059 | 0.0697323 |
| Chordal_MD    | True    |             4 |  20 |  78.6877 |  81.8856 |      1.04064 | 0.0816153 |
| Chordal_MD    | True    |             5 |  20 |  78.2641 |  81.3374 |      1.03927 | 0.0795832 |
| Chordal_MFI   | True    |             2 |  20 | 164.931  | 170.621  |      1.0345  | 0.105842  |
| Chordal_MFI   | True    |             5 |  20 |  76.5924 |  79.1762 |      1.03373 | 0.0806809 |
| Chordal_MFI   | False   |             0 |  20 |  94.8658 |  98.0456 |      1.03352 | 0.0789858 |
| Chordal_AMD   | True    |             2 |  20 | 161.188  | 166.386  |      1.03225 | 0.0916816 |


## Case=case2746wop, Perturbation=(0.06, 1045)

### Highest CV (std/mean)

| Formulation   | Merge   |   A_parameter |   n |     mean |      std |        cv |   tail_ratio |
|:--------------|:--------|--------------:|----:|---------:|---------:|----------:|-------------:|
| Chordal_MFI   | True    |             2 |  20 | 108.434  | 10.901   | 0.100531  |      1.06453 |
| Chordal_MD    | False   |             0 |  20 |  69.0222 |  6.35954 | 0.0921376 |      1.06629 |
| Chordal_MD    | True    |             2 |  20 | 119.782  | 10.8179  | 0.0903135 |      1.02905 |
| Chordal_AMD   | True    |             5 |  20 |  47.8451 |  3.31154 | 0.0692137 |      1.06566 |
| Chordal_MD    | True    |             3 |  20 |  54.139  |  3.72669 | 0.0688356 |      1.04928 |
| Chordal_AMD   | True    |             2 |  20 | 105.618  |  6.99185 | 0.0661993 |      1.03776 |
| Chordal_MD    | True    |             4 |  20 |  54.5267 |  3.49421 | 0.0640825 |      1.07746 |
| Chordal_MD    | True    |             5 |  20 |  54.121  |  3.20984 | 0.0593087 |      1.07336 |
| Chordal_MFI   | False   |             0 |  20 |  62.004  |  3.64803 | 0.0588354 |      1.06829 |
| Chordal_MFI   | True    |             3 |  20 |  51.102  |  2.49846 | 0.0488917 |      1.02938 |


### Highest tail ratio (max/median)

| Formulation   | Merge   |   A_parameter |   n |   median |      max |   tail_ratio |        cv |
|:--------------|:--------|--------------:|----:|---------:|---------:|-------------:|----------:|
| Chordal_MD    | True    |             4 |  20 |  54.5049 |  58.7266 |      1.07746 | 0.0640825 |
| Chordal_MD    | True    |             5 |  20 |  53.888  |  57.8414 |      1.07336 | 0.0593087 |
| Chordal_MFI   | False   |             0 |  20 |  61.9447 |  66.1747 |      1.06829 | 0.0588354 |
| Chordal_MD    | False   |             0 |  20 |  71.1403 |  75.856  |      1.06629 | 0.0921376 |
| Chordal_AMD   | True    |             5 |  20 |  48.914  |  52.1256 |      1.06566 | 0.0692137 |
| Chordal_MFI   | True    |             2 |  20 | 112.362  | 119.613  |      1.06453 | 0.100531  |
| Chordal_MD    | True    |             3 |  20 |  55.132  |  57.8491 |      1.04928 | 0.0688356 |
| Chordal_AMD   | True    |             3 |  20 |  49.5324 |  51.6562 |      1.04288 | 0.0306039 |
| Chordal_MFI   | True    |             4 |  20 |  51.8306 |  53.854  |      1.03904 | 0.0468549 |
| Chordal_MFI   | True    |             5 |  20 |  52.1282 |  54.1024 |      1.03787 | 0.0382205 |


### Most rank-unstable strategies (rank_unique_count)

| Formulation   | Merge   |   A_parameter |   rank_unique_count | rank_swaps   |
|:--------------|:--------|--------------:|--------------------:|:-------------|
| Chordal_MD    | True    |             3 |                   8 | True         |
| Chordal_MFI   | True    |             3 |                   7 | True         |
| Chordal_MFI   | True    |             4 |                   7 | True         |
| Chordal_MD    | True    |             4 |                   6 | True         |
| Chordal_AMD   | True    |             4 |                   6 | True         |
| Chordal_MD    | True    |             5 |                   6 | True         |
| Chordal_MFI   | True    |             5 |                   6 | True         |
| Chordal_AMD   | True    |             3 |                   5 | True         |
| Chordal_AMD   | True    |             5 |                   5 | True         |
| Chordal_AMD   | False   |             0 |                   4 | True         |


### Pairwise win-probability extremes

**Most decisive (A much faster than B):**

| FormA       | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:------------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MFI | True     |    3 | Chordal_MFI | True     |    5 |       1 |                    1 |
| Chordal_AMD | False    |    0 | Chordal_AMD | True     |    2 |       1 |                    1 |
| Chordal_MFI | True     |    3 | Chordal_MFI | True     |    4 |       1 |                    1 |
| Chordal_MFI | False    |    0 | Chordal_MFI | True     |    2 |       1 |                    1 |
| Chordal_MD  | True     |    5 | Chordal_MFI | True     |    2 |       1 |                    1 |


**Most uncertain (close to 0.5):**

| FormA      | MergeA   |   AA | FormB       | MergeB   |   AB |   n_rep |   P(A_faster_than_B) |
|:-----------|:---------|-----:|:------------|:---------|-----:|--------:|---------------------:|
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    4 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MD  | True     |    5 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | False    |    0 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    2 |       1 |                    1 |
| Chordal_MD | True     |    3 | Chordal_MFI | True     |    3 |       1 |                    0 |


## Case=case2746wop, Perturbation=(0.07, 1709)

### Highest CV (std/mean)

| Formulation   | Merge   |   A_parameter |   n |     mean |      std |        cv |   tail_ratio |
|:--------------|:--------|--------------:|----:|---------:|---------:|----------:|-------------:|
| Chordal_MD    | True    |             2 |  20 | 172.231  | 38.4216  | 0.223082  |      1.02516 |
| Chordal_MD    | False   |             0 |  20 |  90.9558 | 17.0388  | 0.187331  |      1.02496 |
| Chordal_MD    | True    |             3 |  20 |  76.3314 | 13.9759  | 0.183095  |      1.0272  |
| Chordal_AMD   | True    |             5 |  20 |  74.9944 | 10.6363  | 0.141827  |      1.07122 |
| Chordal_AMD   | True    |             3 |  20 |  68.9128 |  9.60491 | 0.139378  |      1.07011 |
| Chordal_AMD   | True    |             4 |  20 |  64.8611 |  7.96253 | 0.122763  |      1.08767 |
| Chordal_AMD   | False   |             0 |  20 |  95.0504 |  9.09657 | 0.0957026 |      1.14681 |
| Chordal_MFI   | True    |             5 |  20 |  94.9105 |  8.78514 | 0.0925623 |      1.14091 |
| Chordal_MFI   | True    |             4 |  20 |  78.7061 |  7.19807 | 0.0914551 |      1.14983 |
| Chordal_MFI   | True    |             3 |  20 |  84.339  |  7.08361 | 0.0839897 |      1.10802 |


### Highest tail ratio (max/median)

| Formulation   | Merge   |   A_parameter |   n |   median |      max |   tail_ratio |        cv |
|:--------------|:--------|--------------:|----:|---------:|---------:|-------------:|----------:|
| Chordal_MFI   | True    |             4 |  20 |  78.1266 |  89.8325 |      1.14983 | 0.0914551 |
| Chordal_AMD   | False   |             0 |  20 |  94.2072 | 108.038  |      1.14681 | 0.0957026 |
| Chordal_MFI   | True    |             5 |  20 |  93.4931 | 106.667  |      1.14091 | 0.0925623 |
| Chordal_MFI   | True    |             3 |  20 |  84.1797 |  93.2732 |      1.10802 | 0.0839897 |
| Chordal_MFI   | False   |             0 |  20 |  99.9964 | 108.895  |      1.08899 | 0.0792738 |
| Chordal_AMD   | True    |             4 |  20 |  67.6238 |  73.5522 |      1.08767 | 0.122763  |
| Chordal_AMD   | True    |             5 |  20 |  80.9144 |  86.6774 |      1.07122 | 0.141827  |
| Chordal_AMD   | True    |             3 |  20 |  72.8647 |  77.9729 |      1.07011 | 0.139378  |
| Chordal_MD    | True    |             4 |  20 |  81.9681 |  86.9285 |      1.06052 | 0.0724726 |
| Chordal_MFI   | True    |             2 |  20 | 188.165  | 194.921  |      1.0359  | 0.0451171 |

