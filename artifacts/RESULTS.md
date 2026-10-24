# Experiment Results

Metrics on the **PhreshPhish** dataset (raw URL + raw HTML).
Each block is a full modality ablation.

## `real_attention_ablation.json`

- fusion: **attention**, head: **xgb**, LLM features: **False**
- dataset: `data/phreshphish_train.csv` (train=2624, test=906)

| configuration | acc | precision | recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| url | 0.898 | 0.900 | 0.896 | 0.898 | 0.966 | 0.099 |
| html | 0.762 | 0.825 | 0.664 | 0.736 | 0.869 | 0.141 |
| llm | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| url+html (base paper) | 0.898 | 0.915 | 0.879 | 0.896 | 0.971 | 0.082 |
| url+llm | 0.905 | 0.909 | 0.901 | 0.905 | 0.966 | 0.091 |
| html+llm | 0.757 | 0.816 | 0.664 | 0.732 | 0.860 | 0.150 |
| url+html+llm (proposed) | 0.916 | 0.929 | 0.901 | 0.915 | 0.972 | 0.068 |

## `real_attention_llm_ablation.json`

- fusion: **attention**, head: **xgb**, LLM features: **True**
- dataset: `data/phreshphish_llm_train.csv` (train=100, test=60)

| configuration | acc | precision | recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| url | 0.650 | 0.696 | 0.533 | 0.604 | 0.739 | 0.233 |
| html | 0.717 | 0.760 | 0.633 | 0.691 | 0.791 | 0.200 |
| llm | 0.700 | 1.000 | 0.400 | 0.571 | 0.700 | 0.000 |
| url+html (base paper) | 0.717 | 0.842 | 0.533 | 0.653 | 0.866 | 0.100 |
| url+llm | 0.817 | 0.828 | 0.800 | 0.814 | 0.901 | 0.167 |
| html+llm | 0.683 | 0.690 | 0.667 | 0.678 | 0.752 | 0.300 |
| url+html+llm (proposed) | 0.733 | 0.750 | 0.700 | 0.724 | 0.817 | 0.233 |

## `real_concat_ablation.json`

- fusion: **concat**, head: **xgb**, LLM features: **False**
- dataset: `data/phreshphish_train.csv` (train=2624, test=906)

| configuration | acc | precision | recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| url | 0.916 | 0.937 | 0.892 | 0.914 | 0.973 | 0.060 |
| html | 0.834 | 0.884 | 0.770 | 0.823 | 0.918 | 0.102 |
| llm | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| url+html (base paper) | 0.906 | 0.942 | 0.865 | 0.902 | 0.977 | 0.053 |
| url+llm | 0.915 | 0.937 | 0.890 | 0.913 | 0.974 | 0.060 |
| html+llm | 0.840 | 0.877 | 0.790 | 0.832 | 0.918 | 0.110 |
| url+html+llm (proposed) | 0.915 | 0.948 | 0.879 | 0.912 | 0.978 | 0.049 |

## `real_concat_llm_ablation.json`

- fusion: **concat**, head: **xgb**, LLM features: **True**
- dataset: `data/phreshphish_llm_train.csv` (train=100, test=60)

| configuration | acc | precision | recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| url | 0.783 | 0.815 | 0.733 | 0.772 | 0.930 | 0.167 |
| html | 0.717 | 0.710 | 0.733 | 0.721 | 0.798 | 0.300 |
| llm | 0.700 | 1.000 | 0.400 | 0.571 | 0.700 | 0.000 |
| url+html (base paper) | 0.833 | 0.833 | 0.833 | 0.833 | 0.940 | 0.167 |
| url+llm | 0.800 | 0.821 | 0.767 | 0.793 | 0.928 | 0.167 |
| html+llm | 0.750 | 0.742 | 0.767 | 0.754 | 0.803 | 0.267 |
| url+html+llm (proposed) | 0.833 | 0.833 | 0.833 | 0.833 | 0.941 | 0.167 |
