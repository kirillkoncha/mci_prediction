# Automatic Detection of Mild Cognitive Impairment and Alzhemier’s Disease Using Machine Learning

This repository contains code for the Master's Thesis "Automatic Detection of Mild Cognitive Impairment and Alzhemier’s Disease Using Machine Learning" by Kirill Koncha (under Frank Tsiwah and Robert Hartsuiker supervision).

The project consists of different spontenous speech features extractors. The work was conducted with the usage of the [DementiaBank](https://talkbank.org/dementia/) dataset.

# Preparation

**Firstly**, to run this project one need to install the dependencies:

```
pip install -r requirements.txt
```

**Secondly**, to work with this repository you need to have data from DementiaBank.

**Thirdly,** to one needs to take from `Pitt_data.xlsx` (sheet `data`) file ID of each participant and put ID and MMSE scores of participants to separate sheet called `match`. Columns with MMSE scores should be named: `mmse0`, `mmse1`, `mmse2`, etc.

**Fourthly,** to extract *Propositional Idea Density/Efficiency* features, a list of sentences with low specificity should be obtained (for example, with this [method](https://github.com/jjessyli/speciteller)). Each sentence should have separate line and a file should have name `sentences_specificity_filtered.txt` and be put in the `data` folder.

**Finally**, there should be [downloaded](https://fasttext.cc/docs/en/crawl-vectors.html) fastText embeddings for English in the `data` folder.

# Usage

To run this project, one need to have The Cookie Theft Picture descriptions annotations from DementiaBank in the CHAT format.

## Data Preprocessing

To get CSV-file with CoNLL-U annotations, run the following code:

```python3

from src.data_preprocessing.data_preprocessor import DataPreprocessor

dataset_path = "path_to_cookie_theft_annotations"
output_path = "path_to_save_df"
mmse_xlsx_path = "path_to_excel_with_mmse_scores"

process_dataset(dataset_path, output_path, mmse_xlsx_path, reutrn_output=False, match_mmse=True)
```

## Features Extraction

See the `FeaturesExtractor` in `src.features_extraction.features_extractor` for features and their descriptions.

All features are extracted with the respective method:

```python3
import pandas as pd

from src.features_extraction.features_extractor import FeaturesExtractor

df = pd.read_csv("data.csv")

fe = FeaturesExtractor()

df["mlu"] = df["annotation"].apply(fe.extract_mlu)

```

Some of extraction methods are applied to the `annotation` column (column with CoNLL-U annotations), some features are applied to the `speech` column (column with the transcribed speech). See each function documentation for this information.

*NB:* `extract_sid` and `extract_sid_efficiency` functions require to obtain a list of ICUs. It should be put in the `data` folder. To extract ICUs, use `src.features_extraction.sid_kmeans.kmeans_features`:

```python3
from src.features_extraction.sid_kmeans.kmeans_features import KMeansFeatures

kmeans_features = KMeansFeatures()
vectors = kmeans_features.extract_vectors_for_sid(annotations=df["annotations"].tolist(), reutrn_output=True)

get_clusters(vectors, 23, "path_to_save.csv", save_output=True, reutrn_output=False)
```

To extract `extract_pid` and `extract_pid_efficiency` functions, one have to obtain list of low specificity sentences (see the *Preparation* section).

# Statistical Analysis


