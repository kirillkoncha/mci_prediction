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