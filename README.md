# An NLP Approach to Identify SDoH-related Circumstance and Suicide Crisis from Death Investigation Narratives

## Purpose
This repo includes the source code of our suicide-specific SDoH Extraction methods.

## Requirements
Python Environment: >= 3.7

## Installation and Usage
```
git clone https://github.com/bionlplab/suicide_sdoh.git
cd suicide_sdoh
pip install -r requirements.txt
```

### Train Circumstance classifier

Put the train and test set under directory /data/circumstance. The default backbone model is BioBERT. Run:
```
cd src/Hierarchy-BERT-Circumstance
python Circumstance_classifier.py
```

### Train Crisis classifier

Put the train and test set under directory /data/crisis. The default backbone model is BioBERT. Run:
```
cd src/Hierarchy-BERT-Crisis
python Crisis_classifier.py
```

## Data Access

The data that support the findings of this study are available from the NVDRS Restricted Access Database (RAD) [here](https://www.cdc.gov/violenceprevention/datasources/nvdrs/dataaccess.html). Restrictions apply to the availability of these data. The data are available by request for users meeting certain eligibility criteria.

## Reference

This repository contains source codes for the work introduced in this following paper:
```
An NLP Approach to Identify SDoH-related Circumstance and Suicide Crisis from Death Investigation Narratives. 
Song Wang, Yifang Dang, Zhaoyi Sun, Ying Ding, Jyotishman Pathak, Cui Tao, Yunyu Xiao, Yifan Peng. 2023.
```

## Acknowledgment

Research reported in this publication was supported by the National Library of Medicine (NLM) of the National Institutes of Health (NIH) under grant number 4R00LM013001, National Institute on Aging (NIA) of NIH under grant number RF1AG072799, National Institute of Allergy and Infectious Diseases (NIAID) of NIH under grant number 1R01AI130460, National Science Foundation under grant number 2145640, and the NSF AI Institute for Foundations of Machine Learning (IFML). The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH and NSF.

## License

Copyright BioNLP Lab at Weill Cornell Medicine, 2022.
