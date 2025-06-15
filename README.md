# GINopic: Graph Isomorphism Network-Based Topic Modeling
# ![GINopic Logo](https://github.com/AdhyaSuman/GINopic/blob/master/Miscellaneous/GINopic_logo.png?raw=true)

**GINopic** is a framework for **Graph Isomorphism Network (GIN)-based topic modeling**. It leverages document similarity graphs to improve word correlations, enhancing topic modeling performance.

---

## 🔍 Framework Overview

<p align="center">
  <img src="https://github.com/AdhyaSuman/GINopic/blob/master/Miscellaneous/GINopic_framework.png" width="600"/>
</p>

---

## 📊 Datasets
We used the following datasets for evaluation:

| Dataset        | Source  | # Docs  | # Words | # Labels |
|---------------|---------|---------|---------|----------|
| **20NewsGroups** | [20Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) | 16,309 | 1,612 | 20 |
| **BBC News**   | [BBC-News](https://github.com/MIND-Lab/OCTIS) | 2,225 | 2,949 | 5 |
| **SearchSnippets** | [SearchSnippets](https://github.com/qiang2100/STTM/blob/master/dataset/SearchSnippets.txt) | 12,270 | 2,000 | 8 |
| **Bio** | [Bio](https://github.com/qiang2100/STTM/blob/master/dataset/Biomedical.txt) | 18,686 | 2,000 | 20 |
| **StackOverflow** | [StackOverflow](https://github.com/qiang2100/STTM/blob/master/dataset/StackOverflow.txt) | 15,696 | 1,860 | 20 |

---

## 📘 Tutorials

To understand and use GINopic efficiently, we provide a tutorial notebook that demonstrates how to run the model, evaluate results, and explore the outputs.

You can open it directly in Google Colab using the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdhyaSuman/GINopic/blob/master/Notebooks/Example.ipynb)

> 📁 **Path:** `Notebooks/Example.ipynb`

This notebook demonstrates:

* How to load a dataset and configure GINopic
* Run the topic modeling pipeline
* Evaluate performance and visualize the results

---

## 📖 Citation
This work has been accepted at **NAACL 2024**! 🎉

📄 Read the paper:
- **[ACL Anthology](https://aclanthology.org/2024.naacl-long.342/)**
- **[ArXiv](https://arxiv.org/abs/2404.02115)**

### 📌 BibTeX
```bibtex
@inproceedings{adhya2024ginopic,
    title = "{GIN}opic: Topic Modeling with Graph Isomorphism Network",
    author = "Adhya, Suman and Sanyal, Debarshi",
    editor = "Duh, Kevin and Gomez, Helena and Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.342",
    pages = "6171--6183",
    abstract = "Topic modeling is a widely used approach for analyzing and exploring large document collections. Recent research efforts have incorporated pre-trained contextualized language models, such as BERT embeddings, into topic modeling. However, they often neglect the intrinsic informational value conveyed by mutual dependencies between words. In this study, we introduce GINopic, a topic modeling framework based on graph isomorphism networks to capture the correlation between words. By conducting intrinsic (quantitative as well as qualitative) and extrinsic evaluations on diverse benchmark datasets, we demonstrate the effectiveness of GINopic compared to existing topic models and highlight its potential for advancing topic modeling."
}
```

---

## Acknowledgment
All experiments were conducted using **[OCTIS](https://github.com/MIND-Lab/OCTIS)**, an integrated framework for topic modeling, comparison, and optimization.

📌 **Reference:** Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). *"OCTIS: Comparing and Optimizing Topic Models is Simple!"* [EACL](https://www.aclweb.org/anthology/2021.eacl-demos.31/).

---

🌟 **If you find this work useful, please consider citing our paper and giving a star ⭐ to the repository!**

