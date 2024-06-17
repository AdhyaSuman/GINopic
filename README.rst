
.. image:: https://github.com/AdhyaSuman/GINopic/blob/master/Miscellaneous/GINopic_logo.png?raw=true
  :width: 200
  :align: center
  :alt: Logo
**GINopic** is a framework for Graph Isomorphism Network (GIN)-based topic modeling. It leverages document similarity graphs to improve word correlations within topic modeling.


.. image:: https://github.com/AdhyaSuman/GINopic/blob/master/Miscellaneous/GINopic_framework.png
   :align: center
   :width: 600px
   
Datasets
--------
We have used the following datasets:

+----------------+----------------+--------+---------+----------+
| Name           | Source         | # Docs | # Words | # Labels |
+================+================+========+=========+==========+
| 20NewsGroups   | 20Newsgroups_  | 16309  | 1612    | 20       |
+----------------+----------------+--------+---------+----------+
| BBC_News       | BBC-News_      | 2225   | 2949    | 5        |
+----------------+----------------+--------+---------+----------+
| SearchSnippets | SearchSnippets_| 12270  | 2000    | 8        |
+----------------+----------------+--------+---------+----------+
| Bio            | Bio_           | 18686  | 2000    | 20       |
+----------------+----------------+--------+---------+----------+
| StackOverflow  | StackOverflow_ | 15696  | 1860    | 20       |
+----------------+----------------+--------+---------+----------+

.. _20Newsgroups: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
.. _BBC-News: https://github.com/MIND-Lab/OCTIS
.. _Bio: https://github.com/qiang2100/STTM/blob/master/dataset/Biomedical.txt
.. _SearchSnippets: https://github.com/qiang2100/STTM/blob/master/dataset/SearchSnippets.txt
.. _StackOverflow: https://github.com/qiang2100/STTM/blob/master/dataset/StackOverflow.txt


How to cite this work?
----------------------

This work has been accepted at NAACL 2024!

Read the paper in :

1. `ACL Anthology`_

2. `ArXiv`_

.. _`ACL Anthology`: https://aclanthology.org/2024.naacl-long.342/

.. _`arXiv`: https://arxiv.org/abs/2404.02115


::

 @inproceedings{adhya2024ginopic,
    title = "{GIN}opic: Topic Modeling with Graph Isomorphism Network",
    author = "Adhya, Suman  and
      Sanyal, Debarshi",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.342",
    pages = "6171--6183",
    abstract = "Topic modeling is a widely used approach for analyzing and exploring large document collections. Recent research efforts have incorporated pre-trained contextualized language models, such as BERT embeddings, into topic modeling. However, they often neglect the intrinsic informational value conveyed by mutual dependencies between words. In this study, we introduce GINopic, a topic modeling framework based on graph isomorphism networks to capture the correlation between words. By conducting intrinsic (quantitative as well as qualitative) and extrinsic evaluations on diverse benchmark datasets, we demonstrate the effectiveness of GINopic compared to existing topic models and highlight its potential for advancing topic modeling.",
}
  

Acknowledgment
--------------
All experiments are conducted using OCTIS_ which is an integrated framework for topic modeling for comparing and optimizing topic models.

**OCTIS**: Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). `OCTIS: Comparing and Optimizing Topic models is Simple!`. EACL. https://www.aclweb.org/anthology/2021.eacl-demos.31/

.. _OCTIS: https://github.com/MIND-Lab/OCTIS
