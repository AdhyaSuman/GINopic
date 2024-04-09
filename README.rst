
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
| 20NewsGroups    | 20Newsgroups_   | 16309  | 1612    | 20       |
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

This work has been accepted at ECIR 2023!

Read the paper in `arXiv`_.

If you decide to use this resource, please cite:

.. _`arXiv`: https://arxiv.org/abs/2404.02115


::

 @misc{adhya2024ginopic,
      title={GINopic: Topic Modeling with Graph Isomorphism Network}, 
      author={Suman Adhya and Debarshi Kumar Sanyal},
      year={2024},
      eprint={2404.02115},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      }
  

Acknowledgment
--------------
All experiments are conducted using OCTIS_ which is an integrated framework for topic modeling for comparing and optimizing topic models.

**OCTIS**: Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). `OCTIS: Comparing and Optimizing Topic models is Simple!`. EACL. https://www.aclweb.org/anthology/2021.eacl-demos.31/

.. _OCTIS: https://github.com/MIND-Lab/OCTIS
