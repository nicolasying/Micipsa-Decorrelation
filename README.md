# Micipsa-Decorrelation

Construction of *Similarity-* and *Association-*specific semantic space.

## General Idea

*Similarity* space is based on the penultimate step results of the adapted [WordNetEmbedding](https://github.com/nicolasying/WordNet-Embeddings) algorithm, the step before the computation of a PCA and dimension reduction. [1]
In `Decorrelation_Dimension.ipynb` and `Decorrelation_French.ipynb`, we start by importing the intermediate result (an untruncated semantic space) and calculate a PCA transformer. The dimension selection is detailed in the Master's [thesis](https://github.com/nicolasying/Micipsa-Thesis) chapter Methods. 

French space construction necessitates also lexicon alignment between `DepGloVe` and `WOLF`, which can be located in section `Word Alignment in two embeddings` in `Decorrelation_French.ipynb`.

*Association* is the difference between GLM mapped *Similarity* space onto `DepGloVe` and the original `DepGloVe` space, located in section `Decorrelation` in the notebooks.

Word-pair semantic proximity ranking tasks in both axes are also computed in the notebooks. 

Also available in the French notebook are the visualization of semantic spaces and the semantic ranking task performance variation in function of PCA-dimensions. 

## File structure

- 3 Decorrelation notebooks: code + results
- *.pdf: visualization used in the thesis
- TensorFlow Projector ***: Notebook to transform `word2vec` format semantic spaces to [TensorFlow Projector](http://projector.tensorflow.org/) compatible format for visual examination
- English/French/French_POS: partial resulting semantic spaces (within the file-size limit of GitHub)

[1] Saedi, C., Branco, A., António Rodrigues, J., & Silva, J. (2018, July). WordNet Embeddings. In Proceedings of The Third Workshop on Representation Learning for NLP (pp. 122–131). Melbourne, Australia: Association for Computational Linguistics. Retrieved Novem- ber 28, 2018, from http://www.aclweb.org/anthology/W18-3016