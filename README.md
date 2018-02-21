# Emtions Classifier RUS

Neural network based emotions recognition for russian texts

### 1. Overview

This quiet simple module provides a python class for training, evaluating and applying deep neural network on the task of emotions recognition from text data. It may be implemented in various NLP problems such as chatbots building or social media mining. 

It has to be said that the field of textual emotion detection is still very new, especially speaking of russian texts. Therefore, model's performance provides a great room for improvement.

### 2. Data

As it was mentioned, the task's novelty leads to difficulties with finding a reference model and even a merked dataset. That is why english language dataset was used and translated to russian. 

The original dataset is JULIELab's [EmoBank](https://github.com/JULIELab/EmoBank), based on:

* Sven Buechel and Udo Hahn. 2017. EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. In EACL 2017 - Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics. Valencia, Spain, April 3-7, 2017. Volume 2, Short Papers, pages 578-585. Available: http://aclweb.org/anthology/E17-2092
* Sven Buechel and Udo Hahn. 2017. Readers vs. writers vs. texts: Coping with different perspectives of text understanding in emotion annotation. In LAW 2017 - Proceedings of the 11th Linguistic Annotation Workshop @ EACL 2017. Valencia, Spain, April 3, 2017, pages 1-12. Available: https://sigann.github.io/LAW-XI-2017/papers/LAW01.pdf

This dataset contains about 10 000 english sentences evaluated by experts in Valence-Arousal-Dominance scale which provides mapping to basic emotions in the following manner ([G. Paltoglou et al](https://www.computer.org/csdl/trans/ta/2013/01/tta2013010116-abs.html)):

![]( https://github.com/zaphodbbrx/emotions_classifier_rus/blob/master/img/valence-arousal.gif =500x)

In order to use this dataset for russian texts each sentence from it was tokenized ( nltk's StringTokenizer), lemmatized (WordNetLemmatizer) and finally translated word-by-word to russian via google.translate api.

Also, Valence and Arousal values were transformed into binary classes using threshold tranform. The binary classes obtained were used to generate  multiclass labels of 4 classes for each sentence refering to qudrants of valence-arousal scale, i.e. the final mood classes are:

- "excited, delighted, aroused, astonished"

- "calm, relaxed, content, friendly"

- "angry, annoyed, frustrated, disguted"

- "depressed, bored, sad, gloomy"

### 3. Baseline Model

The absolute majority of sentences in original dataset have Valence and Arousal values of about 3, which makes the obtained mood classses pretty imbalanced. Therefore class weighting is essential when building models for this dataset.

Since the most common solutions for classifier model in NLP tasks are linear models Ridge Classifier was chosen as the baseline model that gives descent performance to compare network model to.

### 4. Neural Network

The network's architecture was inpired by [this post on wildml](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). It contains the following layers:

1. Embedding
2. Convolutional
3. Pooling
4. LSTM
5. Softmax

### 5. Usage

The main functions in emotion_classifier class are:

* **make_classifier**: prepares data, fits and evaluates baseline model
* **make_neural_net**: prepares data, generates Keras sequential model and trains it
* **run_classifier**: applies baseline model to some sample text (make_classifier must be run before this function)
* **run_neural_network**: applies neural network model to some sample text (make_neural_net must be run before this function)



