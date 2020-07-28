# Kaggle Q&A Google Labeling competition
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg) 
![Python 3.6](https://img.shields.io/badge/python-3.3-blue.svg) 
![Spyder 4.1.3](https://img.shields.io/badge/spyder-4.1.3-black) 
![Numpy 1.12.1](https://img.shields.io/badge/numpy-1.12.1-yellow.svg) 
![Matplotlib 2.1.2](https://img.shields.io/badge/matplotlib-2.1.2-blue.svg) 
![Keras 2.3.1](https://img.shields.io/badge/keras-2.3.1-red) 
![Tensorflow 2.1.0](https://img.shields.io/badge/tensorflow-2.1.0-orange) 
![Scikit-image 0.16.2](https://img.shields.io/badge/scikit--image-0.16.2-yellowgreen)  
updated bages!!!  

Code which scored top 16% result in [Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge/overview).  
Basing on algorithm developed by [akensert](https://www.kaggle.com/akensert/quest-bert-base-tf2-0).  
My changes consisted of (indicated by commented out parts of the code):
- change model from BERT to RoBERTa (modifications of: tokenizer, model itself, different configuration of model inter alia, vocalbury size,
maximal position of embedding);
- tunning of training parameters (folds, epochs, batch_size);
- change of arithmetic mean of epochs predictions to weighted average.

## Motivation

To practice deep learning in keras enviroment, transfer learning and get familiar with state of the art of Natural Language Processing.

## Installation

Python is a requirement (Python 3.3 or greater, or Python 2.7). Recommended enviroment is Anaconda distribution to install Python and Jupyter (https://www.anaconda.com/download/).

__Installing dependencies__  
To install can be used pip command in command line.  
  
	pip install -r requirements.txt

__Installing python libraries__  
Exemplary commands to install python libraries:
 
	pip install numpy  
	pip install pandas  
	pip install xgboost  
	pip install seaborn 
	
Additional requirement is Tensorflow GPU support. Process of configuiring it is described [here](https://www.tensorflow.org/install/gpu).

## Code examples

	def create_model():
		q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
		a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
		
		q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
		a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
		
		# Version of attention mask for XLNet
		#q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.float32)
		#a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.float32)
		
		q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
		a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)  
		
	test_predictions = [histories[i].test_predictions for i in range(len(histories))]
	test_predictions = [np.average(test_predictions[i], axis=0, weights=[1./18, 1./6, 2./9, 2./9, 1./3]) for i in range(len(test_predictions))]
	test_predictions = np.mean(test_predictions, axis=0)

	df_sub.iloc[:, 1:] = test_predictions

	df_sub.to_csv('submission.csv', index=False)

## Key Concepts
__Deep Learning__

__Transfer Learning__

__NLP__

__RoBERTa__

__Transformers__

__kaggle__

__Google QUEST Q&A Labeling__
  
## Competition description  
"In this competition, you’re challenged to use this new dataset to build predictive algorithms for different subjective aspects of
question-answering. The question-answer pairs were gathered from nearly 70 different websites, in a "common-sense" fashion.
Our raters received minimal guidance and training, and relied largely on their subjective interpretation of the prompts.
As such, each prompt was crafted in the most intuitive fashion so that raters could simply use their common-sense to complete the task.
By lessening our dependency on complicated and opaque rating guidelines, we hope to increase the re-use value of this data set."[1]  

## Original code by "akensert"  
It uses Huggingface transformer library implementation of BERT to solve this question-answering problem. Name BERT stands for Bidirectional
Encoder Representations from Transformers.
As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads
the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that
it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings
(left and right of the word).[3]
Training of BERT 
Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict
the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. In technical terms,
the prediction of the output words requires:

1. Adding a classification layer on top of the encoder output.
2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
3. Calculating the probability of each word in the vocabulary with softmax.

In case of next sentence prediction algorithm is like this:
1. A [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.
2. A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token
embeddings with a vocabulary of 2.
3. A positional embedding is added to each token to indicate its position in the sequence. The concept and implementation of positional
embedding are presented in the Transformer paper [!!!!!!!!].
4. The entire input sequence goes through the Transformer model.
5. The output of the [CLS] token is transformed into a 2×1 shaped vector, using a simple classification layer (learned matrices of
weights and biases).
6. Calculating the probability of IsNextSequence with softmax.

To implement BERT fine-tuning is required, it consists of:
1. Classification tasks such as sentiment analysis are done similarly to Next Sentence classification, by adding a classification layer on top of
 the Transformer output for the [CLS] token.
2. In Question Answering tasks (e.g. SQuAD v1.1), the software receives a question regarding a text sequence and is required to mark the answer
in the sequence. Using BERT, a Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.
3. In Named Entity Recognition (NER), the software receives a text sequence and is required to mark the various types of entities (Person,
Organization, Date, etc) that appear in the text. Using BERT, a NER model can be trained by feeding the output vector of each token into a
classification layer that predicts the NER label. 

The original English-language BERT model used two corpora in pre-training: BookCorpus and English Wikipedia.

## Description of changes to orginal algorithm (....!!!!)  
Change of main algorithm from BERT to RoBERTa was justified by the fact that second one is an improved version of the first one. The expansion
of the algorithm name is Robustly Optimized BERT Pretraining Approach, it modifications consists of [5]:
(1) training the model longer, with bigger batches, over more data; 
(2) removing the next sentence prediction objective; 
(3) training on longer sequences; 
(4) dynamically changing the masking pattern applied to the training data.

Values of the tuning parameters (folds, epochs, batch_size) was mostly implicated by the kaggle GPU power and competition constrain of kernel
computation limitation to 2 hours run-time.

Final step was calculation of predicitons taking into acount results averaged results for folds.
Weights have been assigned by empricialy tring different values. The change of particular ones was based on the prediction score.
Limitation was only the summing up of weights to one.
Change of arithmetic mean of folds predictions to weighted average improved results in public leaderboard from 0.38459 to 0.38798.
On the other hand as the final scores on the private leaderboard showed it was not good choice. Finally, soo strictly assignment of
weights caused the decrease in final result from 0.36925 to 0.36724.

XLNet was also tested (to show it, the code with it was left commented).
In theory XLNet should overcome BERT limitations. Relying on corrupting the input with masks, BERT neglects dependency between the masked
positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons XLNet, which is characterised by:
1. learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order;
2. overcomes the limitations of BERT thanks to its autoregressive formulation;
3. integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining.
Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering,
natural language inference, sentiment analysis, and document ranking [6].  

After all public score for XLNet version was lower (0.36310) than score (0.37886) for base BERT model and was rejected.  

## Summary  


## Resources
[1] https://www.kaggle.com/c/google-quest-challenge/overview  
[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*,
(https://arxiv.org/abs/1810.04805)  
[3] https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270  
[4] https://www.kdnuggets.com/2018/12/bert-sota-nlp-model-explained.html  
[5] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov,
*RoBERTa: A Robustly Optimized BERT Pretraining Approach*, (https://arxiv.org/abs/1907.11692)  
[6] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le, *XLNet: Generalized Autoregressive Pretraining for Language Understanding*,
(https://arxiv.org/abs/1810.04805)  