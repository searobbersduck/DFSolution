# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-08-28 21:21:59
# @Last Modified by:   Bai
# @Last Modified time: 2019-08-31 11:24:12
import fasttext
import pandas as pd
import jieba
import numpy as np
from sklearn.model_selection import KFold
import argparse
import random

def stopwordset(stopword_path):
  stopwords = [line.strip() for line in open(stopword_path,encoding='UTF-8').readlines()]
  return set(stopwords)


def seg_sentence(sentence, stopwordlist):
  sentence_depart = jieba.cut(sentence.strip())
  seg_result = [x.strip() for x in sentence_depart if x not in stopwordlist]
  seg_result = [x for x in seg_result if x]
  return seg_result

def gen_data_set(data_path, label_path, stopword_path):
  # stopwords = stopwordset(stopword_path)
  stopwords = set([])
  data_df = pd.read_csv(data_path)
  label_df = pd.read_csv(label_path)
  label_map = {}
  for index, row in label_df.iterrows():
    label_map[row['id']] = row['label']

  sentences = []
  labels = []
  for index, row in data_df.iterrows():
    content = row['content']
    if not content or type(content) == float:
      continue
    seg_result = seg_sentence(content, stopwords)
    label = label_map[row['id']]
    sentences.append(seg_result)
    labels.append(label)
  indexs = [i for i in range(len(labels))]
  random.shuffle(indexs)
  return np.array(sentences)[indexs], np.array(labels)[indexs]

def gen_data_set_upsampling(data_path, label_path, stopword_path):
  # stopwords = stopwordset(stopword_path)
  stopwords = set([])
  data_df = pd.read_csv(data_path)
  label_df = pd.read_csv(label_path)
  label_map = {}
  for index, row in label_df.iterrows():
    label_map[row['id']] = row['label']

  sentences = []
  labels = {'0': [], '1':[], '2': []}
  i = 0
  for index, row in data_df.iterrows():
    content = row['content']
    if not content or type(content) == float:
      continue
    seg_result = seg_sentence(content, stopwords)
    label = label_map[row['id']]
    sentences.append(seg_result)
    labels[str(label)].append(i)
    i += 1

  max_num = 0
  append_labels = {'0': [], '1':[], '2': []}
  for k in labels:
    if len(labels[k]) > max_num:
      max_num = len(labels[k])
  for k in labels:
    if len(labels[k]) < max_num:
      for j in range(max_num - len(labels[k])):
        append = random.choice(labels[k])
        sentences.append(sentences[append])
        append_labels[k].append(i)
        i += 1
  labels_align = [0 for i in range(len(sentences))]
  for k in labels:
    for x in labels[k]:
      labels_align[x] = k
  for k in append_labels:
    for x in append_labels[k]:
      labels_align[x] = k    

  indexs = [i for i in range(len(labels_align))]
  random.shuffle(indexs)
  return np.array(sentences)[indexs], np.array(labels_align)[indexs]

def gen_fasttext_dataset(sentences, labels, data_path):
  with open(data_path, 'w', encoding='utf-8') as f:
    for i in range(len(sentences)):
      line = '__label__' + str(labels[i]) + ' ' + ' '.join(sentences[i])
      f.write(line + '\n')


class FasttextEstimator(object):
  """docstring for FasttextEstimator"""
  def __init__(self):
    super(FasttextEstimator, self).__init__()
    self.model = None
  
  def fit(self, *args, **kwargs):
    if len(args) == 0:
        self.model = fasttext.train_supervised(**kwargs)
    else:
      self.model = fasttext.train_supervised(*args, **kwargs)
  def estimate(self, test_data_path):
    return self.model.test_label(path=test_data_path)

  def estimate_marco_f1(self, test_data_path):
    ret = self.model.test_label(path=test_data_path)
    marco_f1 = 0.0
    label_num = 0
    for k in ret:
      label_num += 1
      marco_f1 += ret[k]['f1score']
      print(k + ":" + str(ret[k]['f1score'])) 
    return marco_f1 / label_num


def cv_estimate(n_splits, X_train, Y_train, estimator):
  cv = KFold(n_splits=n_splits)
  ret = 0.0
  i = 0
  for train, test in cv.split(X_train, Y_train):
    data_path = 'fasttext_data_' + str(i) + '.txt'
    gen_fasttext_dataset(np.array(X_train)[train], np.array(Y_train)[train], data_path)
    estimator.fit(input=data_path, lr=0.5, epoch=20, wordNgrams=3)
    test_data_path = "fasttext_data_test_" + str(i) + '.txt'
    gen_fasttext_dataset(np.array(X_train)[test], np.array(Y_train)[test], test_data_path)
    mf1 = estimator.estimate_marco_f1(test_data_path)
    print("fold " + str(i) + ": " + str(mf1))
    ret += mf1
  return ret / float(n_splits)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "parser"
            ""
        )
    )
    parser.add_argument(
        "stopwords_path",
        help="stopwords",
    )
    parser.add_argument(
        "data_path",
        help="data_path",
    )
    parser.add_argument(
        "label_path",
        help="label_path",
    )
    args = parser.parse_args()

    sentences, labels = gen_data_set(args.data_path, args.label_path, args.stopwords_path)
    estimator = FasttextEstimator()
    ret = cv_estimate(5, sentences, labels, estimator)
    print("final marco_f1: " + str(ret))