# 互联网新闻情感分析

## 课题来源
[互联网新闻情感分析](https://www.datafountain.cn/competitions/350)

## 文件说明
1. ***``Generate_TrainFile.ipynb``*** 将训练相关的两个文件合并成一个文件***`train.csv`***
2. ***`Convert_Tsv2Csv`*** 将预测生成的**tsv**格式的文件转换成用于提交的**csv**格式文件
3. ***`sentiment_analysis_of_news.py`***, 训练代码
## 数据集
***``ln -s 'location of your dataset' ./data``***

## 预训练模型
***``ln -s 'location of your model' ./model``***
这里用到的是bert官方提供的中文base的模型

## 训练
调用***`./train.sh`***进行训练，通过设置其中的do_train/do_eval/do_predict,决定是训练或是预测。