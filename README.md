# 命名实体识别
一个可以处理嵌套(nested)和非嵌套的命名实体识别算法</br>
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── save    存放已训练好的模型</br>
├── extract_chinese_and_punct.py</br>
├── ner_train.py    训练代码</br>
├── ner_predict.py    评估和测试代码</br>
# 数据集
[CLUENER](https://www.cluebenchmarks.com/introduce.html)</br>
[GENIA](http://www.geniaproject.org/genia-corpus/pos-annotation)</br>
微软亚研院MSRA命名实体识别数据集
# 使用说明
1.[下载预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
  中文数据集可采用BERT-Base, Chinese等模型</br>
  英文数据集可采用BERT-Base, Cased等模型</br>
  更多的预训练语言模型可参见[bert4keras](https://github.com/bojone/bert4keras)给出的权重。</br>
2.构建数据集(数据集已处理好)</br>
  中文：将下载的cluener数据集放到data/cluener/raw_data文件夹下</br>
     运行```generate_label.py```生成entity_label_map.json和label_map.json</br>
&emsp;&emsp;&emsp;&emsp;&emsp;将下载的zh_msra数据集放到data/zh_msra/raw_data文件夹下</br>
     运行```build_msra_dataset.py```生成train.json和dev.json</br>
  英文：将下载的genia(GENIAcorpus3.02p.tgz)数据集放到data/GENIA/raw_data/genia文件夹下，</br>
     a.运行```parse_genia.py```，生成的数据将会放在data/GENIA/raw_data/processed_genia文件夹下</br>
     b.运行```gen_data_for_genia.py```，</br>
     生成的genia_train.json、genia_dev.json、genia_test.json数据会放在data/GENIA/data文件夹下</br>
     c.运行```generate_label.py```，生成entity_label_map.json和label_map.json</br>
3.训练模型</br>
```
python ner_train.py
```
4.评估和测试</br>
```
python ner_predict.py
```
# 结果
这里只对f1值进行了统计。</br>

| 数据集 | train | dev | test |
| :------:| :------: | :------: | :------: |
| CLUENER | 88.991 | 79.700 |  |
| zh_msra | 98.632 | 91.692 |  |
| GENIA | 89.016 | 74.871 | 73.291 |
