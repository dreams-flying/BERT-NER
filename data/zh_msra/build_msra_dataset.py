# coding:utf-8
from __future__ import print_function
import codecs
import re
import json
import codecs

def wordtag():
    input_file = ['raw_data/train1.txt', 'raw_data/testright1.txt']
    output_file = ['wordtag_train.txt', 'wordtag.txt']
    for input_path, output_path in zip(input_file, output_file):
        input_data = codecs.open(input_path, 'r', 'utf-8')
        output_data = codecs.open(output_path, 'w', 'utf-8')
        for line in input_data.readlines():
            # line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
            line = line.strip().split()

            if len(line) == 0:
                continue
            for word in line:
                word = word.split('/')
                if word[1] != 'o':
                    if len(word[0]) == 1:
                        output_data.write(word[0] + "/B_" + word[1] + " ")
                    elif len(word[0]) == 2:
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        output_data.write(word[0][1] + "/E_" + word[1] + " ")
                    else:
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        for j in word[0][1:len(word[0]) - 1]:
                            output_data.write(j + "/M_" + word[1] + " ")
                        output_data.write(word[0][-1] + "/E_" + word[1] + " ")
                else:
                    for j in word[0]:
                        output_data.write(j + "/o" + " ")
            output_data.write('\n')

def generate_data():
    tag2id = {'': 0,
              'B_ns': 1,
              'B_nr': 2,
              'B_nt': 3,
              'M_nt': 4,
              'M_nr': 5,
              'M_ns': 6,
              'E_nt': 7,
              'E_nr': 8,
              'E_ns': 9,
              'o': 0}

    tag2label = {1: "ns",
                 2: "nr",
                 3: "nt"}

    input_file = ['wordtag_train.txt', 'wordtag.txt']
    output_file = ['train.json', 'test.json']
    for input_path, output_path in zip(input_file, output_file):
        input_data = codecs.open(input_path, 'r', 'utf-8')
        fw = open(output_path, 'w', encoding='utf-8')

        textList = []
        labelList = []
        for line in input_data.readlines():
            # print(len(textList))
            line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
            # print(line)
            for sen in line:
                # print(sen)
                sen = sen.strip().split()
                if len(sen) == 0:
                    continue
                linedata = []
                linelabel = []
                num_not_o = 0
                for word in sen:
                    word = word.split('/')
                    linedata.append(word[0])
                    linelabel.append(tag2id[word[1]])

                    if word[1] != 'o':
                        num_not_o += 1

                if num_not_o != 0:
                    text = ''.join(linedata)
                    tempList = []
                    for i in range(len(linedata)):
                        if linelabel[i] in [1, 2, 3]:
                            j = 0
                            while i + 1 + j < len(linedata):
                                if linelabel[i + j + 1] in [7, 8, 9]:
                                    j += 1
                                    break
                                if linelabel[i + j + 1] in [4, 5, 6]:
                                    j += 1
                                else:
                                    break
                            entity = ''.join(linedata[i: i + j + 1])
                            la = tag2label[linelabel[i]]
                            tempList.append({la: entity})
                    tempDict = {}
                    tempDict['text'] = text
                    tempDict['label'] = tempList
                    l = json.dumps(tempDict, ensure_ascii=False)
                    fw.write(l)
                    fw.write('\n')
                    textList.append(text)
        # input_data.close()


if __name__ == '__main__':
    wordtag()
    generate_data()