# -*- encoding: utf-8 -*-
# @MoudleName: window.py
# @Function : Format the data
# @Author : XsC
# @Time : 2020/04/30 08:12
"""
从csv文件中获取数据(源于国内某知名oj平台)
"""
import os
import csv


def myFormat(row):
    L = []
    for item in row:
        length = len(item)
        i = 0
        str = ""
        while i < length:
            if (item[i] == '.'):
                break
            if (item[i] == '-'):
                str = '0'
                break
            str += item[i]
            i += 1
        L.append(str)
    return L


def formatScore(score):
    if score == 100:
        outscore = '0'
    elif score >= 90:
        outscore = '1'
    elif score >= 80:
        outscore = '2'
    elif score >= 70:
        outscore = '3'
    elif score >= 60:
        outscore = '4'
    else:
        outscore = '5'
    return outscore


def getData(DataName):
    with open(DataName, encoding='GBK') as f, open('details.txt', 'w',
                                                   encoding='utf-8') as fd, open(
            'scores.txt', 'w', encoding='utf-8') as fs:
        reader = csv.reader(f)
        score = []
        for row in reader:
            L = myFormat(row)  # 去括号
            weight = ['0', '0', '20', '20', '25', '25', '25', '20', '15']
            writeList = []
            i = 2
            while i <= 8:
                writeList.append(L[i])
                writeList.append(',')
                writeList.append(weight[i])
                if i <= 7:
                    writeList.append(',')
                i += 1
            writeList.append('\n')
            fd.writelines(writeList)
            str = L[10]
            strlen = len(str)
            score.append(formatScore(int(str[0:strlen - 1])))
            score.append('\n')
        lscore = len(score)
        fs.writelines(score[0:(lscore - 1)])


def getResult(DataName):
    with open(DataName, encoding='GBK') as f, open('results.txt', 'r',
                                                   encoding='utf-8') as fr, open(
            'showResults.txt', 'w', encoding='utf-8') as fsr:
        reader = csv.reader(f)
        for row in reader:
            writeList = []
            L = myFormat(row)  # 去括号
            writeList.append(L[0])  # 写入学号
            writeList.append(',')
            writeList.append(L[1])  # 写入姓名
            writeList.append(',')
            resultsLine = fr.readline()
            writeList.append(resultsLine[0])  # 写入评价
            writeList.append('\n')
            fsr.writelines(writeList)
