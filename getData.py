# -*- coding: utf-8 -*-
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
        while i<length:
            if(item[i]=='.'):
                break
            if(item[i]=='-'):
                str = '0'
                break
            str += item[i]
            i+=1
        L.append(str)
    return L

with open('实验数据.csv',encoding='utf-8') as f, open('details.txt','w',encoding='utf-8') as fd, open('scores.txt','w',encoding='utf-8') as fs:
    reader = csv.reader(f)
    score = ['[']
    for row in reader:
        L = myFormat(row) #去括号
        weight = ['0','0','20','20','25','25','25','20','15']
        writeList = ['[']
        i = 2
        while i<=8:
            writeList.append(L[i])
            writeList.append(',')
            writeList.append(weight[i])
            if i<=7:
                writeList.append(',')
            i+=1
        writeList.append('],')
        writeList.append('\n')
        fd.writelines(writeList)
        str = L[10]
        strlen = len(str)
        score.append(str[0:strlen - 1])
        score.append(',')
    lscore = len(score)
    fs.writelines(score[0:(lscore-1)])
    fs.writelines(']')