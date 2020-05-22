# -*- encoding: utf-8 -*-
# @MoudleName: window.py
# @Function : GUI
# @Author : XsC
# @Time : 2020/05/18 15:17

# GUI
from tkinter import *
import tkinter.filedialog
import copy

# network
import dataLoader
import xscnetwork
import data

defaultNN = 'xscNetwork'
defaultID = 'datas.csv'
NerualNetwork = 'xscNetwork'
InputData = 'datas.csv'
net = xscnetwork.load(NerualNetwork)
data.getData(InputData)
training_data, validation_data, test_data = dataLoader.load_data_wrapper()


# design the functions
def chooseNerualNetwork():
    global net, defaultNN
    lb1.config(text='')
    NerualNetwork = tkinter.filedialog.askopenfilename()
    if NerualNetwork != '':
        try:
            net = xscnetwork.load(NerualNetwork)
            lb1.config(text='您选择的神经网络是' + NerualNetwork)
        except:
            NerualNetwork = defaultNN
            net = xscnetwork.load(NerualNetwork)
            lb1.config(text='您选择的不是神经网络，将使用默认神经网络')
    else:
        NerualNetwork = defaultNN
        net = xscnetwork.load(NerualNetwork)
        lb1.config(text='您没有选择任何神经网络,将使用默认神经网络')


def chooseInputData():
    global InputData, training_data, validation_data, test_data, defaultID
    lb2.config(text='')
    InputData = tkinter.filedialog.askopenfilename()
    if InputData != '':
        try:
            data.getData(InputData)
            lb2.config(text='您选择的输入数据是' + InputData)
        except:
            InputData = defaultID
            data.getData(InputData)
            lb2.config(text='您选择了错误的输入数据，将使用默认数据')
    else:
        InputData = defaultID
        data.getData(InputData)
        lb2.config(text='您没有选择任何输入数据,将使用默认数据')
    training_data, validation_data, test_data = dataLoader.load_data_wrapper()


def calculate():
    lb3.config(text='')
    global net, data
    try:
        test_data_copy = copy.deepcopy(test_data)
        net.getResults(test_data_copy)
        data.getResult(InputData)
    except:
        lb3.config(text='请选择神经网络和输入数据')


def showResults():
    lb4.config(text='')
    text.delete('1.0', 'end')
    try:
        with open('showResults.txt', 'r', encoding='utf-8') as f:
            for line in f:
                text.insert(INSERT, line)
    except:
        lb4.config(text='请先进行计算')


# design the window

root = Tk()
root.title('基于BP神经网络的OJ系统编程能力评价')
root.geometry('500x500')
root.wm_resizable(False, False)
root.configure(background='white')
text = Text(root, width=250, height=250, relief=GROOVE)
text.place(x=0, y=0, height=250, width=500)
# text.pack()
btChooseNerualNetwork = Button(root, text="选择神经网络", fg="black", relief=GROOVE,
                               command=chooseNerualNetwork)
btChooseNerualNetwork.place(relx=0.3, y=300, height=40, width=200)
# btChooseNerualNetwork.pack()
lb1 = Label(root, text='', font=('宋体', 10), bg='white')
lb1.place(relx=0, y=340, height=10, width=500)
# lb1.pack()
btChooseInputData = Button(root, text="选择输入文件", fg="black", relief=GROOVE,
                           command=chooseInputData)
btChooseInputData.place(relx=0.3, y=350, height=40, width=200)
# btChooseInputData.pack()
lb2 = Label(root, text='', font=('宋体', 10), bg='white')
lb2.place(relx=0, y=390, height=10, width=500)
# lb2.pack()
btCalculate = Button(root, text="进行计算", fg="black", relief=GROOVE,
                     command=calculate)
btCalculate.place(relx=0.3, y=400, height=40, width=200)
# btCalculate.pack()
lb3 = Label(root, text='', font=('宋体', 10), bg='white')
lb3.place(relx=0, y=440, height=10, width=500)
# lb3.pack()
btShowResult = Button(root, text="查看结果", fg="black", relief=GROOVE,
                      command=showResults)
btShowResult.place(relx=0.3, y=450, height=40, width=200)
# btShowResult.pack()
lb4 = Label(root, text='', font=('宋体', 10), bg='white')
lb4.place(relx=0, y=490, height=10, width=500)
# lb4.pack()
root.mainloop()
