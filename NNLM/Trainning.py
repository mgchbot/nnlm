#encoding=utf-8
__author__ = 'mgchbot'
import numpy,collections
from numpy import *

dictWordVector2={} #所有词向量
dictWordVector={}
worddim=100
inputTrainText=open("C:\Users\mgchbot\Desktop\wenbenzhaiyao\\2.txt")#读入训练文本

def readandInitWordVector(): #读取训练文本并初始化词向量
    print "init word vectors..."
    i=0
    for line in inputTrainText:
        w=line.split()
        for word in w:
            if not dictWordVector.has_key(word):
                dictWordVector[word]=numpy.random.random(size=worddim)-0.5
                dictWordVector2[i]=word
                i+=1
readandInitWordVector()

HN=60 #隐层神经元数量
wordWindow=4#词窗口大小
EPSILON=0.003#学习速率
I2HVectors=[] #输入层到隐层的参数
H2OVectors=[] #隐层到输出层的参数
Hresult=zeros(HN)#隐层的输出
Oresult=zeros(len(dictWordVector))#输出层的输出
Hbias=zeros(HN) #隐层的偏置
Obias=zeros(len(dictWordVector)) #输出层的偏置
Lpy=[]
Lpa=[]
Lpo=[]
Lpx=[]
words=[]
z0=zeros(worddim)

expectedword=""#训练集中的下一个目标单词


def initNNParameters():#初始化神经网络的参数
    for i in range (0,60):
        I2HVectors.append(numpy.random.random(size=(worddim*wordWindow))-0.5)
    Hbias=numpy.random.random(size=HN)-0.5
    for i in range (0,len(dictWordVector)):
        H2OVectors.append(numpy.random.random(size=HN)-0.5)
    Obias=numpy.random.random(size=len(dictWordVector))-0.5

def calI2H():#计算隐层的输出
    inputvec=zeros(worddim)
    for j in range(0,wordWindow):
        if j==0:
            if dictWordVector.has_key(words[j]):
                inputvec+=dictWordVector[words[j]]
            else:
                inputvec+=z0
        else:
            if dictWordVector.has_key(words[j]):
                inputvec=append(inputvec,dictWordVector[words[j]])
            else:
                inputvec=append(inputvec,z0)
    for j in range(0,HN):
        Hresult[j]=numpy.tanh(dot(inputvec,I2HVectors[j])+Hbias[j])


def calH2O(expectedword=""):#计算输出层的输出
    sum=0
    a=0
    for i in range(0,len(dictWordVector)):
        Oresult[i]=dot(Hresult,H2OVectors[i])+Obias[i]
        sum+=numpy.exp(Oresult[i])
    for i in range(0,len(dictWordVector)):
        Oresult[i]=numpy.exp(Oresult[i])/sum
        # print len(dictWordVector2),len(dictWordVector)
        if  dictWordVector2.has_key(i) and dictWordVector2[i]==expectedword:
            a=Oresult[i]
    return a

def calOutword():
    sum=0
    for i in range(0,len(dictWordVector)):
        Oresult[i]=dot(Hresult,H2OVectors[i])+Obias[i]
        sum+=numpy.exp(Oresult[i])
    for i in range(0,len(dictWordVector)):
        Oresult[i]=numpy.exp(Oresult[i])/sum
    maxp=0
    index=0
    i-0
    for i in range(0,len(Oresult)):
        if maxp<Oresult[i]:
            maxp=Oresult[i]
            index=i
        i+=1
    print dictWordVector2[index]
    return dictWordVector2[index]

def updateOut():
    i=0
    del Lpy[:]
    del Lpa[:]
    for keys in dictWordVector:
        if keys==expectedword:
            Lpy.append(1-Oresult[i])
        else:
            Lpy.append(-Oresult[i])
        Obias[i]+=EPSILON*Lpy[i]
        for k in range(0,HN):
            Lpa.append(Lpy[i]*H2OVectors[i][k])
        for k in range(0,HN):
            H2OVectors[i][k]+=EPSILON*Lpy[i]*Hresult[k]
        i+=1

def updateHiden():
    del Lpo[:]
    for k in range(0,HN):
        Lpo.append((1-Hresult[k]*Hresult[k])*Lpa[k])
    del Lpx[:]
    for i in range(0,worddim*wordWindow):
        temp=0
        for k in range(0,HN):
            temp+=I2HVectors[k][i]*Lpo[k]
        Lpx.append(temp)
    for k in range(0,HN):
        Hbias[k]+=EPSILON*Lpo[k]

    for i in range(0,HN):
        for k in range(0,wordWindow*worddim):
            I2HVectors[i][k]+=EPSILON*Lpo[i]*dictWordVector[words[int(k/worddim)]][k%worddim]

def updateInput():
    for i in range(0,wordWindow):
        for k in range(0,worddim):
            dictWordVector[words[i]][k]+=EPSILON*Lpx[i*worddim+k]

def train():
    m=0
    while(True):
        Psum=0
        m+=1
        if m>1000:
            break
        for line in open("C:\Users\mgchbot\Desktop\wenbenzhaiyao\\2.txt"):
            w=line.split()
            if(len(w)<5):
                continue
            for i in range(0,len(w)-4):
                del words[:]
                for j in range(i,i+4):
                    words.append(w[j])
                expectedword=w[i+4]
                calI2H()
                Psum+=calH2O(expectedword)
                updateOut()
                updateHiden()
                updateInput()
        print Psum

def predict():
    print "input 4 words:"
    del words[:]
    words.append(raw_input("".decode('utf-8').encode('gbk')))
    words.append(raw_input("".decode('utf-8').encode('gbk')))
    words.append(raw_input("".decode('utf-8').encode('gbk')))
    words.append(raw_input("".decode('utf-8').encode('gbk')))
    while(True):
        calI2H()
        words.append(calOutword())
        del words[0]

initNNParameters()
train()
predict()

