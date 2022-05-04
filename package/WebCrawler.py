# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:05:24 2019

@author: 180218
"""

import requests
from bs4 import BeautifulSoup
#from .temp_package import *
#from .file import *
class WebCrawler(object):
    index_col=False
    header=None
    usecols=None
    head=0
    nrows=None
    specific=None
    standard_time=None
    rule=''
    tag=None
    erase=None
    target_roll=None
    r=None
    class_=None
    @classmethod
    def clr(cls):
        cls.index_col=False
        cls.header=None
        cls.usecols=None
        cls.head=0
        cls.nrows=None
        cls.specific=None 
    @classmethod
    def figure(cls,**kwargs):
        print('parameter setting...')
        for keys in kwargs:
            if keys=='url':
                cls.url=kwargs[keys]
            if keys=='class_':
                cls.class_=kwargs[keys]
            if keys=='rule' or keys=='manualinput':
                cls.rule=kwargs[keys]
                Reader.clr()
                Reader.figure(path=cls.rule,seperate='=')
                raw=Reader.export()
                for i in range(len(raw)):
                    setattr(cls,raw[i,0],raw[i,1].split(','))
#                for i in range(len(raw)):
#                    if len(raw[i,1].split(','))>1:
#                        setattr(cls,raw[i,0],raw[i,1].split(','))
#                    else:
#                        setattr(cls,raw[i,0],raw[i,1])#tag=
                    print('xxxx')
                    print(raw[i,1].split(','))
                    print('xxxx')

        cls.r = requests.get(cls.url)
        cls.r.encoding='utf-8' #显式地指定网页编码，一般情况可以不用
        # 確認是否下載成功
        if cls.r.status_code == requests.codes.ok:
            print('surfing success',cls.url)
                
            
    @classmethod
    def export(cls,**kwargs):
        cls.df = pd.read_csv(cls.path, index_col=cls.index_col,header=cls.header,usecols=cls.usecols,nrows=cls.nrows)
        print(cls.df.shape)
        ndarray=cls.df.as_matrix(columns=cls.df.columns[:])
        if cls.specific is not None:
            ndarray=ndarray[ndarray[:,cls.specific[0]]==cls.specific[1],:]
        for keys in kwargs:
            if keys=='specific':
                ndarray=ndarray[ndarray[:,kwargs[keys][0]]==kwargs[keys][1],:]
        
        if cls.standard_time is not None:
            array=cls.df.iloc[:,cls.standard_time].str.replace('-',' ').str.replace(':',' ').str.split(' ',expand=True).as_matrix()
    #        print(array)
            array=np.array(array,dtype=float)
            temp_ts=array[:,3]*60*60+array[:,4]*60+array[:,5]+array[:,6]*0.001
            ts=temp_ts-temp_ts[0]
            ndarray[:,cls.standard_time]=ts
            ndarray=np.array(ndarray,dtype=float)
        return ndarray
    @classmethod
    def timeanalysis(cls,index):
        cls.ts=cls.df.iloc[:,index].str.replace('-',' ').str.replace(':',' ').str.replace('.',' ').str.split(' ',expand=True).as_matrix()
        return cls.ts

    def __init__(self,url):
        self.url=url
        
        self.r = requests.get(self.url)
        self.r.encoding='gbk' #显式地指定网页编码，一般情况可以不用
        # 確認是否下載成功
        if self.r.status_code == requests.codes.ok:
          # 以 BeautifulSoup 解析 HTML 程式碼
          print('surfing success',self.url)
    def setup_rule(self,path):
        Reader.figure(path='D:/iii/exp_data/webcraw/tarot/config.txt')

        raw=Reader.export()
    @classmethod
    def soup(cls):
        result=[]
        soup = BeautifulSoup(cls.r.text, 'html.parser')
#        print(cls.tag)
        if cls.class_ is None:
            
            stories = soup.find_all(cls.tag)
        else:
            stories = soup.find_all(cls.tag,class_=cls.class_)
        for s in stories:
#           print(s.text)
#            print(s.text)
            result.append(s.text)
#        print(result)
#        print(cls.target_roll)
        if cls.target_roll is not None:
            T=[result[i] for i in list(map(int, cls.target_roll))]
        else:
            T=result
        if cls.erase is not None:
            for i in range(len(T)):
                for j in range(len(cls.erase)):
    #                print(cls.erase[j])
                    T[i].replace(cls.erase[j],'')
#        T = [result[i].replace('\n','').replace(' ','') for i in index]

        return T
#if __name__=='main':
#    print(getattr(WebCrawler(),nrow))
