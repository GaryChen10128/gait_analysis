# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:00:54 2019

@author: 180218
"""

import numpy as np
import pandas as pd
class Reader(object):
    index_col=False
    header=None
    usecols=None
    head=0
    nrows=None
    specific=None
    standard_time=None
    config=False
    seperate=None
    encoding=None
    @classmethod
    def clr(cls):
        cls.index_col=False
        cls.header=None
        cls.usecols=None
        cls.head=0
        cls.nrows=None
        cls.specific=None 
        cls.standard_time=None
        cls.config=False
        cls.seperate=None
        cls.encoding=None
    @classmethod
    def get_UpperPath(cls,yourpath='.'):
        import os.path
        
        print(os.path.abspath(os.path.join(yourpath, os.pardir))) 
        out=os.path.abspath(os.path.join(yourpath, os.pardir))
        upper_path=out.replace('\\','/')
        return upper_path

    @classmethod
    def figure(cls,**kwargs):
        print('parameter setting...')
        for keys in kwargs:
            if keys=='bias':
                cls.bias=kwargs[keys]
            if keys=='path':
                cls.path=kwargs[keys]
            if keys=='usecols':
                cls.usecols=kwargs[keys]     
            if keys=='head':
                cls.head=kwargs[keys] 
            if keys=='header':
                cls.header=kwargs[keys] 
            if keys=='seperate':
                cls.seperate=kwargs[keys] 
            if keys=='standard_time':
                cls.standard_time=kwargs[keys] 
                print(cls.standard_time)
            if keys=='config':
                cls.config=kwargs[keys] 
            if keys=='encoding':
                cls.encoding=kwargs[keys] 
            if keys=='shape':
                cls.l=kwargs[keys][0]
                cls.w=kwargs[keys][1]
                print(cls.l,cls.w)
                cls.usecols=[i for i in range(cls.head,cls.w+cls.head)]
                cls.nrows=cls.l
            if keys=='log':
                cls.bias=kwargs[keys].bias
                cls.path=kwargs[keys].path
                cls.usecols=kwargs[keys].usecols
                cls.head=kwargs[keys].head
                cls.l=kwargs[keys].l
                cls.w=kwargs[keys].w
                cls.nrows=kwargs[keys].nrows
                cls.specific=kwargs[keys].specific
            if keys=='specific':
                cls.specific=kwargs[keys]
            print(keys,kwargs[keys])
    @classmethod
    def export(cls,**kwargs):
        if cls.config:
            cls.df = pd.read_csv(cls.path, index_col=cls.index_col,header=cls.header,usecols=cls.usecols,nrows=cls.nrows)
            print(cls.df.shape)
            # ndarray=cls.df.as_matrix(columns=cls.df.columns[:])
            ndarray=cls.df.values
            ndarray=ndarray[:,cls.usecols]
            
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
        else:
#            cls.df = pd.read_csv(cls.path, sep=cls.seperate, header=cls.header)
            cls.df = pd.read_csv(cls.path, sep=cls.seperate, header=cls.header,encoding=cls.encoding)
#            cls.df = pd.read_csv(cls.path, sep='=', header=None)
            return cls.df.values
            # return cls.df.as_matrix()
            
                
        

    @classmethod
    def timeanalysis(cls,index):
        cls.ts=cls.df.iloc[:,index].str.replace('-',' ').str.replace(':',' ').str.replace('.',' ').str.split(' ',expand=True).as_matrix()
        return cls.ts
class Writer(object):
    
    automode=True
    path=''
    encoding=None
    @classmethod
    def figure(cls,**kwargs):
        print('parameter setting...')
        for keys in kwargs:
            if keys=='path':
                cls.path=kwargs[keys]
                print('setting',keys)
            if keys=='automode':
                cls.path=kwargs[keys]
                print('setting',keys)
            if keys=='encoding':
                cls.encoding=kwargs[keys]
                print('setting',keys)                
    @classmethod
    def appendfile(cls,array,keys=None):
        cls.f=open(cls.path, "a+")
        tag=''
        if keys is not None:
            for key in keys:
                tag+=key+','
            cls.f.write(tag+str(tuple(array.reshape(1, -1)[0])).replace('(','').replace(')','\n'))
        else:
            cls.f.write(str(tuple(array.reshape(1, -1)[0])).replace('(','').replace(')','\n'))
        if cls.automode:
            cls.f.close()    
    @classmethod
    def appendstring(cls,array,keys=None):
#        cls.f=open(cls.path, "a+",encoding ='utf-8')
#        cls.f=open(cls.path, "a+")
#        cls.f=open(cls.path, "a+",encoding='gb18030',newline='')
        cls.f=open(cls.path, "a+",encoding=cls.encoding,newline='')
        
        tag=''
        x=''
        for i in range(len(array)):
            x=x+array[i]+', '
        x=x.replace('\n','')
        x=x.replace(':',', ')
        if keys is not None:
            for key in keys:
                tag+=key+','
            print('debug',tag+', '+x)
            cls.f.write(tag+', '+x+'\r\n')
        else:
            cls.f.write(x+'\r\n')
        if cls.automode:
            cls.f.close()    
    @classmethod
    def close(cls):
        cls.f.close()  

class Converter(object):
    conbined=True
    mypath='./temp/'
    block_file_key='desktop'
    sql_pathway='./temp/garbagedb.db'
    new_table_header='imu'
    table_type='imu'

    @classmethod
    def figure(cls,**kwargs):
        print('parameter setting...')
        for keys in kwargs:
            if keys=='conbined':
                cls.conbined=kwargs[keys]
            if keys=='mypath':
                cls.mypath=kwargs[keys]
            if keys=='block_file_key':
                cls.block_file_key=kwargs[keys]   
            if keys=='sql_pathway':
                cls.sql_pathway=kwargs[keys]        
            if keys=='new_table_header':
                cls.new_table_header=kwargs[keys]                
    @classmethod
    def start(cls):
        onlydir = [f for f in listdir(cls.mypath) if isdir(join(cls.mypath, f))]
        print('directory list: ',onlydir)
        db=SQLite(path=cls.sql_pathway)
        for dir_item in onlydir:
            dbname=dir_item
            fpath=cls.mypath+dbname+'/'
            table_name=cls.new_table_header+dir_item
            print('try table '+dir_item+'...')
            onlyfiles = [f for f in listdir(fpath) if isfile(join(fpath, f))]
            
            #確認是否已存在此table，存在的話就直接跳過
            if table_name in db.get_table().tolist():
                print(table_name,' table  exist, skip ...')
                continue
            #table不存在，建立新table
            print(table_name,' table does not exist !creating table...')
        
            db.create_table(table_name,dtype=cls.table_type)
            emg_table=Table(db,table_name)
        
            for no in range(len(onlyfiles)):
        #        print(path)
                if onlyfiles[no].find(cls.block_file_key)==-1:
                    print(onlyfiles[no])
                else:
                    continue
                path=fpath+onlyfiles[no]
                #imu
                df = pd.read_csv(path,header=None)
                df=df.as_matrix()
                df=np.array(df,dtype=object)
                print('file-shape is ',df.shape)
                if df.shape[1]==12:
                    df=df[:,1:]
                if len(df)==0:
                    continue
                #emg
        
        #        emg_ts,emg_utr,samplerate=load_emg3(path)
        #        emg_utr=np.array(emg_utr,dtype=object)
        #        df=emg_utr
                #end
                emg_table.insert_array(array=df)      

