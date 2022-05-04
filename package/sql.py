# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:04:29 2018

@author: 180218
"""
import numpy as np
import sqlite3
class SQLite(object):
    
    def __init__(self,path=None):
        if path is not None:
            print('table initializing '+path)
            self._path=path
            self._tables=self.get_table()
            self.table_name=''
    @property
    def path(self):
        return self._path
    @property
    def table_list(self):
        i=0
        for table in self._tables:
            print(str(i)+' '+table)
            i+=1
    def get_table(self):
        conn=sqlite3.connect(self._path)
        
        cursor=conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables=np.array(cursor.fetchall()).flatten()
        i=0
        for table in tables:
            print(str(i)+' '+table)
            i+=1
        cursor.close()
        conn.close() 
        return tables
    def create_table(self,table_name,dtype='imu'):
        if dtype=='imu':
            keytable=['ts','dev_id','a_x','a_y','a_z','g_x','g_y','g_z','m_x','m_y','m_z']
            dtypetable=['integer NOT NULL','integer NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL','REAL NOT NULL']
        if dtype=='emg':
            keytable=['ch1','ch2']
            dtypetable=['REAL NOT NULL','REAL NOT NULL']
            
        #-------------------get sql instruction---------------
        sql_str=[keytable[i]+' '+dtypetable[i] for i in range(len(keytable))]
        strf='('
        for i in range(len(sql_str)):
            strf+=sql_str[i]+','
        strf=strf[:-1]
        strf+=');'    
        #-------------------get sql instruction---------------
            
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
#        if len(np.argwhere(self._tables==table_name))!=0:
#            print(table_name+' already exist!! so the creattion is faild!')
#            return
        if dtype=='imu':
            print("CREATE TABLE IF NOT EXISTS "+table_name+strf)
            cursor.execute("CREATE TABLE IF NOT EXISTS "+table_name+strf)
        else:
            cursor.execute("CREATE TABLE IF NOT EXISTS "+table_name+" ( ts integer NOT NULL PRIMARY KEY, ch1 REAL NOT NULL, ch2 REAL NOT NULL);")

        print(table_name+' creating success')
        conn.commit()
        self._tables=self.get_table()
        cursor.close()
        conn.close()
    def get_table_info(self,table_name):
        
        if len(np.argwhere(self._tables==table_name))==0:
            print(table_name+' not exist')
            return
        str_sql='PRAGMA table_info('+table_name+')'
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        cursor.execute(str_sql)
        values=cursor.fetchall()
        print('cid name type notnull dflt_value pk')
        self.d_type=[]
        self.d_key=[]
        self.table_name=table_name
        for row in values:
            print(row)
            self.d_type.append(row[2])
            self.d_key.append(row[1])
        print(table_name+' loading success')
#        conn.commit()
        cursor.close()
        conn.close()
#        return self
    def get_info(self,table_name=None):
        if self.table_name!='':
            str_sql='PRAGMA table_info('+self.table_name+')'
        else:
            str_sql='PRAGMA table_info('+table_name+')'
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        cursor.execute(str_sql)
        values=cursor.fetchall()
        self.w=len(values)
#        print(self.table_name)
        if self.table_name!='':
            cursor.execute('SELECT Count(rowid) FROM '+self.table_name)
        else:
            cursor.execute('SELECT Count(rowid) FROM '+table_name)
        values=cursor.fetchall()
#        print('count',values)
        self.l,=values[0]
        self.shape=[self.l,self.w]
        cursor.close()
        conn.close()
        
    def inputsql(self,language):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        cursor.execute(language)
        values=cursor.fetchall()
        for row in values:
            print(row)
        conn.commit()
        cursor.close()
        conn.close()    
    def enable_auto_vacuum(self):
        conn=sqlite3.connect(self._path)
        str_auto='PRAGMA auto_vacuum;'
        cursor=conn.cursor()
        cursor.execute(str_auto)
        print('enable auto_vacuum success')
        conn.commit()
        cursor.close()
        conn.close() 
    def vacuum(self):
        conn=sqlite3.connect(self._path)
        str_auto='VACUUM;'
        cursor=conn.cursor()
        cursor.execute(str_auto)
        print('vacuum success')
        conn.commit()
        cursor.close()
        conn.close() 
    def disable_auto_vacuum(self):
        conn=sqlite3.connect(self._path)
        str_auto='PRAGMA auto_vacuum = NONE;' #=1 也行
        cursor=conn.cursor()
        cursor.execute(str_auto)
        print('disable auto_vacuum success')
        conn.commit()
        cursor.close()
        conn.close() 
    @property
    def values(self,length=10):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        
        cursor.execute("select * from "+self.table_name+" order by ts limit "+str(length))
        
        result=cursor.fetchall()
        num_rows = int(cursor.rowcount)
        return np.array(result)
    def export_array(self):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        cursor.execute("select * from "+self.table_name)
        result=cursor.fetchall()
        num_rows = int(cursor.rowcount)
        return np.array(result)
    def export_data(self,win=100000,condition=''):
        conn=sqlite3.connect(self._path)
        
        shift=win
        str_where=''
        out=[]
        if condition!='':
            str_where=' where '+condition
#            print(str_where)
            self.shape[0]=int(self.shape[0]/8)
        for i in range(0,self.shape[0],shift):
            cursor=conn.cursor()
            print('loading...',str(int(i/1000))+'k','/',str(int(self.shape[0]/1000))+'k',round(i/self.shape[0]*100,2),'%')
            if (self.shape[0]-i)<shift:
                shift=self.shape[0]-i
            try:
                if str_where!='':
                    cursor.execute("select * from (select * from "+self.table_name+str_where+")"+' limit '+str(shift)+' offset '+str(i))
                else:
                    cursor.execute("select * from "+self.table_name+' limit '+str(shift)+' offset '+str(i))
#                result=cursor.fetchall()
            except RuntimeError:
                print('OperationError')
            out.append(cursor.fetchall())
            cursor.close()
        plane=np.empty((0,self.shape[1]))
        for i in out:
            print(plane.shape)
            print(np.array(i).shape)
            if np.array(i).shape[0]==0:
                continue
            plane=np.vstack((plane,np.array(i)))
#        num_rows = int(cursor.rowcount)
        self.shape=[self.l,self.w]
        return plane
    def export_filterd(self,key,dev_id):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()

        cursor.execute("select * from "+self.table_name+' where '+key+'=='+str(dev_id))
        result=cursor.fetchall()
        return np.array(result)
    def browse_data(self,tablename=None):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        if self.table_name!='':
            cursor.execute("SELECT * from "+self.table_name)
        else:
            cursor.execute("SELECT * from "+tablename)
        values=cursor.fetchall()
        for row in values:
            print(row)
        cursor.close()
        conn.close() 
    def view(self,tablename=None,length=10):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        if self.table_name!='':
            cursor.execute("select * from "+self.table_name+" order by ts limit "+str(length))
        else:
            cursor.execute("select * from "+tablename+" order by ts limit "+str(length))

        values=cursor.fetchall()
        for row in values:
            print(row)
        cursor.close()
        conn.close() 
    def insert_array(self,tablename=None,array=None):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        if self.table_name!='':
            sqlstr=' VALUES ('+('?,'*self.shape[1])[:-1]+' )'
            cursor.executemany('INSERT INTO '+self.table_name+sqlstr, (array))

#            cursor.executemany('INSERT INTO '+self.table_name+' VALUES (?,?,?,?,?,?,?,?,?,?,? )', (array))
        else:
            cursor.executemany('INSERT INTO '+tablename+' VALUES (?,?,?)', array)
        conn.commit()
        cursor.close()
        conn.close() 
    def update(self,key,value,tablename=None,condition=''):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        if condition!='':
            str_where=' where '+condition
        if self.table_name!='':
            sqlstr=' VALUES ('+('?,'*self.shape[1])[:-1]+' )'
#            UPDATE imu01 SET ts = 0+9 WHERE rowid == 0;
            cursor.execute('update '+self.table_name+' set '+key+'='+str(value)+ str_where)

#            cursor.executemany('INSERT INTO '+self.table_name+' VALUES (?,?,?,?,?,?,?,?,?,?,? )', (array))
        else:
            cursor.executemany('INSERT INTO '+tablename+' VALUES (?,?,?)', array)
        conn.commit()
        cursor.close()
        conn.close() 
    def update2(self,key,value,tablename=None,condition=''):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        if condition!='':
            str_where=' where '+condition
        if self.table_name!='':
            sqlstr=' VALUES ('+('?,'*self.shape[1])[:-1]+' )'
#            UPDATE imu01 SET ts = 0+9 WHERE rowid == 0;
            cursor.execute('update '+self.table_name+' set '+key+'='+key+'+'+str(value)+ str_where)

#            cursor.executemany('INSERT INTO '+self.table_name+' VALUES (?,?,?,?,?,?,?,?,?,?,? )', (array))
        else:
            cursor.executemany('INSERT INTO '+tablename+' VALUES (?,?,?)', array)
        conn.commit()
        cursor.close()
        conn.close() 

    def drop_table(self,table_name):
        conn=sqlite3.connect(self._path)
        cursor=conn.cursor()
        cursor.execute('drop table '+table_name)
        values=cursor.fetchall()
        conn.commit()
        for row in values:
            print(row)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables=np.array(cursor.fetchall()).flatten()
        self._tables=self.get_table()
        conn.commit()
        cursor.close()
        conn.close()   
        
class Table(SQLite):
    def __init__(self,db,table_name):
#        super.__init__(path=db.path)
        self._path=db.path
        self.table_name=table_name
        self.get_info()
        
#    def browse_data(self):
#        conn=sqlite3.connect(self.path)
#        cursor=conn.cursor()
#        cursor.execute("SELECT * from "+self.table_name)
#        values=cursor.fetchall()
#        for row in values:
#            print(row)
#        cursor.close()
#        conn.close()    
