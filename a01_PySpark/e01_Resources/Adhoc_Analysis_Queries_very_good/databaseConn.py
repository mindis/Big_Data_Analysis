from configparser import ConfigParser
import pyodbc
import os
import pandas as pd
import numpy as np

def ConnnectionLink(DRIVER,DSN,AUTH,UID,PWD,Database):
    '''
     ConnnectionLink function create connection string for Teradata
     >>ConnnectionLink(DRIVER,DSN,AUTH,UID,PWD,Database)
       'DRIVER={Teradata};DBCNAME=*********;Authentication=LDAP;UID=userid;PWD=pasword;DATABASE=pafsdss'
    '''
    return ';'.join([DRIVER,DSN,AUTH,UID,PWD,Database])

if __name__=='__main__':
    parser=ConfigParser()
    #Read authentication acess
    parser.read('config.ini')
    UID = parser.get('Teradata','username')
    PWD = parser.get('Teradata','password')
    DSN =parser.get('Teradata', 'DSN')
    DRIVER='DRIVER={Teradata}'
    AUTH = 'Authentication=LDAP'
    DATABASE='DATABASE=pafsdss'
    #Create connection string
    conn_string = ConnnectionLink(DRIVER,DSN,AUTH,UID,PWD,DATABASE)
    print(conn_string)

    #Establish Database Connnection
    try:
        conn=pyodbc.connect(conn_string,autocommit=True, ANSI=True)
        print('The connection is successfully establised')
    except:
        print('Somthing is wrong')

    #Disable Pooling
    pyodbc.pooling = False

    # print driver and database info
    print('-ODBC version        =',conn.getinfo(10))
    print( '-DBMS name           =',conn.getinfo(17))
    print( '-DBMS version        =',conn.getinfo(18))
    print( '-Driver name         =',conn.getinfo(6))
    print( '-Driver version      =',conn.getinfo(7))
    print( '-Driver ODBC version =',conn.getinfo(77))

    #Create Sql Queries
    sql = """
    .....................
    """
    #Read table/Sql queries into dataframe
    ###df = pd.read_sql(sql, conn)

    # Alternative approach is to create cursor
    ###cursor = conn.cursor()
    #Execute SQL Queries
    ###cursor.execute(sql)
    #fetch result set rows
    ###for row in cursor:
        ###print(row)
    #close
    ###cursor.close()
    #disconnect
    ##conn.close()
