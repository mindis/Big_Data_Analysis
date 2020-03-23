# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import psycopg2
import configparser
import os
import sys

def read_credential(filename):
    parser = configparser.ConfigParser()
    try:
        parser.read(filename)
        
        host= parser.get('Redshift','host')
        user =parser.get('Redshift','user')
        password=parser.get('Redshift','password')
        dbname=parser.get('Redshift','dbname')
        port=parser.get('Redshift','port') 
        return host, user, password, dbname, port
    except OSError as e:
        print('we encountered error {}'.format(e))

    
def redshif_connector(dbname, host, port, user, password):
   
    try:
        conn = psycopg2.connect(dbname=dbname, host=host, port=int(port), user=user, password=password)
        print('Connection is Successful')
        return conn
    except OSError as e:
        print('we encountered error {}'.format(e))
        
      

if __name__=='__main__':
    #if os.getcwd() !='H:\\Projectcsv':
        #os.chdir('H:\\Projectcsv')
          
    filename =sys.argv[1]
    host, user, password, dbname, port = read_credential(filename)
    
    redshift_connector(dbname, host, port, user,password)
    
    
    
    