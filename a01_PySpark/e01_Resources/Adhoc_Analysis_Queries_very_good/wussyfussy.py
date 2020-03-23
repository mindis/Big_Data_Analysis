import itertools as it
from functools import partial
import collections as cl
import pandas as pd
import csv
import sys

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process




import csv
class Fuz:
    def __init__(self, filename):
        self.filename = filename
    
    def load_data(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f) 
            result =list(''.join(row) for row in reader if row[8]=='Generic')
        return result
    
    def combinatric(self):
        res = self.load_data()
        result = it.combinations(res, 2)
        #result = list(result)
        return result 

    def counter(self):
        res = self.load_data()
        return cl.Counter(res).most_common()
    
    def transformer(self):
        result = self.combinatric()
        res= {item: fuzz.ratio(item[0],item[1]) for item in result}
        return res     


if __name__ == '__main__':
    filename = sys.argv[1]
    fz = Fuz(filename)
    result = fz.transformer()
    
    for item in result:
        if result[item] > 90:
            print(item)