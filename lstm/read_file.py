import os
import glob
import re

def read_corpus_file(path, n=None):
    filenames = [p for p in glob.glob(path+"/**", recursive=True)
          if re.search('^(?!(LICENSE)|(README)|CHANGES).*(txt)$', p)]

    print("LICENCE.txt" in filenames)

    texts = []
    if n is not None:
        filenames = filenames[:n]
    for filename in filenames:
        f = open(filename, 'r')
        data = f.read()
        texts.append(data)


    return texts
