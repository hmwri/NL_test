import re

import MeCab
from collections import Counter
from tqdm import tqdm



def preprocess(texts,unk=4,predict=False):
    indexes = {}
    tagger = MeCab.Tagger("-Owakati")
    if predict:
        text = removeSymbols(texts)
        return tagger.parse(text).split()

    ntexts = []
    i = 0
    for text in tqdm(texts):
        text = removeSymbols(text)
        node = tagger.parseToNode(text)
        while node:
            text = node.surface
            if text:
                idxs = indexes.get(text)
                if idxs is None:
                    indexes[text] = [i]
                elif len(idxs) < unk:
                    indexes[text].append(i)
                ntexts.append(text)
                i += 1
            node = node.next

    replaceUnk(ntexts, indexes,unk)
    return ntexts


def removeSymbols(text):
    text = re.sub(r"https?://[\w!\?/\+\-_~=;\.,\*&@#\$%\(\)'\[\]]+", " ", text)
    text = re.sub(r"[0-9]+-[0-9]+-[0-9]+T[0-9]+:[0-9]+:[0-9]+\+[0-9]+", " ", text)
    text = re.sub(
        r"[_－―─＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？・,/『』【】「」→←○《》≪≫\n\u3000]+",
        " ", text)
    return text

def replaceUnk(texts, indexes,unk):
    counter = Counter(texts)
    for k, v in tqdm(counter.items()):
        if v <= unk:
            print(k)
            for i in indexes[k]:
                texts[i] = "<UNK>"


def replace(texts, str, target):
    for i, t in enumerate(texts):
        if t == str:
            texts[i] = target


def make_corpus(texts):
    word_to_id = {}
    id_to_word = {}
    corpus = []
    for text in tqdm(texts):
        id = word_to_id.get(text)
        if id is None:
            id = len(word_to_id)
            word_to_id[text] = id
            id_to_word[id] = text
        corpus.append(id)
    return corpus ,word_to_id, id_to_word



def to_word(ids, id_to_word):
    text = ""
    for id in ids:
        id = id.item()
        text += id_to_word[id]
    return text
