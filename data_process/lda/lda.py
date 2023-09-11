'''
python lda.py random_doc_sample.jsonl
'''
import argparse
import json
import numpy as np
import pprint
from nltk.tokenize import word_tokenize
import os
import string
import collections
import gensim
import tqdm
import ftfy

stopwords = set('for a of the and to in is as '.split())
translator = str.maketrans('','', string.punctuation + '‘“’”')
_MALLET_BIN = 'Mallet/bin/mallet'


def strip_punctuation(string_in):
    return string_in.translate(translator)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('jsonl_in')

    return parser.parse_args()


def get_tokens(text, lower=True, remove_punct=True):
    if lower:
        text = text.lower()

    if remove_punct:
        text = strip_punctuation(text)

    text = ftfy.fix_text(' '.join(word_tokenize(text)))
    return [t for t in text.split() if t not in stopwords]


def main():
    global _MALLET_BIN

    args = parse_args()

    docs = []

    with open(args.jsonl_in) as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            docs.append(get_tokens(' '.join(data['text_list'])))

    word_count = collections.Counter()
    for tok in docs:
        word_count.update(set(tok))

    print(len(docs))
    print(.1*len(docs))
    # keep vocab item if word is seen more than 25 times and is in less than 10% of docs
    valid_set = set([w for w,c in word_count.items() if c >= 25 and c <= .1 * len(docs)])
    print('Vocab size: {}'.format(len(valid_set)))

    docs = [[t for t in toks if t in valid_set] for toks in docs]
    dictionary = gensim.corpora.Dictionary(docs)
    docs = [dictionary.doc2bow(t) for t in docs]

    topics = 30
    model = gensim.models.wrappers.ldamallet.LdaMallet(
        _MALLET_BIN,
        corpus=docs,
        num_topics=topics,
        id2word=dictionary,
        optimize_interval=10,
        iterations=2000)

    topics = model.show_topics(formatted=False, num_topics=topics, num_words=100)
    with open('topics.tsv', 'w') as f:
        for t in topics:
            f.write('{}\t{}\n'.format(t[0], ' '.join([wt[0] for wt in t[1]])))

    doc_topics = model.load_document_topics()
    dt_mat = []
    for dt in doc_topics:
         dt_mat.append(np.array([f for t, f in dt]))
    dt_mat = np.vstack(dt_mat)
    np.save('doc_topics.npy', dt_mat)


if __name__ == '__main__':
    main()