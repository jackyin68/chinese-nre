# bio data generator for nre, KG

import codecs
import os
import re

import tensorflow as tf
import itertools
from GRU.BIO.similarity import edit_distance_str
from GRU.BIO.nre import nre_evaluate

flags = tf.app.flags
flags.DEFINE_string("raw_file", os.path.join("rawdata", "article-bio.txt"), "Path for raw data")
flags.DEFINE_string("nre_file", os.path.join("rst", "article-nre.txt"), "Path for nre data")
flags.DEFINE_string("nre_clean_file", os.path.join("rst", "article-clean-nre.txt"), "Path for clean nre data")
flags.DEFINE_string("nre_rst_file", os.path.join("rst", "article-nre-rst.txt"), "Path for nre result")
FLAGS = tf.app.flags.FLAGS


def load_article_sentences(path):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.strip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            sentence.append(line)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def save_to_file(bio_file, sentence, tags_result):
    for i, word in enumerate(sentence):
        word = word.strip()
        line = word + " " + tags_result[i] + "\n"
        bio_file.write(line)
    bio_file.write("\n")


keywords = ["方方", "汪芳"]
keyflags = ["B-PER"]


def match_keywords(words):
    words = "".join(words)
    keywords_rst = []
    bool_match_keywords = False
    for keyword in keywords:
        pattern = re.compile(r'{keyword}'.format(keyword=keyword))
        word_list = pattern.findall(str(words))
        keywords_rst.extend(word_list)
    if len(keywords_rst) >= 1:
        bool_match_keywords = True
    return bool_match_keywords


def match_enough_person(w_flags):
    w_flags = "".join(w_flags)
    word_flag_rst = []
    bool_match_enough_person = False
    for keyflag in keyflags:
        pattern = re.compile(r'{keyflag}'.format(keyflag=keyflag))
        flag_list = pattern.findall(str(w_flags))
        word_flag_rst.extend(flag_list)
    if len(word_flag_rst) >= 2:
        bool_match_enough_person = True
    return bool_match_enough_person


def sentence_composer(words, w_flags):
    sentence_list = []
    word_list = []
    keyword_list = []
    word = ""
    words_str = "".join(words)
    for i, flag in enumerate(w_flags):
        if flag == 'B-PER':
            word = word + words[i]
        elif flag == 'I-PER':
            word = word + words[i]
        elif flag == 'E-PER':
            word = word + words[i]
            if word in keywords:
                keyword_list.append(word)
            else:
                word_list.append(word)
            word = ""
    for keyword in keyword_list:
        for word in word_list:
            sentence_list.append(keyword + " " + word + " " + words_str)

    for keyword_tuple in itertools.combinations(keyword_list, 2):
        if keyword_tuple[0] != keyword_tuple[1]:
            sentence_list.append(keyword_tuple[0] + " " + keyword_tuple[1] + " " + words_str)
    return sentence_list


def evaluate_sentences(sentences, nre_file):
    for i, sentence in enumerate(sentences):
        words = []
        w_flags = []
        for word_flag in sentence:
            word_flag_split = word_flag.split(" ")
            words.append(word_flag_split[0])
            w_flags.append(word_flag_split[1])
        if match_keywords(words) and match_enough_person(w_flags):
            sentence_list = sentence_composer(words, w_flags)
            for sent in sentence_list:
                nre_file.write(sent)
                nre_file.write('\n')


def nre_file_generator(raw_file_path, nre_file_path):
    sentences = load_article_sentences(raw_file_path)
    nre_file = open(nre_file_path, "w")
    evaluate_sentences(sentences, nre_file)
    nre_file.close()


def nre_file_remove_duplicate(nre_file_path, nre_clean_file_path):
    nre_file = open(nre_file_path, "r")
    sentences = nre_file.readlines()
    nre_file.close()

    nre_clean_file = open(nre_clean_file_path, "w")
    sentences_len = len(sentences)
    remove_sen_index = []
    for i in range(sentences_len - 1):
        for j in range(i + 1, sentences_len):
            distan_info = edit_distance_str(sentences[i], sentences[j])
            print(distan_info['Distance'], distan_info['Similarity'])
            if distan_info['Similarity'] > 0.9:
                remove_sen_index.append(j)
    for i, sentence in enumerate(sentences):
        if i not in remove_sen_index:
            nre_clean_file.write(sentence)
    nre_clean_file.close()


def main(_):
    raw_file_path = FLAGS.raw_file
    nre_file_path = FLAGS.nre_file
    nre_clean_file_path = FLAGS.nre_clean_file
    nre_rst_file_path = FLAGS.nre_rst_file
    nre_file_generator(raw_file_path, nre_file_path)  # nre file generator
    nre_file_remove_duplicate(nre_file_path, nre_clean_file_path)
    nre_evaluate(nre_clean_file_path, nre_rst_file_path)  # nre result


if __name__ == "__main__":
    tf.app.run(main)
