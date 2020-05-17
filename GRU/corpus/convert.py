relation_dict = {}


def relation_mapping(relation_mapping_file_path):
    # global relation_dict
    relation_mapping_file = open(relation_mapping_file_path, "r")
    mapping_file_sentences = relation_mapping_file.readlines()
    print(mapping_file_sentences)
    for map_sentence in mapping_file_sentences:
        split_word = map_sentence.split("\t")
        relation_dict[split_word[1].strip()] = split_word[2].strip()
    relation_mapping_file.close()


def relation_corpus_file_generator(raw_file_path, raw_relation_file_path, relation_corpus_file_path):
    print(relation_dict)
    raw_sentences_dict = {}
    raw_file = open(raw_file_path, "r")
    raw_sentences = raw_file.readlines()
    for raw_sentence in raw_sentences:
        split_raw_relation_sentence = raw_sentence.split("\t")
        raw_sentences_dict[split_raw_relation_sentence[0].strip()] = [split_raw_relation_sentence[1].strip(),
                                                                      split_raw_relation_sentence[2].strip(),
                                                                      split_raw_relation_sentence[3].replace(" ", "")]
    raw_file.close()
    raw_relation_file = open(raw_relation_file_path, "r")
    raw_relation_sentences = raw_relation_file.readlines()
    raw_relation_file.close()

    relation_corpus_file = open(relation_corpus_file_path, "a")
    for raw_relation_sentence in raw_relation_sentences:
        split_raw_relation_sentence = raw_relation_sentence.split("\t")
        relation = split_raw_relation_sentence[1].strip()
        if relation != "0":
            relation_list = relation.split()
            for rel in relation_list:
                relation_map = relation_dict[rel]
                sentence_dict = raw_sentences_dict[split_raw_relation_sentence[0].strip()]
                sentence = sentence_dict[0] + "\t" + sentence_dict[1] + "\t" + relation_map + "\t" + sentence_dict[2]
                relation_corpus_file.write(sentence)
    relation_corpus_file.close()


raw_train_file_path = "sent_train.txt"
raw_relation_train_file_path = "sent_relation_train.txt"
raw_dev_file_path = "sent_dev.txt"
raw_relation_dev_file_path = "sent_relation_dev.txt"
relation_mapping_file_path = "relation_mapping.txt"
relation_corpus_file_path = "nre_relation_corpus.txt"

relation_mapping(relation_mapping_file_path)
relation_corpus_file_generator(raw_train_file_path, raw_relation_train_file_path, relation_corpus_file_path)
relation_corpus_file_generator(raw_dev_file_path, raw_relation_dev_file_path, relation_corpus_file_path)
