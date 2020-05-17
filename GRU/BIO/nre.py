import tensorflow as tf
import numpy as np
from GRU import network


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def nre_evaluate(nre_clean_file_path, nre_rst_file_path):
    pathname = "../model/ATT_GRU_model-9000"
    wordembedding = np.load('../data/vec.npy')
    test_settings = network.Settings()
    test_settings.vocab_size = 16693
    test_settings.num_classes = 12
    test_settings.big_num = 1

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            names_to_vars = {v.op.name: v for v in tf.global_variables()}
            saver = tf.train.Saver(names_to_vars)
            saver.restore(sess, pathname)

            print('reading word embedding data...')
            vec = []
            word2id = {}
            f = open('../origin_data/vec.txt', encoding='utf-8')
            content = f.readline()
            content = content.strip().split()
            dim = int(content[1])
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [(float)(i) for i in content]
                vec.append(content)
            f.close()
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            print('reading relation to id')
            relation2id = {}
            id2relation = {}
            f = open('../origin_data/relation2id.txt', 'r', encoding='utf-8')
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])
                id2relation[int(content[1])] = content[0]
            f.close()

            nre_clean_file = open(nre_clean_file_path, encoding='utf-8')
            lines = nre_clean_file.readlines()

            for line in lines:
                en1, en2, sentence = line.strip().split()
                print("实体1: " + en1)
                print("实体2: " + en2)
                print(sentence)
                relation = 0
                en1pos = sentence.find(en1)
                if en1pos == -1:
                    en1pos = 0
                en2pos = sentence.find(en2)
                if en2pos == -1:
                    en2post = 0
                output = []
                # length of sentence is 70
                fixlen = 70
                # max length of position embedding is 60 (-60~+60)
                maxlen = 60

                # Encoding test x
                for i in range(fixlen):
                    word = word2id['BLANK']
                    rel_e1 = pos_embed(i - en1pos)
                    rel_e2 = pos_embed(i - en2pos)
                    output.append([word, rel_e1, rel_e2])

                for i in range(min(fixlen, len(sentence))):

                    word = 0
                    if sentence[i] not in word2id:
                        # print(sentence[i])
                        # print('==')
                        word = word2id['UNK']
                        # print(word)
                    else:
                        word = word2id[sentence[i]]
                    output[i][0] = word
                test_x = []
                test_x.append([output])

                # Encoding test y
                label = [0 for i in range(len(relation2id))]
                label[0] = 1
                test_y = []
                test_y.append(label)

                test_x = np.array(test_x)
                test_y = np.array(test_y)

                test_word = []
                test_pos1 = []
                test_pos2 = []

                for i in range(len(test_x)):
                    word = []
                    pos1 = []
                    pos2 = []
                    for j in test_x[i]:
                        temp_word = []
                        temp_pos1 = []
                        temp_pos2 = []
                        for k in j:
                            temp_word.append(k[0])
                            temp_pos1.append(k[1])
                            temp_pos2.append(k[2])
                        word.append(temp_word)
                        pos1.append(temp_pos1)
                        pos2.append(temp_pos2)
                    test_word.append(word)
                    test_pos1.append(pos1)
                    test_pos2.append(pos2)

                test_word = np.array(test_word)
                test_pos1 = np.array(test_pos1)
                test_pos2 = np.array(test_pos2)

                prob, accuracy = test_step(test_word, test_pos1, test_pos2, test_y)
                prob = np.reshape(np.array(prob), (1, test_settings.num_classes))[0]
                print("关系是:")
                top3_id = prob.argsort()[-3:][::-1]
                for n, rel_id in enumerate(top3_id):
                    print("No." + str(n + 1) + ": " + id2relation[rel_id] + ", Probability is " + str(prob[rel_id]))
