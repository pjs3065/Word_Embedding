# -*- coding: utf-8 -*-

import pickle
import DataPreProc as dp
import MyTransitionParser as tp
import MyFeatureModel as fm

def make_corpus_and_model():
    model = fm.FeatureModel() # FeatureModel 인스턴스 생성
    parser = tp.TransitionParse(model) # TransitionParse 인스턴스 생성, stack, queue, predict, transition_sequence 메모리 생성

    cr = dp.CorpusReader()
    cr.set_file("/home/kang/Development/wangki/parsing/data/train.txt", "/home/kang/Development/wangki/parsing/data/train.out")

    num_data = 0
    # 데이터 계속 읽는 부분
    while True:
        # 코퍼스 한개씩 처리
        data = cr.get_next() # data = 한 코퍼스
        if data is None:
            break

        num_data += 1
        total_sample = parser.make_gold_corpus(data)
        cr.write_out_data(total_sample)

    print('데이터 생성 : ' + str(num_data) + '문장')
    cr.close_file()

    return model

def save_model():
    f_model = make_corpus_and_model() # feature 정보들이 추가된 데이터
    f = open('/home/kang/Development/wangki/parsing/model/f_model.dat', 'wb')
    pickle.dump(f_model, f)
    f.close()

save_model()

