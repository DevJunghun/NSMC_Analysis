import re
import math
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        documents = [line.split('\t')[1:] for line in f.read().splitlines()]
        documents = documents[1:]
    return documents

def load_token(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        token_list = list(set([token.replace("\n", "") for token in f.readlines()]))
    return token_list

def load_vector(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        doc_list = [line.split()[2:] for line in f.read().splitlines()]
        vector_list = list()
        for i in doc_list:
            for j in i:
                temp = j.split(':')
                temp_list = list()
                temp_list.append(int(temp[0]))
                temp_list.append(float(temp[1]))
                vector_list.append(temp_list)
    return vector_list

def tokenizer_train(docs):
    with open('train_token.txt', 'w', encoding='utf-8') as f:
        for i in range(len(docs)):
            for j in range(len(docs[i][0]) - 1):
                f.write(docs[i][0][j:j+2])
                f.write('\n')

def tokenizer_test(docs):
    with open('test_token.txt', 'w', encoding='utf-8') as f:
        for i in range(len(docs)):
            for j in range(len(docs[i][0]) - 1):
                f.write(docs[i][0][j:j+2])
                f.write('\n')

def tf(token, doc):
    return doc[0].count(token)

def idf(token, docs):
    df = 0
    for doc in docs:
        if token in doc[0]: df += 1
    return math.log(len(docs) / (df + 1))

def vectorizer_train(docs, token):
    with open('train_vector.txt', 'w', encoding='utf-8') as f:
        for i in range(len(docs)):
            tf_value, idf_value = 0, 0
            for j in range(len(docs[i][0]) - 1):
                tf_value = tf(docs[i][0][j:j+2], docs[i])
                idf_value = idf(docs[i][0][j:j+2], docs)
                tfidf = tf_value * idf_value
                f.write(f"{docs[i][-1]} {token.index(docs[i][0][j:j+2]) + 1}:{tfidf:.15f}")
            f.write("\n")

def vectorizer_test(docs, token):
    with open('test_vector.txt', 'w', encoding='utf-8') as f:
        for i in range(len(docs)):
            tf_value, idf_value = 0, 0
            for j in range(len(docs[i][0]) - 1):
                tf_value = tf(docs[i][0][j:j+2], docs[i])
                idf_value = idf(docs[i][0][j:j+2], docs)
                tfidf = tf_value * idf_value
                f.write(f"{docs[i][-1]} {token.index(docs[i][0][j:j+2]) + 1}:{tfidf:.15f}")
            f.write("\n")

def clean_doc(doc):
    doc = re.sub("[0-9]{2,3}-[0-9]{4}-[0-9]{4}", "", doc) # 전화번호 분리
    doc = re.sub("^[_0-9a-zA-Z]*@[0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)+", "", doc) # 이메일 분리
    doc = re.sub("[0-9]{6}-[0-9]{7}", "", doc) # 주민번호 분리
    doc = re.sub("[^가-힣]{2, }", "", doc) # 한글 이외 문자열 분리
    doc = re.sub("[A-Za-zㄱ-ㅎㅏ-ㅣ]", "", doc) # 영어, 한글 자음과 모음 분리
    doc = re.sub("[-=+,#/\?:^$.@*\"※~&%!』\\|\(\)\[\]\<\>`\'…]", "", doc) # 특수문자 분리
    
    return doc

def blank_doc(doc):
    return doc.replace(" ", "_")

if __name__ == "__main__":
    train_docs = load_file('.\\ratings_train.txt')
    test_docs = load_file('.\\ratings_test.txt')

    # 아스키 문자열 분리, 한글 이외 문자열 분리
    for i in range(len(train_docs)):
        train_docs[i][0] = clean_doc(train_docs[i][0])
        train_docs[i][0] = blank_doc(train_docs[i][0])
    for i in range(len(test_docs)):
        test_docs[i][0] = clean_doc(test_docs[i][0])
        test_docs[i][0] = blank_doc(test_docs[i][0])

    # 토큰화 후 파일로 저장
    tokenizer_train(train_docs)
    tokenizer_test(test_docs)

    train_token = load_token("train_token.txt")
    test_token = load_token("test_token.txt")
    token = sorted(list(set(train_token + test_token)))

    # 아래 코드는 최초 1회만 실행, train_vector.txt와 test_vector.txt 파일 생성 시 주석 처리할 것
    vectorizer_train(train_docs, token)
    vectorizer_test(test_docs, token)

    train_vector = load_vector('train_vector.txt')[:50000]
    test_vector = load_vector('test_vector.txt')
    
    num = int(input("1 ~ 50000 정수를 입력해주세요: "))
    check_vector = test_vector[num - 1]
    sim_list = list()
    for i in range(len(train_vector)):
        if cosine_similarity([train_vector[i]], [check_vector]) > 0.99999999:
            sim_list.append((i, train_vector[i]))
    sim_list = sorted(random.sample(sim_list, 5), key=lambda x:x[0])
    test_review = load_file('.\\nsmc-master\\nsmc-master\\ratings_train.txt')[num - 1]
    emotion_list = list()
    print()
    print(f"{num}번째 영화평 : {' '.join(test_review[:1])}")
    print()
    print(f"{num}번째 영화평과 가장 유사한 5개의 영화평은 다음과 같습니다.\n")
    for n, i in enumerate(sim_list):
        sim_review = load_file('.\\nsmc-master\\nsmc-master\\ratings_test.txt')[i[0] - 1]
        emotion_list.append(sim_review[1])
        print(f"{n + 1} : {' '.join(sim_review[:1])}")
    print()
    if emotion_list.count('1') > emotion_list.count('0'):
        print(f"5개 영화평의 긍부정 개수가 많은 값은 1로 {emotion_list.count('1')}개입니다. 영화평이 영화에 대해 긍정적입니다.")
    else:
        print(f"5개 영화평의 긍부정 개수가 많은 값은 0으로 {emotion_list.count('0')}개입니다. 영화평이 영화에 대해 부정적입니다.")