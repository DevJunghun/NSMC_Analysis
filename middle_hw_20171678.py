import re
import math

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        documents = [line.split('\t')[1:] for line in f.read().splitlines()]
        documents = documents[1:]
    return documents

def load_token(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        token_list = list(set([token.replace("\n", "") for token in f.readlines()]))
        return token_list

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

    # vectorizer_train(train_docs, token)
    # vectorizer_test(test_docs, token)