**Naver sentiment movie corpus를 활용한 데이터 분석**
- - -
## 기능

Naver sentiment movie corpus(NSMC)를 활용해서 영화 리뷰에 대한 데이터를 분석합니다.  
* 데이터 전처리(한글 제외 문자 분리, 리뷰 토큰화, 문서 벡터 생성)
* 토큰 빈도 분석, 리뷰 유사도 분석, 리뷰어 감정 분석
* 유사한 리뷰 추출, 동일한 감정 분석 결과 추출
* 긍부정 일치, 불일치 정확도 계산  
- - -

## 데이터

분석에 필요한 데이터는 해당 [github](https://github.com/e9t/nsmc)에서 받을 수 있습니다.  
분석에 사용할 데이터는 `ratings_train.txt`와 `ratings_test.txt`로 각각 학습 데이터와 테스트 데이터입니다.
- - -

## 생성 파일

`ratings_train.txt`의 각 리뷰를 토큰화한 파일 `train_token.txt`와  
`ratings_test.txt`의 각 리뷰를 토큰화한 파일 `test_token.txt`가 생성됩니다.  
`ratings_train.txt`의 각 리뷰를 문서벡터화한 `train_vector.txt`와  
`ratings_test.txt`의 각 리뷰를 문서벡터화한 `test_vector.txt`가 생성됩니다.  
  
**프로그램을 여러 번 실행할 경우 파일을 생성하는 데에 많은 시간이 소요되므로, 최초 실행이 완료되면 코드 내에 `아래 두 줄의 코드는 최초 1회만 실행`이라는 주석 아래의 코드를 주석 처리하면 더 빠르고 효율적인 분석이 가능합니다.**
- - -

## 개발 현황

* 데이터 전처리(한글 제외 문자 분리, 리뷰 토큰화, 문서 벡터 생성)
* 토큰 빈도 분석, 리뷰 유사도 분석, 리뷰어 감정 분석
* 유사한 리뷰 추출, 동일한 감정 분석 결과 추출
* `ratings_test.txt`에 대한 긍부정 일치 개수 계산   
* Confusion Matrix를 이용한 긍부정 일치, 불일치 정확도 계산   
- - -

## 개발 예정 
* [SVM](https://www.cs.cornell.edu/people/tj/svm_light/)(Support Vector Machine) 이용하여 분류 및 성능 평가  
* bigram, sentencePiece 토큰화 기법을 이용하여 문서벡터 재구성, 분류 및 성능 평가
- - -
