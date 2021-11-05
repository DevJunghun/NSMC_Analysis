**Naver sentiment movie corpus를 활용한 데이터 분석**
- - -
## 기능

Naver sentiment movie corpus(NSMC)를 활용해서 영화 리뷰에 대한 데이터를 분석합니다.  
* 데이터 전처리(한글 제외 문자 분리, 리뷰 토큰화, 문서 벡터 생성)
* 토큰 빈도 분석, 리뷰 유사도 분석, 리뷰어 감정 분석
* 유사한 리뷰 추출, 동일한 감정 분석 결과 추출
- - -

## 데이터

분석에 필요한 데이터는 해당 [github](https://github.com/e9t/nsmc)에서 받을 수 있습니다.  
분석에 사용할 데이터는 `ratings.train.txt`와 `ratings.test.txt`로 각각 학습 데이터와 테스트 데이터입니다.
- - -

## 생성 파일

`ratings.train.txt`의 각 리뷰를 토큰화한 파일 `train_token.txt`와  
`ratings.test.txt`의 각 리뷰를 토큰화한 파일 `test.token.txt`가 생성됩니다.  
`ratings.train.txt`의 각 리뷰를 문서벡터화한 `train_vector.txt`와  
`ratings.test.txt`의 각 리뷰를 문서벡터화한 `test_vector.txt`가 생성됩니다.
- - -

## 개발 현황

* 데이터 전처리(한글 제외 문자 분리, 리뷰 토큰화, 문서 벡터 생성)
* 토큰 빈도 분석, 리뷰 유사도 분석, 리뷰어 감정 분석
- - -

## 개발 예정

* 유사한 리뷰 추출, 동일한 감정 분석 결과 추출
* (기능 필요 시 추가)
- - -
