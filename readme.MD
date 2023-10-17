# pthon_project_carsuri


## 프로젝트 소개
django와 딥러닝을 이용한 차량파손이미지 분류 서비스
<br>

## 개발 기간
* 23.05.22일 - 23.06.13일

### 맴버구성


### 개발 환경
- `Python 3.10.9`
- `JDK 11.0.16.1`
- **IDE** : PyCharm Community Edition 2023.1.2
- **Framework** : Django 4.2
- **Database** : Maria DB 10.10.2, Amazon RDS
- **ORM** : Django ORM
- **Deep learning** : TensorFlow GPU 2.6, nVidia CUDA 11.2

## 주요 기능
#### 메인 페이지
- 동적 하위 카테고리 생성
- 유저 업로드 파일 타입 검사
#### 결과 페이지
- 이미지 별 부위, 파손 타입 출력
- 차량 모델과 부위, 파손 타입으로 예상 수리비 산출
- 유저 위치기반 카센터 추천 목록 제공(구글 api 연동)
