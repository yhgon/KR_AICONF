# 한국슈퍼컴퓨팅컨퍼런스(KSC) 2022  
### 29 Sep. 2022
## TUTORIAL - GPU 튜토리얼 (대금)

[류현곤](hryu@nvidia.com)
[survey link](https://forms.gle/kYzZUcv6kQv8bGyw8)

### TIME	PROGRAM
 - 13:30~13:40	GPU Bootcamp튜토리얼 소개  서완석 상무 (NVIDIA)
 - 13:40~14:30	튜토리얼 Part 1 : OpenMP와 OpenACC이용한 GPU 병렬 프로그래밍 소개  류현곤 부장 (NVIDIA)
 - 14:30~15:30	튜토리얼 Part 2 : OpenACC를 이용한 GPU 병렬 프로그래밍 실습  류현곤 부장 (NVIDIA)
 - 15:30~16:00	Break
 - 16:00~17:00	튜토리얼 Part 3 : OpenMP를 이용한 GPU 병렬 프로그래밍 실습  류현곤 부장 (NVIDIA)
 - 17:00~17:30	튜토리얼 Part 4 : 프로파일링을 통한 최적화 실습  류현곤 부장 (NVIDIA)
 



# 실습환경 구축
- colab.research.google.com 회원가입/로그인
- 파일 메뉴에서 열기 실행. 
 - colab에서 실습 jupyter 파일 열기 
  - [colab jupyter file Link](https://colab.research.google.com/drive/1OxJvMwD7FCP1aE8Kb-qGzW0XuSxW0Yt0?usp=sharing)
  - [github jupyter file link](https://github.com/yhgon/KR_AICONF/blob/master/ksc22/KSC22_GPU_tutorial.ipynb)
 - File 메뉴에서  copy in Drive 실행 (개인 실습 환경으로 변경)  
- DevOps 환경 구축 
  - GPU 활성화
  - 연결된 GPU 확인  `!nvidis-smi`
  - NVIDIA HPC SDK 설치
- 실습 시작

