# 파일 시스템 경로 조작 기능 사용
# - Path : 파일 경로 객체 생성 및 조작
from pathlib import Path
# 시스템 경로 제어 기능 사용
# - sys : 모듈 검색 경로 관리
import sys

# 현재 파일 기준 디렉토리 절대 경로 정의
HERE = Path(__file__).resolve().parent
# 상위 작업 폴더 경로 정의
LEVEL_ROOT = HERE.parent
# sys.path에 상위 경로 추가
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 딥러닝 연산 및 텐서 조작 기능 사용
# - torch : PyTorch 텐서 및 신경망 모듈
# - F : 신경망 함수 모듈 (functional)
import torch
import torch.nn.functional as F
# matplotlib 백엔드 설정 (GUI 백엔드 충돌 방지)
# - matplotlib : 시각화 라이브러리
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 커스텀 시각화 및 유틸리티 함수 임포트
# - save_vector_port : 벡터 시각화 저장 함수
# - cosine : 코사인 유사도 계산 함수
# - write_json : JSON 파일 저장 함수
from plotting import save_vector_plot
from common.support import cosine, write_json

# 난수 생성 시드 고정으로 재현성 확보
torch.manual_seed(42)
# 연산 스레드 수를 1개로 제한
torch.set_num_threads(1)

# 벡터 유사도 분석 및 임베딩 처리 메인 함수
def run_analysis():
    # PPT 33장과 같은 교육용 2D 벡터. SentenceTransformer가 만든 실제 의미 벡터가 아닙니다.
    # 교육용 2D 벡터 좌표 딕셔너리 정의
    # - PPT 33장과 같은 교육용 2D 벡터
    # - SentenceTransformer가 만든 실제 의미 벡터가 아님
    coords = {"A":[1.0,0.0],"B":[0.8,0.2],"C":[-1.0,0.0]}

    # TODO D20-03-A01: A-B와 A-C 벡터의 Cosine Similarity를 계산하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A01
    # raise NotImplementedError("D20-03-A01: TODO를 완성하세요")

    # 벡터 A와 B 사이의 코사인 유사도 계산
    s_ab=cosine(coords["A"],coords["B"])
    # 벡터 A와 C 사이의 코사인 유사도 계산
    s_ac=cosine(coords["A"],coords["C"])
    # END D20-03-A01

    # TODO D20-03-A02: 점수의 부호·범위를 보고 어느 벡터 방향이 유사한지 판별하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A02
    # raise NotImplementedError("D20-03-A02: TODO를 완성하세요")

    # 유사도 결과 해석 및 평가 딕셔너리 정의
    interpretation={"A_B":"방향이 비슷함" if s_ab>0.9 else "차이 있음",
                    "A_C":"반대 방향" if s_ac<-0.99 else "차이 있음",
                    "note":"코사인이 높다는 사실만으로 RAG 답변의 사실성을 보장하지 않는다"}
    # END D20-03-A02

    # TODO D20-03-A03: 2차원 후보 D를 추가해 가장 유사한 벡터를 점수순으로 정리하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A03
    # raise NotImplementedError("D20-03-A03: TODO를 완성하세요")

    # 새로운 2차원 후보 벡터 D 추가
    coords["D"]=[0.5,0.8]
    # 기준 벡터 A와 각 후보 벡터 간의 코사인 유사도를 계산하고 유사도 기준 내림차순 정렬
    ranked=sorted(((key,cosine(coords["A"],val)) for key,val in coords.items() if key!="A"), key=lambda x:(-x[1],x[0]))
    # END D20-03-A03

    # TODO D20-03-A04: 벡터를 행렬로 쌓은 Tensor shape (개수, 차원)를 기록하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A04
    # raise NotImplementedError("D20-03-A04: TODO를 완성하세요")

    # 좌표 딕셔너리 값을 PyTorch 텐서 행렬로 변환
    matrix=torch.tensor(list(coords.values()),dtype=torch.float32)
    # 단일 벡터 shape와 전체 행렬 shape 기록
    shapes={"vector":list(matrix[0].shape), "matrix":list(matrix.shape)}
    # END D20-03-A04

    # TODO D20-03-A05: Token ID를 nn.Embedding에 넣고 출력 shape를 확인하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A05
    # raise NotImplementedError("D20-03-A05: TODO를 완성하세요")

    # 임베딩 레이어 정의 (어휘 수 12, 임베딩 차원 4)
    embedding=torch.nn.Embedding(12,4)
    # 토큰 ID 텐서 정의
    ids=torch.tensor([[2,5,3,0],[2,7,8,3]],dtype=torch.long)
    # 토큰 ID를 임베딩 레이어에 통과시켜 벡터 변환
    token_vectors=embedding(ids)
    # 출력된 토큰 벡터의 shape 기록
    token_shape=list(token_vectors.shape)
    # END D20-03-A05

    # TODO D20-03-A06: 임의 Token 벡터 평균을 계산하고 의미 검색용 모델과 차이를 기록하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A06
    # raise NotImplementedError("D20-03-A06: TODO를 완성하세요")

    # 시퀀스 차원(dim=1)을 기준으로 토큰 벡터들의 평균 계산 (Mean Pooling)
    toy_pooled=token_vectors.mean(dim=1)
    # 풀링된 벡터 shape 기록
    pooled_shape=list(toy_pooled.shape)
    # 교육용 토크나이저/임베딩과 실제 의미 검색 모델 간의 한계점 기록
    limitations=["임의 초기화된 Token 벡터의 평균은 학습된 Sentence Embedding이 아니다",
                 "2D 투영에서 가까워 보여도 원래 고차원 이웃 관계가 보존된다고 보장되지 않는다"]
    # END D20-03-A06

    # TODO D20-03-A07: 2D 점 그래프와 cosine_report.json을 outputs에 저장하세요.
    # 입력·대상: PPT의 A/B/C 2차원 좌표를 사용합니다.
    # 완료 확인: outputs/cosine_report.json, vectors.png를 확인합니다.
    # 주의: 수치 비교는 모델의 문장 의미 이해나 RAG 정확성 평가가 아닙니다.
    # BEGIN D20-03-A07
    # raise NotImplementedError("D20-03-A07: TODO를 완성하세요")

    # 2D 벡터 분포 그래프를 이미지 파일로 저장
    save_vector_plot(coords,HERE/"outputs"/"vectors.png")
    # 분석 결과 전체를 JSON 파일로 저장
    write_json(HERE/"outputs"/"cosine_report.json",{
        "vectors":coords,"cosine":{"AB":s_ab,"AC":s_ac},
        "interpretation":interpretation,"ranked":ranked,"shapes":shapes,
        "token_shape":token_shape,"pooled_shape":pooled_shape,"limitations":limitations,
    })
    # 실행 결과 요약 콘솔 출력
    print(f"[03] AB={s_ab:.4f} AC={s_ac:.4f}, shape={shapes['matrix']}")
    # END D20-03-A07

"""
2차원 벡터 간의 코사인 유사도를 계산하고, PyTorch 텐서 연산 및 임베딩(Embedding) 구조를 실습하는 스크립트이다.
벡터 간 방향성 판별 및 유사도 순위 정렬을 수행하며, 임의 토큰 임베딩의 풀링 결과와 실제 의미 검색 모델 간의 차이점을 분석한다.
최종적으로 2D 벡터 분포 시각화 이미지와 상세 분석 결과가 담긴 JSON 리포트를 생성하도록 구현하였다.
"""