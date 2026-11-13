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
# sys.path에 상위 경로 설정
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 수학 연산 모듈 임포트
# - math : 제곱근 등 수학 함수 제공
import math
# 딥러닝 연산 및 신경망 모듈 임포트
# - torch : PyTorch 텐서 연산 라이브러리
import torch
# matplotlib 백엔드 설정 (GUI 백엔드 충돌 방지)
# - matplotlib : 시각화 라이브러리 설정 관리
import matplotlib
# 2D 플롯 및 그래프 시각화 모듈 임포트
# - plt : 맷플롯립 pyplot 인터페이스
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 커스텀 시각화 및 유틸리티 함수 임포트
# - save_attention_heatmap : 어텐션 히트맵 시각화 저장 함수
# - write_json : JSON 파일 저장 함수
from plotting import save_attention_heatmap
from common.support import write_json

# 난수 생성 시드 고정으로 연산 재현성 확보
torch.manual_seed(42)
# 연산 스레드 수를 1개로 제한하여 CPU 자원 제어
torch.set_num_threads(1)

# 셀프 어텐션 및 코제럴 마스크(Causal Mask) 분석 수행 메인 함수
def run_analysis():
    # 쿼리 텐서 정의 (3행 2열)
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    # 키와 밸류 텐서를 쿼리와 동일하게 복제하여 초기화
    k,v=q.clone(),q.clone()

    # TODO D20-05-A01: Scaled Dot-Product Q@K.T/sqrt(d_k)로 (3,3) Score를 계산하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A01
    # raise NotImplementedError("D20-05-A01: TODO를 완성하세요")

    # 스케일드 닷 프로덕트 어텐션 점수 계산
    # - 쿼리와 키의 행렬곱을 수행한 뒤 키의 차원 크기의 제곱근으로 나누어 스케일링
    scores=q@k.T/math.sqrt(q.shape[-1])
    # 점수 텐서의 shape가 (3, 3)인지 검증
    assert scores.shape==(3,3)
    # END D20-05-A01

    # TODO D20-05-A02: 미래 Key를 차단하는 upper-triangle Causal Mask를 작성하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A02
    # raise NotImplementedError("D20-05-A02: TODO를 완성하세요")

    # 상단 삼각형 영역을 True로 채우는 코제럴 마스크 생성 (대각선 기준 위쪽 차단)
    mask=torch.triu(torch.ones(3,3,dtype=torch.bool),diagonal=1)
    # 마스크 위치의 점수 값을 음의 무한대로 치환하여 소프트맥스 적용 시 0이 되도록 처리
    masked_scores=scores.masked_fill(mask,float("-inf"))
    # END D20-05-A02

    # TODO D20-05-A03: Score를 Softmax Weight로 바꾸고 Mask된 칸=0, 행 합=1을 확인하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A03
    # raise NotImplementedError("D20-05-A03: TODO를 완성하세요")

    # 마스킹된 점수에 소프트맥스를 적용하여 어텐션 가중치 산출
    weights=torch.softmax(masked_scores,dim=-1)
    # 각 행의 가중치 합 계산
    row_sums=weights.sum(dim=-1)
    # 각 행의 가중치 합이 1에 가까운지 검증
    assert torch.allclose(row_sums,torch.ones(3),atol=1e-6)
    # 마스킹된 위치의 가중치가 정확히 0인지 검증
    assert torch.all(weights[mask]==0)
    # END D20-05-A03

    # TODO D20-05-A04: Mask 없는 Softmax 결과와 Mask 있는 결과를 비교하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A04
    # raise NotImplementedError("D20-05-A04: TODO를 완성하세요")

    # 마스크가 없는 일반 점수에 소프트맥스 적용
    plain_weights=torch.softmax(scores,dim=-1)
    # 마스크 유무에 따른 가중치 차이의 절대값 합 계산
    diff=float(torch.abs(plain_weights-weights).sum())
    # END D20-05-A04

    # TODO D20-05-A05: Causal Weight를 Query행·Key열 기준으로 Heatmap에 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A05
    # raise NotImplementedError("D20-05-A05: TODO를 완성하세요")

    # 어텐션 가중치 히트맵을 지정된 경로에 이미지 파일로 저장
    save_attention_heatmap(weights,HERE/"outputs"/"attention_heatmap.png")
    # END D20-05-A05

    # TODO D20-05-A06: 동일 입력에 대해 nhead=1과 nhead=2의 가중치 shape를 비교하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A06
    # raise NotImplementedError("D20-05-A06: TODO를 완성하세요")

    # 멀티헤드 비교를 위한 입력 텐서 정의 (배치 1, 길이 3, 차원 4)
    x=torch.tensor([[[1.,0.,0.,1.],[0.,1.,1.,0.],[1.,1.,0.,0.]]])
    head_weights={}
    # 헤드 수가 1개일 때와 2개일 때 각각 반복 수행
    for heads in (1,2):
        torch.manual_seed(42)
        # 멀티헤드 어텐션 모듈 정의
        mha=torch.nn.MultiheadAttention(4,heads,batch_first=True,dropout=0.0)
        # 평가 모드로 설정
        mha.eval()
        # 기울기 계산 비활성화 상태에서 멀티헤드 어텐션 연산 수행
        with torch.no_grad():
            out,w=mha(x,x,x,attn_mask=mask,need_weights=True,average_attn_weights=False)
        # 가중치 텐서의 shape가 (배치, 헤드수, 쿼리길이, 키길이)인지 검증
        assert w.shape==(1,heads,3,3)
        # 마스킹된 위치의 가중치가 모두 0인지 검증
        assert torch.all(w[:,:][...,mask]==0)
        # 헤드별 가중치 shape와 첫 번째 쿼리의 가중치 목록 저장
        head_weights[str(heads)]={"shape":list(w.shape),"query0":w[0,:,0,:].tolist()}
    # END D20-05-A06

    # TODO D20-05-A07: Weight를 모델 의사결정의 유일한 원인으로 해석할 수 없는 이유를 적으세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A07
    # raise NotImplementedError("D20-05-A07: TODO를 완성하세요")

    # 어텐션 가중치 해석의 한계점에 대한 설명 노트 작성
    interpretation_limit=[
        "Attention은 Value 혼합 비율이지 모든 후속 FFN·Residual·Head 계산을 포함하지 않는다",
        "학습되지 않은 장난감 Head의 가중치에서 자연어 이해를 추론할 수 없다",
        "Head 수 변화 외에도 투영 파라미터·분할 차원이 달라 직접 성능 비교가 아니다",
    ]
    # END D20-05-A07

    # TODO D20-05-A08: Score·Causal Mask·Weight·행 합과 Output을 JSON으로 저장하세요.
    # 입력·대상: PPT Q/K/V 3×2 예제와 Bool Causal Mask를 사용합니다.
    # 완료 확인: outputs/attention_report.json의 masked weights=0, row_sums=1을 확인합니다.
    # 주의: Softmax 이전 scores와 이후 weights를 구분하고, 행=Query·열=Key로 표시합니다.
    # BEGIN D20-05-A08
    # raise NotImplementedError("D20-05-A08: TODO를 완성하세요")

    # 어텐션 가중치와 밸류 텐서의 행렬곱을 통해 최종 출력 계산
    output=weights@v
    # 전체 어텐션 분석 결과를 JSON 파일로 저장
    write_json(HERE/"outputs"/"attention_report.json",{
        "scores":scores.tolist(),"causal_mask":mask.tolist(),
        "weights":weights.tolist(),"row_sums":row_sums.tolist(),
        "output":output.tolist(),"no_mask_weights":plain_weights.tolist(),
        "masked_vs_unmasked_l1":diff,"multi_head":head_weights,
        "limits":interpretation_limit,
    })
    # 실행 결과 요약 콘솔 출력
    print("[05]",weights.round(decimals=2).tolist(),"row_sums",row_sums.tolist())
    # END D20-05-A08

"""
Scaled Dot-Product Attention과 Causal Mask의 동작 원리를 분석하고 시각화하는 스크립트이다.
쿼리, 키, 밸류 간의 어텐션 점수 계산 및 소프트맥스를 거친 가중치 산출 과정을 실습하며, 마스킹에 따른 제약과 멀티헤드 어텐션의 구조적 차이를 검증한다.
최종적으로 어텐션 가중치 히트맵 이미지와 상세 분석 결과가 담긴 JSON 리포트를 생성하도록 구현하였다.
"""
