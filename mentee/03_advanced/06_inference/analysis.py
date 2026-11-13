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
# 딥러닝 연산 및 신경망 모듈 임포트
# - torch : PyTorch 텐서 연산 라이브러리
# - F : 신경망 함수 모듈 (functional)
import torch
import torch.nn.functional as F
# 공통 유틸리티 모듈 임포트
# - write_json : JSON 파일 저장 함수
from common.support import write_json

# 난수 생성 시드 고정으로 연산 재현성 확보
torch.manual_seed(42)
# 연산 스레드 수를 1개로 제한하여 CPU 자원 제어
torch.set_num_threads(1)

# 로짓(Logit) 변환, 온도 조절, 샘플링 및 추론 분석 수행 메인 함수
def run_analysis():
    # 예시 로짓 텐서 정의 (출력A, 출력B, 출력C에 해당하는 값)
    logits=torch.tensor([2.0,1.0,0.0])
    # 토큰 라벨 리스트 정의
    labels=["출력A","출력B","출력C"]

    # TODO D20-06-A01: Logit[2,1,0]을 Softmax로 바꾸어 확률의 합이 1인지 확인하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A01
    # raise NotImplementedError("D20-06-A01: TODO를 완성하세요")

    # 로짓에 소프트맥스 함수를 적용하여 확률 분포로 변환
    prob=torch.softmax(logits,dim=0)
    # 확률의 총합이 1에 가까운지 검증
    assert torch.isclose(prob.sum(),torch.tensor(1.0),atol=1e-6)
    # END D20-06-A01

    # TODO D20-06-A02: Temperature 0.5·1.0·2.0에서 확률 분포를 비교하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A02
    # raise NotImplementedError("D20-06-A02: TODO를 완성하세요")

    # 온도(Temperature) 파라미터(0.5, 1.0, 2.0) 변화에 따른 확률 분포 계산 결과를 딕셔너리로 생성
    temperature_results={str(t):torch.softmax(logits/t,dim=0).tolist() for t in (0.5,1.0,2.0)}
    # END D20-06-A02

    # TODO D20-06-A03: Greedy로 다음 Token 후보를 선택하고 학습/추론 차이를 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A03
    # raise NotImplementedError("D20-06-A03: TODO를 완성하세요")

    # 가장 높은 확률을 가진 인덱스를 그리디(Greedy) 방식으로 선택
    selected_index=int(torch.argmax(prob).item())
    # 선택된 인덱스에 해당하는 토큰 라벨 추출
    selected_token=labels[selected_index]
    # 모델 학습 단계와 추론 단계의 차이점 정의 딕셔너리
    training_vs_inference={
        "training":"정답 Token과 예측을 비교해 loss/backward/optimizer.step으로 파라미터 갱신",
        "inference":"가중치를 갱신하지 않고 현재 문맥에서 다음 Token을 선택",
        "autoregressive":"선택한 Token을 문맥에 추가하고 EOS/길이 제한까지 반복",
    }
    # END D20-06-A03

    # TODO D20-06-A04: Top-k(k=2)의 후보 집합과 정규화된 확률을 구하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A04
    # raise NotImplementedError("D20-06-A04: TODO를 완성하세요")

    # 상위 후보 개수 설정
    k=2
    # 확률 분포에서 상위 k개의 값과 인덱스 추출
    values,indices=torch.topk(prob,k)
    # 선택된 상위 확률들의 합이 1이 되도록 정규화
    normalized=(values/values.sum()).tolist()
    # Top-k 후보 집합 딕셔너리 구성
    top_k={"indices":indices.tolist(),"labels":[labels[i] for i in indices],"normalized_probs":normalized}
    # END D20-06-A04

    # TODO D20-06-A05: 같은 seed로 확률 샘플링을 재현하고 Greedy와 차이를 설명하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A05
    # raise NotImplementedError("D20-06-A05: TODO를 완성하세요")

    # 시드가 고정된 난수 생성기 객체 생성
    generator=torch.Generator().manual_seed(42)
    # 확률 분포를 기반으로 다항 분포 샘플링 수행하여 인덱스 추출
    sampled_index=int(torch.multinomial(prob,1,generator=generator).item())
    # 샘플링된 인덱스에 해당하는 토큰 라벨 추출
    sampled_token=labels[sampled_index]
    # END D20-06-A05

    # TODO D20-06-A06: Top-p(p=0.75)에 포함되는 최소 후보 집합을 계산하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A06
    # raise NotImplementedError("D20-06-A06: TODO를 완성하세요")

    # 확률을 내림차순으로 정렬
    sorted_p,sorted_ix=torch.sort(prob,descending=True)
    # 정렬된 확률의 누적 합 계산
    csum=torch.cumsum(sorted_p,dim=0)
    # 누적 확률 임계값(0.75)을 초과하는 항목 마스킹 설정
    remove=csum>0.75
    remove[1:]=remove[:-1].clone()
    remove[0]=False
    keep=~remove
    # 유지할 후보의 인덱스와 확률 추출
    kept_ix=sorted_ix[keep]
    kept_p=sorted_p[keep]
    # Top-p 후보 집합 딕셔너리 구성
    top_p={"indices":kept_ix.tolist(),"labels":[labels[i] for i in kept_ix],
            "normalized_probs":(kept_p/kept_p.sum()).tolist()}
    # END D20-06-A06

    # TODO D20-06-A07: 작은 선형 모델 한 번의 학습 단계에서 파라미터가 바뀌는지 확인하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A07
    # raise NotImplementedError("D20-06-A07: TODO를 완성하세요")

    # 선형 모델 정의 (입력 차원 4, 출력 차원 3, 편향 제외)
    model=torch.nn.Linear(4,3,bias=False)
    # SGD 옵티마이저 정의 (학습률 0.1)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    # 임의의 입력 텐서 및 타겟 정의
    x=torch.tensor([[1.,0.,0.,0.]])
    target=torch.tensor([1])
    # 학습 전 기존 가중치 복제 저장
    old_weight=model.weight.detach().clone()
    # 옵티마이저 기울기 초기화
    optimizer.zero_grad()
    # 순전파 진행 후 크로스 엔트로피 손실 함수 계산
    loss=F.cross_entropy(model(x),target)
    # 역전파(Backward) 수행하여 기울기 계산
    loss.backward()
    # 옵티마이저 스텝을 통해 파라미터 갱신
    optimizer.step()
    # 가중치가 변경되었는지 여부 확인
    changed=not torch.allclose(old_weight,model.weight.detach())
    # 가중치 변경 여부 검증
    assert changed
    # 기울기 계산 비활성화 상태에서 모델 평가 수행
    with torch.no_grad():
        eval_scores=model(x)
    # 학습 단계 검증 결과 딕셔너리 구성
    training_check={"train_loss":float(loss.item()),"weights_changed":changed,
                    "inference_scores":eval_scores.squeeze(0).tolist()}
    # END D20-06-A07

    # TODO D20-06-A08: KV Cache가 저장하는 것·저장하지 않는 것과 비용을 설명하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A08
    # raise NotImplementedError("D20-06-A08: TODO를 완성하세요")

    # KV 캐시(Key-Value Cache)의 역할과 트레이드오프에 대한 설명 노트 작성
    kv_note={
        "stores":"이전에 계산한 Attention Key/Value를 재사용해 반복 계산을 줄임",
        "not":"영구 기억이나 사실성 검증, 학습된 새 지식이 아님",
        "tradeoff":"시퀀스가 길면 Cache 메모리 사용량이 증가",
    }
    # END D20-06-A08

    # TODO D20-06-A09: 확률·선택·학습/추론 차이를 inference_report.json에 기록하세요.
    # 입력·대상: PPT의 예시 Logit [2,1,0]을 사용합니다.
    # 완료 확인: outputs/inference_report.json의 확률합=1과 T별 확률 변화·후보를 확인합니다.
    # 주의: Logit은 확률이 아니며 학습과 추론에서 parameter 변화 여부가 다릅니다.
    # BEGIN D20-06-A09
    # raise NotImplementedError("D20-06-A09: TODO를 완성하세요")

    # 전체 추론 분석 결과를 JSON 파일로 저장
    write_json(HERE/"outputs"/"inference_report.json",{
        "logits":logits.tolist(),"probs":prob.tolist(),"temperature":temperature_results,
        "greedy":{"index":selected_index,"token":selected_token},
        "sampling":{"index":sampled_index,"token":sampled_token},
        "top_k":top_k,"top_p":top_p,"training_vs_inference":training_vs_inference,
        "training_step":training_check,"kv_cache":kv_note,
    })
    # 실행 결과 요약 콘솔 출력
    print("[06]",selected_token,"probs",prob.round(decimals=3).tolist())
    # END D20-06-A09

"""
로짓(Logit) 값을 소프트맥스 함수를 통해 확률로 변환하고, 온도 조절, 그리디, 샘플링, Top-k 및 Top-p 전략을 실습하는 스크립트이다.
모델 학습 단계와 추론 단계의 파라미터 변화 차이를 비교 검증하며, KV 캐시의 동작 원리와 트레이드오프를 확인한다.
최종적으로 상세 분석 결과가 담긴 JSON 리포트를 생성하도록 구현하였다.
"""