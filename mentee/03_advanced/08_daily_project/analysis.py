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
# - F : 신경망 함수 모듈 (functional)
import torch
import torch.nn.functional as F
# 공통 유틸리티 및 데이터 모듈 임포트
# - DATA : 공통 테스트 데이터
# - ToyTokenizer : 교육용 토크나이저
# - attention : 어텐션 계산 함수
# - cosine : 코사인 유사도 계산 함수
# - write_json : JSON 파일 저장 함수
# - write_md : 마크다운 파일 저장 함수
from common.support import DATA, ToyTokenizer, attention, cosine, write_json, write_md

# 난수 생성 시드 고정으로 연산 재현성 확보
torch.manual_seed(42)
# 연산 스레드 수를 1개로 제한하여 CPU 자원 제어
torch.set_num_threads(1)

# PPT 78~85장의 일일프로젝트. 앞선 계산을 재사용하는 통합 결과입니다.
# 이 코드는 실제 학습된 LLM이 아니며 예시 Token·Vector·Attention·Logit을 결합합니다.
# 일일 프로젝트 통합 분석 수행 메인 함수
def run_analysis():
    # 데이터셋에서 내용(content) 항목들 추출
    texts=[d["content"] for d in DATA]
    # 교육용 토크나이저 초기화
    tokenizer=ToyTokenizer(texts)

    # TODO D20-08-A01: 공통 지식 A의 문장을 Token·ID로 분석하고 Token 수·Context 제한을 기록하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A01
    # raise NotImplementedError("D20-08-A01: TODO를 완성하세요")

    # 데이터셋의 첫 번째 문장 입력 설정
    input_sentence=DATA[0]["content"]
    # 토크나이저를 이용해 최대 길이 12 기준으로 문장 인코딩 수행
    token_report=tokenizer.encode(input_sentence,max_length=12)
    # 유효 토큰 길이가 최대 길이(12) 이하인지 검증
    assert token_report["valid_length"]<=12
    # END D20-08-A01

    # TODO D20-08-A02: 교육용 벡터 A/B/C의 Cosine 표를 만들고 비슷한 방향을 찾아보세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A02
    # raise NotImplementedError("D20-08-A02: TODO를 완성하세요")

    # 교육용 2D 벡터 좌표 딕셔너리 정의
    coords={"A":[1.,0.],"B":[.8,.2],"C":[-1.,0.]}
    # 기준 벡터 A와 나머지 벡터들 간의 코사인 유사도 계산 결과를 담은 딕셔너리 생성
    cosine_table={f"A_{k}":cosine(coords["A"],v) for k,v in coords.items() if k!="A"}
    # END D20-08-A02

    # TODO D20-08-A03: Q/K/V 3×2 Attention에 Causal Mask를 적용하고 행 합=1·미래 가중치=0을 확인하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A03
    # raise NotImplementedError("D20-08-A03: TODO를 완성하세요")

    # 쿼리 텐서 정의 (3행 2열)
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    # 코제럴 마스크(Causal Mask)를 적용하여 어텐션 점수, 가중치, 출력, 마스크 텐서 계산
    scores,weights,attn_out,mask=attention(q,q,q,causal=True)
    # 각 행의 가중치 합이 1에 가까운지 검증
    assert torch.allclose(weights.sum(dim=-1),torch.ones(3))
    # 마스킹된 미래 위치의 가중치가 모두 0인지 검증
    assert torch.all(weights[mask]==0)
    # END D20-08-A03

    # TODO D20-08-A04: Logit→Softmax→Greedy 다음 Token 선택 흐름을 구현하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A04
    # raise NotImplementedError("D20-08-A04: TODO를 완성하세요")

    # 예시 로짓 텐서 정의
    logits=torch.tensor([2.,1.,0.])
    # 로짓에 소프트맥스 함수를 적용하여 확률 분포로 변환
    probs=torch.softmax(logits,dim=0)
    # 가장 높은 확률을 가진 인덱스를 그리디(Greedy) 방식으로 선택
    selected=int(probs.argmax().item())
    # 선택된 인덱스에 해당하는 다음 토큰 라벨 추출
    next_token=["출력A","출력B","출력C"][selected]
    # END D20-08-A04

    # TODO D20-08-A05: 개인 문장 세 개를 바꾸어 입력 길이·padding과 truncation 상태를 비교하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A05
    # raise NotImplementedError("D20-08-A05: TODO를 완성하세요")

    # 사용자 정의 커스텀 문장 리스트 정의
    custom_texts=["실습 기록은 개인 폴더에 보관합니다.", "서버를 프로젝트 폴더에서 실행합니다.", "오류 메시지와 실행 명령을 같이 적습니다."]
    # 커스텀 문장들을 위한 교육용 토크나이저 초기화
    custom_tokenizer=ToyTokenizer(custom_texts)
    # 최대 길이 12 기준으로 배치 토큰화 수행
    custom_batch=custom_tokenizer.batch(custom_texts,max_length=12)
    # 배치 내 각 문장의 유효 토큰 길이 목록 추출
    custom_lengths=[row["valid_length"] for row in custom_batch]
    # END D20-08-A05

    # TODO D20-08-A06: Mask 없는 Attention과 Causal Attention의 차이를 정량 비교하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A06
    # raise NotImplementedError("D20-08-A06: TODO를 완성하세요")

    # 마스크가 없는 일반 어텐션 계산 수행
    _,plain_w,_,_=attention(q,q,q,causal=False)
    # 코제럴 마스크가 적용된 가중치와 일반 가중치 간의 절대값 차이 총합 계산
    mask_diff=float(torch.abs(weights-plain_w).sum())
    # END D20-08-A06

    # TODO D20-08-A07: 데이터·모델·검색 책임을 FastAPI/DB/LLM/RAG/Agent로 명확히 매핑하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A07
    # raise NotImplementedError("D20-08-A07: TODO를 완성하세요")

    # 플랫폼 컴포넌트별 책임 범위 매핑 딕셔너리 정의
    platform_map={
        "FastAPI":"HTTP 입력·검증·응답; 지식 관리 엔드포인트",
        "PostgreSQL":"지식 원본·분류·업데이트 기록(후속 설계)",
        "LLM":"입력 Token에 조건부로 다음 Token 분포 생성",
        "Retriever_RAG":"관련 근거 검색·출처 제공(11/20 이후 구현)",
        "Agent":"도구 선택·단계 실행, 권한 확인 필요(후속 구현)",
    }
    # END D20-08-A07

    # TODO D20-08-A08: 임의 2D 좌표와 Attention Weight가 의미 이해를 증명하지 못한다는 한계 설명을 추가하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A08
    # raise NotImplementedError("D20-08-A08: TODO를 완성하세요")

    # 교육용 모델과 실제 LLM 간의 한계점에 대한 설명 리스트 정의
    limitations=[
        "Tokenizer는 실제 사전학습 모델의 서브워드 Tokenizer가 아니다",
        "벡터는 임의 교육용 좌표로 문장 의미 검색 성능이 없다",
        "Attention은 자연어 모델의 학습된 Head가 아니라 장난감 Score 예제이다",
        "Logit은 예시 세 수치이며 실제 모델 vocabulary 분포가 아니다",
        "RAG·Agent 동작을 구현한 것이 아니며 이후 과목에서 연결한다",
    ]
    # END D20-08-A08

    # TODO D20-08-A09: PPT의 네 단계와 플랫폼 연결 그림을 Markdown으로 완성하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A09
    # raise NotImplementedError("D20-08-A09: TODO를 완성하세요")

    # 마크다운 요약 보고서 작성을 위한 라인 구성
    lines=[
        "# 일일프로젝트: 문장 하나가 LLM 출력이 되기까지", "",
        f"입력: {input_sentence}", "",
        f"- Token 및 Special Token 포함 입력 길이: {token_report['valid_length']}",
        f"- Token IDs: {token_report['input_ids']}",
        f"- 교육용 Cosine A-B / A-C: {cosine_table['A_B']:.4f} / {cosine_table['A_C']:.4f}",
        f"- Causal Mask Attention Row Sum: {weights.sum(dim=-1).tolist()}",
        f"- 다음 Token 예시 선택: {next_token}", "",
        "```mermaid", "flowchart LR", "  U[입력 문장] --> T[Token/ID] --> E[Embedding] --> A[Attention/Mask] --> L[Logit/Softmax] --> O[다음 Token]",
        "  R[후속 Retriever] -. 검색 근거 .-> T", "  DB[(원본 PostgreSQL)] -. 원본/메타데이터 .-> R",
        "  F[FastAPI] -. 입력/응답 .-> U", "```", "",
        "주의: 그림은 개념적 위치 표시이며 오늘 실제 RAG·Agent 서비스를 실행한 것이 아닙니다.",
    ]
    # 마크다운 내용을 지정된 파일로 저장
    write_md(HERE/"outputs"/"one_page_summary.md","\n".join(lines)+"\n")
    # END D20-08-A09

    # TODO D20-08-A10: Token·Cosine·Attention·Logit 검증 결과를 종합 JSON으로 저장하세요.
    # 입력·대상: 앞 실습의 ToyTokenizer·벡터·Attention·Logit 예시를 함께 사용합니다.
    # 완료 확인: outputs/daily_project.json과 one_page_summary.md의 네 단계가 일치해야 합니다.
    # 주의: 실제 LLM·RAG·Agent를 새로 구현했다고 보고하지 않습니다.
    # BEGIN D20-08-A10
    # raise NotImplementedError("D20-08-A10: TODO를 완성하세요")

    # 전체 분석 결과를 통합하여 딕셔너리 구조로 구성하고 JSON 파일로 저장
    write_json(HERE/"outputs"/"daily_project.json",{
        "source_data":DATA,"token":token_report,"cosine":cosine_table,
        "scores_shape":list(scores.shape),"attention_weights":weights.tolist(),
        "row_sums":weights.sum(dim=-1).tolist(),"causal_mask":mask.tolist(),
        "logits":logits.tolist(),"probabilities":probs.tolist(),
        "next_token_demo":next_token,"custom_token_lengths":custom_lengths,
        "masked_unmasked_l1":mask_diff,"platform":platform_map,
        "limitations":limitations,"scope":"교육용 독립 수치 계산; 실제 LLM/RAG/Agent 아님",
    })
    # 실행 결과 요약 콘솔 출력
    print("[08] daily_project.json + one_page_summary.md 생성 완료")
    # END D20-08-A10

"""
토큰화, 2D 벡터 코사인 유사도, 코제럴 마스크 기반 어텐션, 로짓 소프트맥스 및 그리디 샘플링 등 앞선 학습 내용을 종합하는 일일 프로젝트 스크립트이다.
사용자 정의 문장 처리와 정량 비교를 수행하며, LLM 아키텍처와 플랫폼(FastAPI, PostgreSQL, RAG, Agent) 간의 책임 범위를 명확히 매핑한다.
최종적으로 일일 프로젝트의 핵심 요약이 담긴 원페이지 마크다운 문서와 상세 분석 결과가 포함된 종합 JSON 리포트를 생성하도록 구현하였다.
"""