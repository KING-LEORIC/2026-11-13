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
# 딥러닝 연산 및 신경망 모듈 임포트
# - torch : PyTorch 텐서 연산 라이브러리
# - nn : 신경망 레이어 및 모듈 정의 패키지
import torch
from torch import nn
# 공통 유틸리티 모듈 임포트
# - write_json : JSON 파일 저장 함수
from common.support import write_json

# 난수 생성 시드 고정으로 연산 재현성 확보
torch.manual_seed(42)
# 연산 스레드 수를 1개로 제한하여 CPU 자원 제어
torch.set_num_threads(1)

# 트랜스포머 인코더 레이어 분석 및 텐서 연산 수행 메인 함수
def run_analysis():
    # 토큰 ID 텐서 정의 (배치 크기 2, 시퀀스 길이 5)
    token_ids=torch.tensor([[2,4,5,3,0],[2,6,7,3,0]],dtype=torch.long)

    # TODO D20-04-A01: Token ID 텐서와 nn.Embedding 출력의 shape (B,T,D)를 구하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A01
    # raise NotImplementedError("D20-04-A01: TODO를 완성하세요")

    # 임베딩 레이어 정의 (총 어휘 수 16, 임베딩 차원 8)
    embedding=nn.Embedding(num_embeddings=16,embedding_dim=8)
    # 토큰 ID를 임베딩 레이어에 통과시켜 벡터 변환 (출력 shape: 배치 크기, 시퀀스 길이, 임베딩 차원)
    x=embedding(token_ids)
    # 토큰 ID 텐서의 shape 기록
    id_shape=list(token_ids.shape)
    # 임베딩 출력 텐서의 shape 기록
    emb_shape=list(x.shape)
    # END D20-04-A01

    # TODO D20-04-A02: EncoderLayer(d_model=8,nhead=2)로 전달하고 shape 보존을 검증하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A02
    # raise NotImplementedError("D20-04-A02: TODO를 완성하세요")

    # 트랜스포머 인코더 레이어 정의
    # - d_model=8 : 모델 차원 8
    # - nhead=2 : 멀티헤드 어텐션 헤드 수 2
    # - dim_feedforward=16 : 피드포워드 신경망 내부 차원 16
    # - batch_first=True : 배치 차원이 첫 번째 오도록 설정 (B, T, D)
    # - dropout=0.0 : 드롭아웃 비율 0으로 설정하여 결정론적 연산 수행
    layer=nn.TransformerEncoderLayer(d_model=8,nhead=2,dim_feedforward=16, batch_first=True,dropout=0.0)
    # 평가 모드로 설정하여 드롭아웃 등 비활성화
    layer.eval()
    # 패딩 위치(0번 토큰)를 판별하기 위한 마스크 생성
    pad_mask=(token_ids==0)
    # 기울기 계산 비활성화 상태에서 인코더 레이어 연산 수행
    with torch.no_grad():
        y=layer(x,src_key_padding_mask=pad_mask)
    # 입력 텐서와 출력 텐서의 shape가 동일한지 검증
    assert x.shape==y.shape
    # 인코더 출력 텐서의 shape 기록
    out_shape=list(y.shape)
    # END D20-04-A02

    # TODO D20-04-A03: Residual·LayerNorm·FFN이 바꾸는 값과 보존하는 shape를 기록하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A03
    # raise NotImplementedError("D20-04-A03: TODO를 완성하세요")

    # 트랜스포머 구성 요소별 역할 및 입출력 shape 정의 딕셔너리
    components={
        "attention":"토큰 사이 정보 가중합",
        "residual":"입력과 변환 출력을 더해 학습 경로를 제공",
        "layernorm":"특징값을 정규화하여 수치 흐름 안정화",
        "ffn":"위치별 feature 변환 (길이 자체를 바꾸지 않음)",
        "input_shape":emb_shape,"output_shape":out_shape,
    }
    # END D20-04-A03

    # TODO D20-04-A04: Padding 위치가 attention의 Key/Value에서 제외되지만 출력 위치가 자동 0이 아님을 확인하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A04
    # raise NotImplementedError("D20-04-A04: TODO를 완성하세요")

    # 패딩 마스크에서 True(패딩 위치)인 좌표 인덱스 추출
    padding_positions=pad_mask.nonzero(as_tuple=False).tolist()
    # 패딩 위치에 해당하는 출력 값들의 절대값 합 계산
    pad_values=y[pad_mask].abs().sum().item()
    # 패딩 마스크의 동작 원리 및 특징에 대한 설명 노트 작성
    padding_note="key padding mask는 해당 KEY 위치를 못 보게 하며 출력의 PAD 위치를 자동 0으로 만들지는 않는다"
    # END D20-04-A04

    # TODO D20-04-A05: d_model=8에서 nhead=3은 나눠지지 않아 실패함을 안전하게 검사하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A05
    # raise NotImplementedError("D20-04-A05: TODO를 완성하세요")

    try:
        # 모델 차원(8)이 어텐션 헤드 수(3)로 나누어떨어지지 않아 예외가 발생하는지 테스트
        nn.MultiheadAttention(embed_dim=8,num_heads=3,batch_first=True)
    except AssertionError:
        # 어설션 예외 발생 시 플래그를 True로 설정
        head_divisibility_error=True
    # 헤드 수 나누기 오류가 정상적으로 발생했음을 검증
    assert head_divisibility_error
    # END D20-04-A05

    # TODO D20-04-A06: 입력 shape가 다른 테스트와 encoder/decoder의 마스크 차이를 기록하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A06
    # raise NotImplementedError("D20-04-A06: TODO를 완성하세요")

    # 임의의 shape를 가진 새로운 입력 텐서 생성 (배치 3, 길이 4, 차원 8)
    new_x=torch.randn(3,4,8)
    # 기울기 계산 비활성화 상태에서 새로운 입력 텐서를 인코더 레이어에 통과시켜 연산 수행
    with torch.no_grad():
        new_y=layer(new_x)
    # 새로운 입출력 shape 기록
    extra_shapes={"new_input":list(new_x.shape),"new_output":list(new_y.shape)}
    # 아키텍처 관련 주의 사항 노트 작성
    architecture_note="TransformerEncoderLayer는 생성형 decoder-only LM 전체가 아니며 causal mask는 별도 설계"
    # END D20-04-A06

    # TODO D20-04-A07: 모든 shape와 Mask 의미를 transformer_report.json에 정리하세요.
    # 입력·대상: 예시 Token IDs는 batch=2, length=5입니다.
    # 완료 확인: outputs/transformer_report.json의 입력·출력 shape가 (2,5,8)인지 확인합니다.
    # 주의: EncoderLayer를 Decoder 기반 LLM 완성본과 혼동하지 않습니다.
    # BEGIN D20-04-A07
    # raise NotImplementedError("D20-04-A07: TODO를 완성하세요")

    # 전체 트랜스포머 인코더 분석 결과를 JSON 파일로 저장
    write_json(HERE/"outputs"/"transformer_report.json",{
        "ids":id_shape,"embedding":emb_shape,"encoder_output":out_shape,
        "pad_key_mask":pad_mask.tolist(),"padding_positions":padding_positions,
        "padding_output_abs_sum":pad_values,"padding_note":padding_note,
        "components":components,"bad_head_rejected":head_divisibility_error,
        "extra_shapes":extra_shapes,"architecture_note":architecture_note,
    })
    # 실행 결과 요약 콘솔 출력
    print("[04]",id_shape,"=>",emb_shape,"=>",out_shape)
    # END D20-04-A07

"""
TransformerEncoderLayer를 활용하여 임베딩, 멀티헤드 어텐션, 잔차 연결(Residual), 층 정규화(LayerNorm), FFN 연산을 실습하는 스크립트이다.
입출력 텐서의 shape 변화를 검증하고, Key Padding Mask의 동작 방식과 어텐션 헤드 수 분할 제약 조건을 확인한다.
최종적으로 트랜스포머 레이어의 구조적 특징과 분석 결과가 담긴 JSON 리포트를 생성하도록 구현하였다.
"""
