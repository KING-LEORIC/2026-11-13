"""강의 공통 오프라인 계산 도구: 실제 사전학습 LLM이 아닙니다.

교육용 정규표현식 토큰화·작은 벡터·가중치 계산만 제공합니다.
Transformer의 tokenizer/embedding 품질을 이 도구로 평가할 수 없습니다.
"""
# 파일 시스템 경로 조작 기능 사용
# - Path : 파일 경로 객체 생성 및 조작
from pathlib import Path
# JSON 직렬화 및 역직렬화 기능 사용
# - json : JSON 데이터 처리 모듈
import json
# 수학 연산 모듈 임포트
# - math : 제곱근 등 수학 함수 제공
import math
# 정규표현식 패턴 매칭 기능 사용
# - re : 문자열 검색 및 분리 모듈
import re
# 딥러닝 연산 및 텐서 조작 기능 사용
# - torch : PyTorch 텐서 연산 라이브러리
import torch

# 공통 테스트 데이터셋 정의
# - 학습, 환경, 질문 카테고리와 ID, 타이틀, 내용 포함
DATA = [
    {"id":"A", "title":"실습 파일 보관", "content":"실습 파일은 개인 폴더에 보관합니다.", "category":"learning"},
    {"id":"B", "title":"서버 실행", "content":"프로젝트 폴더에서 서버를 실행합니다.", "category":"environment"},
    {"id":"C", "title":"질문 작성", "content":"오류 메시지와 시도한 작업을 함께 적습니다.", "category":"learning"},
]
# 특수 토큰 목록 정의 (패딩, 알 수 없음, 시작, 종료 토큰)
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# 텍스트를 단어 및 기호 단위로 분리하는 유틸리티 함수
# - text : 분리할 대상 문자열
def words(text: str) -> list[str]:
    """교육용: 한글 연속 글자/영문/숫자/문장부호 분리. 서브워드 토크나이저가 아님."""
    # 한글 연속 글자, 영문자, 숫자, 문장부호를 정규표현식으로 각각 매칭하여 리스트로 반환
    # - 주의: 교육용이며 실제 서브워드 토크나이저가 아님
    return re.findall(r'[가-힣]+|[A-Za-z]+|\d+|[^\w\s]', text)

# 교육용 토크나이저 클래스 정의
# - BERT tokenizer 대체가 아니며, 오프라인 개념 확인용 규칙 기반 어휘표 구현
class ToyTokenizer:
    """BERT tokenizer 대체가 아닌, 오프라인 개념 확인용 규칙 기반 어휘표."""
    # 토크나이저 초기화 메서드
    # - corpus : 어휘 사전 구축을 위한 코퍼스 데이터
    def __init__(self, corpus):
        # 코퍼스 내 모든 텍스트에서 단어를 추출하고 중복을 제거한 뒤 정렬하여 고유 단어 집합 생성
        wordset=sorted({word for line in corpus for word in words(line)})
        # 특수 토큰과 고유 단어 집합을 결합하여 어휘 사전(단어 -> 인덱스) 구축
        self.vocab={tok:i for i,tok in enumerate(SPECIAL + wordset)}
        # 인덱스로 단어를 조회할 수 있는 역방향 사전 구축
        self.reverse={i:t for t,i in self.vocab.items()}

    # 텍스트 인코딩 메서드
    # - text : 인코딩할 대상 문자열
    # - max_length : 최대 토큰 길이 제한
    # - padding_to : 패딩을 채울 목표 길이
    def encode(self, text: str, max_length=None, padding_to=None):
        # 입력 텍스트를 단어 리스트로 분리
        raw=words(text)
        # 시작 토큰([BOS])과 종료 토큰([EOS])을 앞뒤로 추가
        tokens=["[BOS]"]+raw+["[EOS]"]
        # 자름(Truncation) 여부 플래그 초기화
        truncated=False
        # 최대 길이가 지정된 경우 처리
        if max_length is not None:
            # BOS/EOS 토큰을 포함하기 위해 최소 길이는 2 이상이어야 함을 검증
            if max_length<2: raise ValueError("BOS/EOS를 포함하려면 max_length>=2")
            # 토큰 길이가 최대 길이를 초과하는 경우
            if len(tokens)>max_length:
                # 지정된 최대 길이에 맞춰 자르고 끝에 [EOS] 토큰 부착
                tokens=tokens[:max_length-1]+["[EOS]"]
                truncated=True
        # 패딩을 제외한 유효 토큰 길이 기록
        valid=len(tokens)
        # 패딩 목표 길이가 지정된 경우 처리
        if padding_to is not None:
            # 패딩 목표 길이는 유효 토큰 길이보다 크거나 같아야 함을 검증
            if padding_to<valid: raise ValueError("padding_to must be >= valid length")
            # 부족한 길이만큼 [PAD] 토큰 추가
            tokens=tokens+["[PAD]"]*(padding_to-valid)
        # 각 토큰을 어휘 사전의 ID로 변환 (사전에 없는 경우 [UNK] 토큰 ID 매핑)
        ids=[self.vocab.get(t,self.vocab["[UNK]"]) for t in tokens]
        # 패딩 토큰 위치는 0, 실제 토큰 위치는 1로 설정하는 어텐션 마스크 생성
        mask=[0 if t=="[PAD]" else 1 for t in tokens]
        # 인코딩 결과 딕셔너리 반환
        return {"tokens":tokens,"input_ids":ids,"attention_mask":mask,
                "valid_length":valid,"truncated":truncated,"original_length":len(raw)+2}

    # 배치 인코딩 메서드
    # - texts : 인코딩할 텍스트 리스트
    # - max_length : 최대 토큰 길이 제한
    def batch(self, texts, max_length=None):
        # 리스트 내 각 텍스트를 개별 인코딩
        items=[self.encode(t,max_length=max_length) for t in texts]
        # 배치 내 인코딩 결과 중 가장 긴 시퀀스 길이 계산
        max_size=max(len(i["input_ids"]) for i in items)
        # 가장 긴 길이에 맞춰 모든 텍스트에 패딩 적용 후 리스트 반환
        return [self.encode(t,max_length=max_length,padding_to=max_size) for t in texts]

# LLM 문맥 예산(Context Budget) 계산 함수
# - window : 전체 허용 토큰 창 크기
# - parts : 영역별 할당 토큰 수 딕셔너리
# - output_reserved : 출력 예약 토큰 수
def context_budget(window, parts:dict, output_reserved:int):
    # 입력 영역별 사용된 총 토큰 수 계산
    used=sum(parts.values())
    # 예산 계산 결과 딕셔너리 반환 (창 크기, 사용량, 예약량, 남은 용량, 예산 적합 여부)
    return {"window":window,"input_used":used,"output_reserved":output_reserved,
            "remaining":window-used-output_reserved,
            "fits":used+output_reserved<=window}

# 코사인 유사도(Cosine Similarity) 계산 함수
# - a : 첫 번째 벡터
# - b : 두 번째 벡터
def cosine(a,b):
    # 입력 a를 32비트 부동소수점 텐서로 변환
    a=torch.as_tensor(a,dtype=torch.float32)
    # 입력 b를 32비트 부동소수점 텐서로 변환
    b=torch.as_tensor(b,dtype=torch.float32)
    # PyTorch 내장 코사인 유사도 함수를 호출하여 스칼라 값으로 반환
    return float(torch.nn.functional.cosine_similarity(a,b,dim=0))

# 어텐션 점수 및 가중치 계산 함수
# - q : 쿼리(Query) 텐서
# - k : 키(Key) 텐서
# - v : 밸류(Value) 텐서
# - causal : 코제럴 마스크(Causal Mask) 적용 여부
def attention(q,k,v,causal=True):
    # 쿼리, 키, 밸류를 32비트 부동소수점 텐서로 변환
    q=torch.as_tensor(q,dtype=torch.float32)
    k=torch.as_tensor(k,dtype=torch.float32)
    v=torch.as_tensor(v,dtype=torch.float32)
    # 스케일드 닷 프로덕트 어텐션 점수 계산 (Q와 K 전치행렬의 곱 나누기 키 차원 크기의 제곱근)
    scores=q@k.T/math.sqrt(q.shape[-1])
    # 상단 삼각형 영역을 True로 채우는 코제럴 마스크 생성
    mask=torch.triu(torch.ones_like(scores,dtype=torch.bool),diagonal=1)
    # 코제럴 마스크가 활성화된 경우 마스크 위치를 음의 무한대로 치환한 뒤 소프트맥스 적용, 아니면 일반 점수에 소프트맥스 적용
    weights=torch.softmax(scores.masked_fill(mask,float('-inf')) if causal else scores,dim=-1)
    # 점수, 어텐션 가중치, 가중치와 밸류의 행렬곱 결과(출력), 마스크 텐서 반환
    return scores,weights,weights@v,mask

# JSON 파일 저장 함수
# - path : 저장할 파일 경로
# - data : 저장할 파이썬 객체 데이터
def write_json(path, data):
    # 경로 객체 생성 및 상위 디렉토리 자동 생성 (존재하지 않는 경우)
    p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
    # 데이터를 들여쓰기 2칸의 UTF-8 형식 JSON 문자열로 파일에 기록
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

# 마크다운 파일 저장 함수
# - path : 저장할 파일 경로
# - text : 저장할 마크다운 문자열
def write_md(path,text):
    # 경로 객체 생성 및 상위 디렉토리 자동 생성 (존재하지 않는 경우)
    p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
    # 텍스트를 UTF-8 형식으로 파일에 기록
    p.write_text(text,encoding='utf-8')

# JSON 파일 읽기 함수
# - path : 읽어올 파일 경로
def read_json(path):
    # 지정된 경로의 JSON 파일을 읽어 파이썬 객체로 역직렬화하여 반환
    return json.loads(Path(path).read_text(encoding='utf-8'))
