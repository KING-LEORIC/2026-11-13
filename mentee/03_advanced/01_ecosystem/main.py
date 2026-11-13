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
# 공통 유틸리티 모듈 임포트
# - write_json : JSON 파일 저장 함수
# - write_md : 마크다운 파일 저장 함수
from common.support import write_json, write_md

# 강의 PPT 15장: 개념도에 실제 구현한 것과 향후 구현할 것을 구분합니다.
# LLM을 실제 호출하지 않습니다. diagram.md는 그림을 그리는 Mermaid 텍스트입니다.
# 아키텍처 개념도 및 메타데이터 생성 메인 함수
def main():

    # TODO D20-01-A01: FastAPI·PostgreSQL·LLM·응답의 역할 네 가지를 짧은 문장으로 작성하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A01
    # raise NotImplementedError("D20-01-A01: TODO를 완성하세요")

    # 핵심 컴포넌트별 역할 정의 딕셔너리
    roles = {
        "FastAPI": "HTTP 요청을 검증하고 함수·응답을 연결한다",
        "PostgreSQL": "원본 지식·분류를 관계형 테이블에 저장한다",
        "LLM": "문맥과 입력을 바탕으로 다음 Token을 생성한다",
        "Response": "호출한 Client에 JSON 결과를 전달한다",
    }
    # END D20-01-A01

    # TODO D20-01-A02: 요청→API→데이터·LLM→응답 연결을 올바른 순서의 화살표로 표현하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A02
    # raise NotImplementedError("D20-01-A02: TODO를 완성하세요")

    # 컴포넌트 간 데이터 흐름 및 연결 관계(엣지) 리스트 정의
    edges = [
        ("Client", "FastAPI", "HTTP 요청"),
        ("FastAPI", "PostgreSQL", "원본 지식 조회·수정"),
        ("PostgreSQL", "FastAPI", "원본 데이터"),
        ("FastAPI", "LLM", "입력·문맥(후속 구현)"),
        ("LLM", "Response", "생성 결과(후속 구현)"),
        ("Response", "Client", "JSON"),
    ]
    # END D20-01-A02

    # TODO D20-01-A03: 검색 단계 Retriever와 근거·출처 검증 위치를 연결 목록에 추가하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A03
    # raise NotImplementedError("D20-01-A03: TODO를 완성하세요")

    # 검색 및 검증 관련 엣지 추가
    edges.extend([
        ("FastAPI", "Retriever", "질문·검색 조건(후속 구현)"),
        ("Retriever", "PostgreSQL", "원본·메타데이터 참조 가능"),
        ("Retriever", "LLM", "검색된 근거(후속 구현)"),
        ("LLM", "EvidenceCheck", "답변·출처 검증(후속 구현)"),
    ])
    # Retriever 및 EvidenceCheck 역할 추가 정의
    roles["Retriever"] = "관련 근거를 찾는 검색 계층(11/20 후속 구현)"
    roles["EvidenceCheck"] = "답변의 근거·출처 일치 여부를 확인한다"
    # END D20-01-A03

    # TODO D20-01-A04: 저장과 생성의 책임을 별도 문장으로 기록하고, 오늘 구현 여부를 구분하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A04
    # raise NotImplementedError("D20-01-A04: TODO를 완성하세요")

    # 저장 및 생성 책임 범위와 구현 여부 정의 딕셔너리
    responsibility = {
        "storage": "PostgreSQL은 지식 저장·관계를 책임진다",
        "generation": "LLM은 다음 Token의 분포와 출력을 책임진다",
        "implemented_today": ["역할 개념도 작성"],
        "not_implemented": ["새 RAG 검색", "새 Agent 실행", "실제 LLM API 호출"],
    }
    # END D20-01-A04

    # TODO D20-01-A05: Agent 도구의 조회·수정 권한 차이를 표현하고, 허가 없는 변경을 금지하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A05
    # raise NotImplementedError("D20-01-A05: TODO를 완성하세요")

    # 도구 권한 관리 역할 추가
    roles["ToolPermission"] = "조회/변경 권한을 실행 전에 애플리케이션이 확인한다"
    # 도구 실행 전 승인 및 권한 확인 관련 엣지 추가
    edges.append(("Agent", "ToolPermission", "도구 실행 전 승인 확인(후속 구현)"))
    edges.append(("ToolPermission", "FastAPI", "허가된 범위의 요청만 전달"))
    # END D20-01-A05

    # TODO D20-01-A06: 외부 문서의 명령 문구를 상위 지시처럼 신뢰하면 안 되는 경계를 표시하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A06
    # raise NotImplementedError("D20-01-A06: TODO를 완성하세요")

    # 신뢰 경계(Trust Boundary) 정책 정의 딕셔너리
    trust_boundary = {
        "untrusted": ["외부 문서 본문", "검색된 근거의 지시문"],
        "rule": "외부 텍스트는 데이터로 취급하며 시스템 권한을 부여하지 않는다",
        "scope": "설계 설명만; Agent·보안 시스템은 실제 구현하지 않음",
    }
    # END D20-01-A06

    # TODO D20-01-A07: 완성한 연결 관계를 Mermaid 그림과 JSON 요약으로 저장하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-A07
    # raise NotImplementedError("D20-01-A07: TODO를 완성하세요")

    # 전체 노드 이름 정렬 및 고유 ID 매핑 생성
    node_names = sorted({n for a,b,_ in edges for n in (a,b)})
    node_id = {name:f"N{i}" for i,name in enumerate(node_names)}
    # Mermaid 문법의 흐름도(flowchart) 라인 구성
    lines = ["flowchart LR"]
    lines += [f'  {node_id[name]}["{name}"]' for name in node_names]
    lines += [f'  {node_id[a]} -->|"{label}"| {node_id[b]}' for a,b,label in edges]
    # 생성된 Mermaid 텍스트를 마크다운 파일로 저장
    write_md(HERE/"outputs"/"architecture.md", "# AI 지식 플랫폼 개념도\n\n```mermaid\n"+"\n".join(lines)+"\n```\n")
    # 아키텍처 관련 모든 정보를 JSON 파일로 저장
    write_json(HERE/"outputs"/"architecture.json", {
        "roles":roles,"edges":[list(x) for x in edges],
        "responsibility":responsibility,"trust_boundary":trust_boundary,
    })
    # END D20-01-A07
    # 완료 메시지 콘솔 출력
    print("[01] architecture.md / architecture.json 생성 완료")

# 스크립트 직접 실행 시 메인 함수 호출
if __name__=="__main__":
    main()

"""
AI 지식 플랫폼의 아키텍처 개념도와 역할 분담 구조를 정의하고 문서화하는 스크립트이다.
FastAPI, PostgreSQL, LLM 등의 컴포넌트 역할과 요청/응답 흐름을 정의하고, 향후 구현될 Retriever 및 보안·권한 경계를 설정한다.
최종적으로 그 결과를 Mermaid 마크다운 다이어그램과 JSON 구조화 파일로 출력하도록 구현하였다.
"""
