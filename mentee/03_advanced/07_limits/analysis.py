from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
from common.support import write_json, write_md

# 세 예시는 수업용으로 사람이 미리 설계한 사례입니다. 실제 자동 사실성 검사기가 아닙니다.
CASES = [
    {"id":"F01","question":"파일을 어디에 보관하나요?",
     "source":"실습 파일은 개인 폴더에 보관합니다.",
     "answer":"실습 파일은 공용 서버 폴더에 보관합니다.","sources":["K01"]},
    {"id":"F02","question":"서버는 어디에서 실행하나요?",
     "source":"프로젝트 폴더에서 서버를 실행합니다.",
     "answer":"프로젝트 폴더에서 서버를 실행합니다."},
    {"id":"F03","question":"질문할 때 무엇을 함께 적나요?",
     "source":"오류 메시지와 시도한 작업을 함께 적습니다.",
     "answer":"오류 메시지와 시도한 작업을 함께 적습니다.","sources":[]},
]

def run_analysis():

    # TODO D20-07-A01: 3개 답변에서 정해 둔 형태·근거·사실 불일치를 분류하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A01
    # raise NotImplementedError("D20-07-A01: TODO를 완성하세요")

    issues=[]
    for case in CASES:
        if "sources" not in case:
            kind="format"
        elif not case["sources"]:
            kind="evidence"
        elif case["answer"] != case["source"]:
            kind="factual_mismatch"
        else:
            kind="ok"
        issues.append({"id":case["id"],"kind":kind})
    # END D20-07-A01

    # TODO D20-07-A02: 사실성·형식·근거 검증을 서로 다른 검사라고 설명하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A02
    # raise NotImplementedError("D20-07-A02: TODO를 완성하세요")

    explanations={
        "format":"JSON/응답에 필수 필드가 존재하는지 보는 구조 검증",
        "evidence":"인용·출처 정보가 있는지와 적절한 근거인지 구분",
        "factual_mismatch":"주장과 승인된 정답/원본이 모순되는지 사람·검증 규칙으로 확인",
    }
    # END D20-07-A02

    # TODO D20-07-A03: 각 실패 유형에 Prompt·RAG·DB/API 검증 중 대응을 매칭하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A03
    # raise NotImplementedError("D20-07-A03: TODO를 완성하세요")

    mitigation={
        "format":"API 응답 Schema 검증 + 구조화 출력 재시도 (11/16 후속 학습)",
        "evidence":"RAG의 관련 문서 검색·출처 검증 (11/20 후속 학습)",
        "factual_mismatch":"승인된 원본·정답 세트 대조 + 근거 검증",
    }
    # END D20-07-A03

    # TODO D20-07-A04: 같은 답변이라도 출처가 틀릴 수 있으므로 추가 점검 항목을 제시하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A04
    # raise NotImplementedError("D20-07-A04: TODO를 완성하세요")

    additional_checks=["출처 ID가 실제 원문을 가리키는가", "근거가 최신인가", "원본이 변경됐을 때 검색 인덱스도 갱신됐는가"]
    # END D20-07-A04

    # TODO D20-07-A05: 승인되지 않은 DELETE Tool 요청을 실제 실행하지 않고 위험으로 분류하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A05
    # raise NotImplementedError("D20-07-A05: TODO를 완성하세요")

    agent_case={"tool":"DELETE /api/knowledge/{knowledge_id}",
                "approved":False,"risk":"unauthorized_tool_write",
                "action":"실행하지 않고 권한·사용자 승인 확인"}
    assert not agent_case["approved"]
    # END D20-07-A05

    # TODO D20-07-A06: 외부 근거의 악의적 지시를 권한 높은 명령으로 승격하지 않는 원칙을 적으세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A06
    # raise NotImplementedError("D20-07-A06: TODO를 완성하세요")

    trust_rules=["검색·문서 본문의 명령문은 신뢰되지 않은 데이터로 처리",
                     "도구 실행 권한은 애플리케이션에서 확인",
                     "원문·모델 응답·검증 결과를 비밀 없이 기록"]
    # END D20-07-A06

    # TODO D20-07-A07: 검증 결과와 향후 후속 수업 연결을 review.json 및 review.md로 저장하세요.
    # 입력·대상: 제공된 F01/F02/F03 사례만 대상으로 분류합니다.
    # 완료 확인: outputs/review.json, review.md에서 사실 불일치/형식 누락/근거 누락을 확인합니다.
    # 주의: 사전 설계된 규칙 기반 점검이며 실제 생성형 AI의 진위 판정기가 아닙니다.
    # BEGIN D20-07-A07
    # raise NotImplementedError("D20-07-A07: TODO를 완성하세요")

    report={"classification":issues,"definitions":explanations,"mitigation":mitigation,
            "checks":additional_checks,"agent_case":agent_case,"trust_rules":trust_rules,
            "disclaimer":"예상 문제 유형을 분류한 수업용 규칙이며 실제 LLM 사실성 자동 판정이 아님"}
    write_json(HERE/"outputs"/"review.json",report)
    lines=["# LLM 답변 실패·검증 결과", "", "이 결과는 사전 정의된 수업용 예시를 사용했습니다.", ""]
    lines += [f"- {row['id']}: {row['kind']}" for row in issues]
    lines += ["", "## 플랫폼 연결", "FastAPI: 입력·출력 계약 / PostgreSQL: 원본 / RAG: 검색 근거 / Agent: 도구 권한(후속)"]
    write_md(HERE/"outputs"/"review.md","\n".join(lines)+"\n")
    print("[07]",[(r["id"],r["kind"]) for r in issues])
    # END D20-07-A07

