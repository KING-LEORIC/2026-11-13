"""Attention 가중치의 Heatmap 시각화: 계산 함수와 분리."""
# 파일 시스템 경로 조작 기능 사용
# - Path : 파일 경로 객체 생성 및 조작
from pathlib import Path
# matplotlib 백엔드 설정 (GUI 백엔드 충돌 방지)
# - matplotlib : 시각화 라이브러리 설정 관리
import matplotlib
matplotlib.use("Agg")
# 2D 플롯 및 그래프 시각화 모듈 임포트
# - plt : 맷플롯립 pyplot 인터페이스
import matplotlib.pyplot as plt

# 어텐션 가중치 히트맵 시각화 및 저장 함수 정의
# - weights : 시각화할 어텐션 가중치 텐서
# - filepath : 저장할 파일 경로
def save_attention_heatmap(weights,filepath):
    # 저장할 파일 경로 객체 생성
    # 파일 저장을 위한 상위 디렉토리 생성 (존재하지 않는 경우 생성)
    path=Path(filepath);path.parent.mkdir(parents=True,exist_ok=True)
    # 4.8x4 인치 크기의 도화지(Figure)와 축(Axes) 객체 생성
    fig,ax=plt.subplots(figsize=(4.8,4))
    # 어텐션 가중치 텐서를 numpy 배열로 변환한 뒤 파란색 계열(Blues) 컬러맵으로 이미지 표시 (값 범위 0~1)
    heat=ax.imshow(weights.detach().cpu().numpy(),vmin=0,vmax=1,cmap="Blues")
    # 축 설정
    # - xticks : X축 눈금 설정 (Key 인덱스 기준 0부터 2까지)
    # - yticks : Y축 눈금 설정 (Query 인덱스 기준 0부터 2까지)
    # - xlabel : X축 레이블 (Key index)
    # - ylabel : Y축 레이블 (Query index)
    ax.set(xticks=range(3),yticks=range(3),xlabel="Key index",ylabel="Query index")
    # 3x3 격자의 모든 행(Query)과 열(Key)을 순회하며 가중치 숫자 표기
    for i in range(3):
        for j in range(3):
            # 각 셀의 중앙에 소수점 둘째 자리까지 가중치 텍스트 주석 추가
            ax.text(j,i,f"{weights[i,j]:.2f}",ha="center",va="center")
    # 컬러바 추가 및 레이아웃 간격 자동 조정
    fig.colorbar(heat,ax=ax);fig.tight_layout()
    # 지정된 DPI로 이미지 저장 후 도화지 메모리 해제
    fig.savefig(path,dpi=130);plt.close(fig)

"""
Attention 가중치의 Heatmap 시각화 책임만 담당하는 시각화 도구이자 계산 분리 모듈이다.
입력받은 어텐션 가중치 텐서를 파란색 계열의 히트맵 이미지로 변환하고, 각 셀마다 소수점 둘째 자리의 가중치 값을 텍스트로 표시한다.
Query 행과 Key 열 기준의 3x3 격자 분포를 명확히 확인할 수 있도록 지정된 경로에 고해상도 이미지 파일로 저장하도록 구현하였다.
"""