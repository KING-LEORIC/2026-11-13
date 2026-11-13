"""2차원 벡터의 표시 책임만 담당하는 심화용 도구."""
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

# 2D 벡터 시각화 및 저장 함수 정의
# - coords : 시각화할 좌표 딕셔너리 (레이블과 x, y 좌표)
# - filepath : 저장할 파일 경로
def save_vector_plot(coords: dict, filepath):
    # 저장할 파일 경로 객체 생성
    # 파일 저장을 위한 상위 디렉토리 생성 (존재하지 않는 경우 생성)
    path=Path(filepath);path.parent.mkdir(parents=True,exist_ok=True)
    # 6x4 인치 크기의 도화지(Figure)와 축(Axes) 객체 생성
    fig,ax=plt.subplots(figsize=(6,4))
    # 좌표 딕셔너리의 모든 항목(레이블과 좌표) 순회
    for label,(x,y) in coords.items():
        # 산점도(Scatter plot)에 점 찍기
        ax.scatter(x,y,s=70)
        # 점 주변에 레이블 텍스트 주석 추가
        ax.annotate(label,(x,y),xytext=(5,5),textcoords="offset points")
    # 축 레이블 및 그래프 제목 설정 (교육용 2D 벡터)
    ax.set(xlabel="x1",ylabel="x2",title="Educational 2D vectors")
    # 원점 기준 X축 및 Y축 기준선 추가, 투명도가 있는 격자 그리드 표시
    ax.axhline(0,lw=.6);ax.axvline(0,lw=.6);ax.grid(alpha=.2)
    # 레이아웃 간격 자동 조정, 지정된 DPI로 이미지 저장 후 도화지 메모리 해제
    fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)

"""
2차원 벡터의 표시 책임만 담당하는 심화용 도구이자 시각화 모듈이다.
입력받은 2D 벡터 좌표를 Matplotlib을 활용해 산점도와 레이블 형태로 그린다.
원점 기준선과 그리드를 포함한 교육용 벡터 분포 그래프를 지정된 경로에 고해상도 이미지 파일로 저장하도록 구현하였다.
"""