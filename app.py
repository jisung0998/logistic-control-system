import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. 인공지능 백엔드 엔진 구축 (의사결정트리 학습)
np.random.seed(42)
data_size = 100
X = np.column_stack((np.random.uniform(50, 200, data_size), np.random.uniform(30, 100, data_size), 
                     np.random.uniform(10, 80, data_size), np.random.uniform(5, 50, data_size)))
y = np.array([2 if X[i,0]>150 and X[i,2]<30 else (1 if X[i,0]>120 or X[i,1]>80 or X[i,2]<45 else 0) for i in range(data_size)])

dec_tree = DecisionTreeClassifier(max_depth=3, criterion='entropy')
dec_tree.fit(X, y)

# 2. 물류 가상 관제 대시보드 UI 화면 디자인
st.set_page_config(page_title="물류 과밀화 관제 시스템", layout="wide")

st.title("🚚 지능형 도심 물류센터 실시간 과밀화 관제 시스템")
st.markdown("---")
st.markdown("### 📡 데이터 기반 실시간 IoT 센서 및 교통망 네트워크 상황 설정")

col_in1, col_in2, col_in3, col_in4 = st.columns(4)

with col_in1:
    inbound = st.slider("📥 시간당 화물 반입량 (단위: 톤)", 50, 200, 110)
with col_in2:
    dock = st.slider("🏗️ 상하차 도크 점유율 (단위: %)", 30, 100, 60)
with col_in3:
    speed = st.slider("🚗 진입로 평균 통행 속도 (단위: km/h)", 10, 80, 55)
with col_in4:
    density = st.slider("📈 주변 도로 밀집도 (단위: 대/km)", 5, 50, 20)

# 3. 입력값 동적 바인딩 및 실시간 AI 연산 수행
current_data = np.array([[inbound, dock, speed, density]])
prediction = dec_tree.predict(current_data)[0]
proba = dec_tree.predict_proba(current_data)[0]

st.markdown("---")
st.subheader("📊 인공지능(AI) 분석 및 리스크 진단 결과")

res_col1, res_col2 = st.columns(2)

with res_col1:
    if prediction == 0:
        st.success("## 🟢 [상태] 정상 가동 중 (Normal)")
        st.info("💡 **알림:** 현재 모든 지표가 안정적입니다. 물류 순환이 원활하게 통제되고 있습니다.")
    elif prediction == 1:
        st.warning("## 🟡 [상태] 지연 주의 예보 (Warning)")
        st.info("💡 **알림:** 화물량 증가 혹은 진입로 정체 가능성이 감지됩니다. 모니터링을 강화하세요.")
    else:
        st.error("## 🔴 [상태] 심각한 과밀화 마비 (Overload)")
        st.info("🚨 **경고:** 화물 폭증 및 교통 네트워크 마비 상태 돌입! 즉시 주변 우회 배송 차량 스케줄링을 가동하세요.")

with res_col2:
    st.write("**⚠️ [의사결정트리 노드별 실시간 예측 가중치]**")
    st.progress(float(proba[0]), text=f"정상 가동 확률: {proba[0]*100:.1f}%")
    st.progress(float(proba[1]), text=f"지연 주의 확률: {proba[1]*100:.1f}%")
    st.progress(float(proba[2]), text=f"센터 마비 확률: {proba[2]*100:.1f}%")
