import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#--- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================
# 1. DATA STRUCTURE & ALGORITHMS + ML LOGIC
# ==========================================

# [Algorithm] SORTING: Merge Sort (O(n log n))
def merge_sort(data, key, reverse=False):
    if len(data) <= 1: return data
    mid = len(data) // 2
    left = merge_sort(data[:mid], key, reverse)
    right = merge_sort(data[mid:], key, reverse)
    return merge(left, right, key, reverse)

def merge(left, right, key, reverse):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        condition = left[i][key] >= right[j][key] if reverse else left[i][key] <= right[j][key]
        if condition:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

# [AI/ML] NEW: Machine Learning Model (Linear Regression)
def train_and_predict(mid, att, work, db):
    # ดึงข้อมูลจากฐานข้อมูลมาทำ Training Data
    df = pd.DataFrame(db)
    if len(df) < 20: # ถ้าข้อมูลน้อยเกินไปให้ใช้ Logic พื้นฐาน
        return predict_performance_basic(mid, att, work)
    
    # เลือกเฉพาะข้อมูลที่มีคะแนน Final (ข้อมูล Mock)
    train_df = df[df['final'] > 0]
    X = train_df[['midterm', 'attendance', 'assignment']]
    y = train_df['final']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict Final Score
    input_data = np.array([[mid, att, work]])
    predicted_final = model.predict(input_data)[0]
    
    # คำนวณสรุป
    current_total = mid + att + work
    passing_score = 50
    needed = max(0, passing_score - current_total)
    chance = ((current_total + predicted_final) / 100) * 100
    
    return current_total, chance, needed, predicted_final, model.score(X, y)

def predict_performance_basic(mid, att, work):
    current_total = mid + att + work
    chance = (current_total / 70) * 100
    needed = max(0, 50 - current_total)
    return current_total, chance, needed, 0, 0

# ==========================================
# 2. CONSTANTS & MOCK DATA (เหมือนเดิม)
# ==========================================
subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}
uni_options = ["Bangkok University", "Chulalongkorn University", "Kasetsart University", "Mahidol University", "Thammasat University", "KMUTT", "KMITL", "อื่นๆ"]

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา"]
    data = []
    for _ in range(n):
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        data.append({
            "name": f"{random.choice(first_names)} {random.choice(last_names)}", 
            "uni": random.choice(uni_options[:-1]), "year": random.randint(1, 4), 
            "subject": random.choice(subjects), "midterm": mid, "attendance": att, 
            "assignment": work, "final": final, "total": mid+att+work+final, 
            "gpa": round(random.uniform(2.0, 4.0), 2), "entry_type": random.choice(["subject_only", "gpa_only"])
        })
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(100)

# ==========================================
# 3. USER INTERFACE (UI)
# ==========================================
st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ด้วย Machine Learning")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน (0-20)", 0, 20)
            consent = st.checkbox("บันทึกข้อมูลเพื่อ Train AI")
        submit = st.form_submit_button("Run AI Prediction")

    if submit:
        # เรียกใช้ฟังก์ชัน ML
        current_total, chance, needed, pred_final, accuracy = train_and_predict(mid, att, work, st.session_state.student_db)
        
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน (Estimate)", f"{int(min(chance, 100))}%")
        c2.metric("คะแนน Final ที่คาดการณ์", f"{pred_final:.2f}")
        c3.metric("Model Accuracy (R²)", f"{accuracy:.2f}")

        with st.expander("🔍 อธิบายการทำงานของ AI"):
            st.write(f"ระบบใช้ **Linear Regression** ในการเรียนรู้จากข้อมูลนักศึกษาเก่า {len(st.session_state.student_db)} คน")
            st.write(f"โดยวิเคราะห์ว่าคะแนน Midterm และงานส่ง มีผลต่อคะแนน Final อย่างไร")
            st.progress(accuracy if accuracy > 0 else 0, text="ความแม่นยำของ Model")

        st.divider()
        v_col, t_col = st.columns([3, 2])
        with v_col: st.video(study_resources.get(u_sub))
        with t_col: 
            st.info(f"AI แนะนำ: เพื่อให้ผ่านเกณฑ์ 50 คะแนน คุณต้องทำคะแนนสอบปลายภาคให้ได้อย่างน้อย {needed} คะแนน")

        if consent:
            st.session_state.student_db.append({"name": u_name, "uni": "Guest", "year": 1, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("บันทึกข้อมูลเข้าสู่ Training Set แล้ว")

# (ส่วน Page 2 และ Page 3 คงเดิมตามโค้ดก่อนหน้าได้เลย)

