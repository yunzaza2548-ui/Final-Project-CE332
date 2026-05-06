import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression

# --- CONFIG ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================================
# PART 1: ALGORITHMS (SORT & SEARCH)
# ==========================================================

# [1.1] Merge Sort (O(n log n))
def merge_sort(data, key):
    if len(data) <= 1: return data
    mid = len(data) // 2
    left = merge_sort(data[:mid], key)
    right = merge_sort(data[mid:], key)
    return merge(left, right, key)

def merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][key].lower() <= right[j][key].lower():
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

# [1.2] Binary Search (O(log n))
def binary_search_all(data, target_name):
    low, high = 0, len(data) - 1
    results = []
    while low <= high:
        mid = (low + high) // 2
        if target_name.lower() in data[mid]['name'].lower():
            results.append(data[mid])
            l, r = mid - 1, mid + 1
            while l >= 0 and target_name.lower() in data[l]['name'].lower():
                results.append(data[l]); l -= 1
            while r < len(data) and target_name.lower() in data[r]['name'].lower():
                results.append(data[r]); r += 1
            return results
        elif data[mid]['name'].lower() < target_name.lower(): low = mid + 1
        else: high = mid - 1
    return results

# ==========================================================
# PART 2: MACHINE LEARNING LOGIC
# ==========================================================

def predict_with_ml(mid, att, work, db):
    df = pd.DataFrame(db)
    train_df = df[(df['entry_type'] == 'subject_only') & (df['final'] > 0)]
    current_total = mid + att + work
    if len(train_df) > 10:
        X, y = train_df[['midterm', 'attendance', 'assignment']], train_df['final']
        model = LinearRegression().fit(X, y)
        pred_final = np.clip(model.predict([[mid, att, work]])[0], 0, 30)
        return current_total, ((current_total + pred_final) / 100) * 100, max(0, 50 - current_total), pred_final, model.score(X, y)
    return current_total, (current_total / 70) * 100, max(0, 50 - current_total), 0, 0.0

# ==========================================================
# PART 3: MOCK DATA & SESSION
# ==========================================================

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}

if 'student_db' not in st.session_state:
    data = []
    for _ in range(150):
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({"name": f"Student {random.randint(100,999)}", "uni": "BU", "year": random.randint(1, 4), "subject": random.choice(subjects), "midterm": mid, "attendance": att, "assignment": work, "final": final, "total": mid+att+work+final, "gpa": round(random.uniform(2.0, 4.0), 2), "entry_type": etype})
    st.session_state.student_db = data

# ==========================================================
# PART 4: UI
# ==========================================================

st.sidebar.title("🎓 EduPredict AI")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- หน้า 1: พยากรณ์ ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 พยากรณ์ผลการเรียนรายวิชา")
    with st.form("f1"):
        u_name = st.text_input("ชื่อ-นามสกุล")
        u_sub = st.selectbox("วิชา", subjects)
        mid = st.number_input("Midterm", 0, 40)
        att = st.number_input("Attendance", 0, 10)
        work = st.number_input("Assignment", 0, 20)
        consent = st.checkbox("ยินยอมบันทึกคะแนนรายวิชา")
        if st.form_submit_button("วิเคราะห์ AI"):
            curr, chance, need, pred, acc = predict_with_ml(mid, att, work, st.session_state.student_db)
            st.metric("โอกาสผ่าน", f"{int(chance)}%")
            st.video(study_resources.get(u_sub))
            if consent:
                st.session_state.student_db.append({"name": u_name, "uni":"BU", "year":1, "subject":u_sub, "midterm":mid, "attendance":att, "assignment":work, "final":0, "total":curr, "gpa":0.0, "entry_type":"subject_only"})

# --- หน้า 2: คำนวณ GPA ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและบันทึก GPA")
    with st.form("f2"):
        u_name = st.text_input("ชื่อ-นามสกุล")
        all_s = [st.number_input(f"วิชา {s}", 0, 100, 50) for s in subjects]
        consent_gpa = st.checkbox("ยินยอมบันทึก GPA ไปยัง Analytics")
        if st.form_submit_button("คำนวณและบันทึก"):
            gpa = round((sum(all_s)/len(all_s)/100)*4, 2)
            st.metric("GPA คาดการณ์", gpa)
            if consent_gpa:
                st.session_state.student_db.append({"name": u_name, "uni":"BU", "year":1, "subject":"N/A", "midterm":0, "attendance":0, "assignment":0, "final":0, "total":0, "gpa":gpa, "entry_type":"gpa_only"})
                st.success("บันทึกข้อมูล GPA เรียบร้อยแล้ว")

# --- หน้า 3: Analytics ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล")
    t1, t2 = st.tabs(["🔍 รายชื่อนักศึกษาแยกตามวิชา", "🏆 ค้นหาเกรดเฉลี่ย (GPA)"])

    with t1:
        st.header("📊 ข้อมูลคะแนนรายวิชา (จากหน้าพยากรณ์)")
        # กรองเฉพาะ subject_only และลบคอลัมน์ gpa ทิ้ง
        df_sub = pd.DataFrame([i for i in st.session_state.student_db if i['entry_type'] == 'subject_only'])
        df_sub = df_sub.drop(columns=['gpa', 'entry_type'])
        st.dataframe(df_sub, use_container_width=True)

    with t2:
        st.header("🔎 ค้นหา GPA (จากหน้าคำนวณ GPA)")
        # กรองเฉพาะ gpa_only และลบคอลัมน์คะแนนรายวิชาทิ้ง
        db_gpa = [i for i in st.session_state.student_db if i['entry_type'] == 'gpa_only']
        sorted_gpa = merge_sort(db_gpa, 'name')
        
        search = st.text_input("ค้นหาชื่อนักศึกษาเพื่อดูเกรด...")
        if search:
            res = binary_search_all(sorted_gpa, search)
            if res:
                st.success(f"พบข้อมูล GPA ของ {search}")
                # แสดงเฉพาะชื่อและ GPA
                st.dataframe(pd.DataFrame(res)[['name', 'uni', 'year', 'gpa']], use_container_width=True)
            else:
                st.error("❌ ไม่พบข้อมูล GPA ของชื่อนี้")
        else:
            st.dataframe(pd.DataFrame(sorted_gpa)[['name', 'uni', 'year', 'gpa']], use_container_width=True)

