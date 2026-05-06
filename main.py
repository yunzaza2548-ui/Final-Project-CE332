import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression

# --- 0. CONFIGURATION ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================================
# 1. ALGORITHMS (SORT & SEARCH)
# ==========================================================

# [SORTING]: Merge Sort (O(n log n))
# ทำหน้าที่เรียงลำดับข้อมูล โดยรองรับการเรียงจากมากไปน้อย (reverse=True) สำหรับปีการศึกษา
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
        val_l = str(left[i][key]).lower()
        val_r = str(right[j][key]).lower()
        
        # เงื่อนไขการเรียงลำดับ
        condition = val_l >= val_r if reverse else val_l <= val_r
        if condition:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

# [SEARCHING]: Binary Search (O(log n))
# ใช้ค้นหาข้อมูลจาก Key ที่กำหนด (ชื่อ หรือ ปีการศึกษา)
def binary_search_all(data, key, target):
    low, high = 0, len(data) - 1
    results = []
    target_val = str(target).lower()
    
    while low <= high:
        mid = (low + high) // 2
        current_val = str(data[mid][key]).lower()
        
        if target_val in current_val:
            results.append(data[mid])
            # ตรวจสอบหาค่าที่ใกล้เคียงข้างเคียง
            l, r = mid - 1, mid + 1
            while l >= 0 and target_val in str(data[l][key]).lower():
                results.append(data[l]); l -= 1
            while r < len(data) and target_val in str(data[r][key]).lower():
                results.append(data[r]); r += 1
            return results
        elif current_val < target_val: low = mid + 1
        else: high = mid - 1
    return results

# ==========================================================
# 2. MACHINE LEARNING LOGIC
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
# 3. MOCK DATA & CONSTANTS
# ==========================================================

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}

if 'student_db' not in st.session_state:
    data = []
    for _ in range(150):
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({
            "name": f"Student {random.randint(100,999)}", 
            "uni": "Bangkok University", 
            "year": random.randint(2564, 2568),
            "subject": random.choice(subjects), 
            "midterm": mid, "attendance": att, "assignment": work, "final": final, 
            "total": mid+att+work+final, "gpa": round(random.uniform(2.0, 4.0), 2), 
            "entry_type": etype
        })
    st.session_state.student_db = data

# ==========================================================
# 4. USER INTERFACE (UI)
# ==========================================================

st.sidebar.title("🎓 EduPredict AI Pro")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- หน้าที่ 1: พยากรณ์ (UI แบบดั้งเดิม) ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน (AI Powered)")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_uni = st.selectbox("มหาวิทยาลัย", ["Bangkok University", "อื่นๆ"])
            u_year = st.number_input("ปีการศึกษา", 2560, 2580, 2567)
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน/โปรเจกต์ (0-20)", 0, 20)
            consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อนำไปพัฒนาระบบ AI")
        submit = st.form_submit_button("เริ่มการพยากรณ์ด้วย AI")

    if submit:
        curr, chance, need, pred, acc = predict_with_ml(mid, att, work, st.session_state.student_db)
        st.subheader("📊 ผลการวิเคราะห์")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(chance)}%")
        c2.metric("คะแนนปัจจุบัน", f"{curr}/70")
        c3.metric("ต้องทำ Final", f"{need} คะแนน")
        st.divider()
        st.video(study_resources.get(u_sub))
        if consent:
            st.session_state.student_db.append({"name": u_name, "uni": u_uni, "year": u_year, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": curr, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("บันทึกข้อมูลเรียบร้อย")

# --- หน้าที่ 2: วิเคราะห์เกรดเฉลี่ย (UI เดิม + ยินยอม) ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    with st.form("gpa_form"):
        u_name_gpa = st.text_input("ชื่อ-นามสกุล")
        u_year_gpa = st.number_input("ปีการศึกษา", 2560, 2580, 2567)
        cols = st.columns(2); all_scores = []
        for i, sub in enumerate(subjects):
            with cols[i % 2]:
                all_scores.append(st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_{i}"))
        gpa_consent = st.checkbox("ยินยอมให้บันทึกข้อมูล GPA เพื่อไปแสดงในหน้า Analytics")
        if st.form_submit_button("คำนวณและบันทึก"):
            gpa = round((sum(all_scores)/len(all_scores)/100)*4, 2)
            st.metric("GPA คาดการณ์", gpa)
            if gpa_consent:
                st.session_state.student_db.append({"name": u_name_gpa, "uni": "Bangkok University", "year": u_year_gpa, "subject": "Overall GPA", "midterm": 0, "attendance": 0, "assignment": 0, "final": 0, "total": 0, "gpa": gpa, "entry_type": "gpa_only"})
                st.success("บันทึกข้อมูล GPA แล้ว")

# --- หน้าที่ 3: Analytics (แยก 2 ส่วน + ค้นหาปีการศึกษาเรียงจากมากไปน้อย) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    t1, t2 = st.tabs(["🔍 รายชื่อนักศึกษา (รายวิชา)", "🏆 ค้นหา GPA"])

    with t1:
        st.header("📊 รายงานคะแนนรายวิชา")
        c1, c2 = st.columns(2)
        with c1: s_name = st.text_input("ค้นหาชื่อ", key="find_name")
        with c2: s_year = st.text_input("ค้นหาปีการศึกษา", key="find_year")
        
        # จัดเรียงปีการศึกษาจากมากไปน้อย (reverse=True)
        db_sub = [i for i in st.session_state.student_db if i['entry_type'] == 'subject_only']
        sorted_sub = merge_sort(db_sub, 'year', reverse=True)
        
        if s_name:
            res = binary_search_all(sorted_sub, 'name', s_name)
            if res: st.dataframe(pd.DataFrame(res).drop(columns=['gpa', 'entry_type']), use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        elif s_year:
            # ต้องเรียงตามปีเพื่อใช้ Binary Search ค้นหาปี
            sorted_for_year = merge_sort(db_sub, 'year') 
            res = binary_search_all(sorted_for_year, 'year', s_year)
            if res: st.dataframe(pd.DataFrame(res).drop(columns=['gpa', 'entry_type']), use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        else:
            st.dataframe(pd.DataFrame(sorted_sub).drop(columns=['gpa', 'entry_type']), use_container_width=True)

    with t2:
        st.header("🏆 รายงานเกรดเฉลี่ย (GPA)")
        c1, c2 = st.columns(2)
        with c1: sg_name = st.text_input("ค้นหาชื่อ (GPA)", key="find_name_g")
        with c2: sg_year = st.text_input("ค้นหาปีการศึกษา (GPA)", key="find_year_g")
        
        db_gpa = [i for i in st.session_state.student_db if i['entry_type'] == 'gpa_only']
        sorted_gpa = merge_sort(db_gpa, 'year', reverse=True)
        
        if sg_name:
            res = binary_search_all(sorted_gpa, 'name', sg_name)
            if res: st.dataframe(pd.DataFrame(res)[['name', 'uni', 'year', 'gpa']], use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        elif sg_year:
            sorted_for_year_g = merge_sort(db_gpa, 'year')
            res = binary_search_all(sorted_for_year_g, 'year', sg_year)
            if res: st.dataframe(pd.DataFrame(res)[['name', 'uni', 'year', 'gpa']], use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        else:
            st.dataframe(pd.DataFrame(sorted_gpa)[['name', 'uni', 'year', 'gpa']], use_container_width=True)

