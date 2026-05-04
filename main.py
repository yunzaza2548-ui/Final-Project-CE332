import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

# --- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredic AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- DATA STRUCTURES & ALGORITHMS ---

def merge_sort(data, key):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort(data[:mid], key)
    right = merge_sort(data[mid:], key)
    
    return merge(left, right, key)

def merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][key] <= right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def binary_search(data, target_name):
    low = 0
    high = len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid]['name'] == target_name:
            return data[mid]
        elif data[mid]['name'] < target_name:
            low = mid + 1
        else:
            high = mid - 1
    return None

# --- MOCK DATA GENERATOR ---
subjects = [
    "Computer Programming", "Data Structures", "Digital Logic", 
    "Embedded Systems", "Operating Systems", "Software Engineering",
    "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"
]

@st.cache_data
def generate_mock_data(n=50):
    first_names = ["สมชาย", "วิภา", "กิตติ", "นารี", "ธนา", "พรทิพย์", "อนันต์", "สิริ", "วัชระ", "ยุพา"]
    last_names = ["ใจดี", "รักเรียน", "เก่งกาจ", "มุ่งมั่น", "เสริมทรัพย์", "วงค์คำ", "ประเสริฐ"]
    universities = ["BU", "CU", "KU", "MU", "TU"]
    
    data = []
    for _ in range(n):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        record = {
            "name": name,
            "uni": random.choice(universities),
            "year": random.randint(1, 4),
            "subject": random.choice(subjects),
            "midterm": random.randint(15, 40),
            "attendance": random.randint(5, 10),
            "assignment": random.randint(10, 20),
            "final": random.randint(10, 30),
        }
        record["total"] = record["midterm"] + record["attendance"] + record["assignment"] + record["final"]
        data.append(record)
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_mock_data(60)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("เลือกหน้า", ["พยากรณ์ผลการเรียน", "วิเคราะห์และพยากรณ์เกรดเฉลี่ย", "Database & Search"])

# --- PAGE 1: PERFORMANCE PREDICTION ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 พยากรณ์สิทธิ์การผ่านวิชา")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("ข้อมูลนักศึกษา")
        u_name = st.text_input("ชื่อ-นามสกุล")
        u_uni = st.text_input("มหาวิทยาลัย")
        u_year = st.selectbox("ชั้นปี", [1, 2, 3, 4])
        u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
    
    with col2:
        st.subheader("คะแนนสะสม")
        mid = st.number_input("คะแนน Midterm (เต็ม 40)", 0, 40)
        att = st.number_input("คะแนนเข้าเรียน (เต็ม 10)", 0, 10)
        work = st.number_input("คะแนนงาน/Project (เต็ม 20)", 0, 20)
    
    if st.button("วิเคราะห์โอกาสผ่าน"):
        current_total = mid + att + work
        chance = (current_total / 70) * 100
        needed_final = max(0, 50 - current_total)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนสะสมปัจจุบัน", f"{current_total}/70")
        c3.metric("ต้องทำ Final อีก", f"{needed_final} คะแนน")
        
        if needed_final > 30:
            st.error("⚠️ โอกาสผ่านน้อยมาก ต้องพยายามในห้องเรียนเพิ่ม!")
        else:
            st.success("✅ มีโอกาสผ่านสูง! สู้ๆ กับการสอบ Final")
            
        st.subheader("📺 คลิปแนะนำเพื่อเพิ่มความรู้")
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # ลิงก์ตัวอย่าง

# --- PAGE 2: GPA PREDICTION ---
elif page == "วิเคราะห์และพยากรณ์เกรดเฉลี่ย":
    st.title("📊 คำนวณและพยากรณ์ GPA")
    
    with st.expander("กรอกข้อมูลรายวิชา (10 วิชา)"):
        user_scores = []
        cols = st.columns(2)
        for i, sub in enumerate(subjects):
            with cols[i%2]:
                score = st.slider(f"คะแนนวิชา {sub}", 0, 100, 50)
                user_scores.append(score)
    
    if st.button("คำนวณและบันทึกข้อมูล"):
        avg_score = sum(user_scores) / 10
        predicted_gpa = (avg_score / 100) * 4
        
        st.balloons()
        st.metric("พยากรณ์เกรดเฉลี่ย (GPA)", f"{predicted_gpa:.2f}")
        
        # กราฟแสดงผล
        df_plot = pd.DataFrame({"Subject": subjects, "Score": user_scores})
        fig = px.bar(df_plot, x="Subject", y="Score", color="Score", title="สรุปคะแนนแต่ละรายวิชา")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3: DATABASE & SEARCH ---
elif page == "Database & Search":
    st.title("📂 ระบบจัดการข้อมูล (Sorting & Search)")
    
    # Sorting
    sort_key = st.selectbox("เรียงลำดับข้อมูลด้วย Merge Sort", ["name", "total", "midterm"])
    sorted_data = merge_sort(st.session_state.student_db, sort_key)
    
    # Search
    search_query = st.text_input("ค้นหาชื่อนักศึกษา (Binary Search)")
    if search_query:
        # ต้อง Sort ชื่อก่อนทำ Binary Search
        search_data = merge_sort(st.session_state.student_db, "name")
        result = binary_search(search_data, search_query)
        if result:
            st.write("🔍 พบข้อมูล:")
            st.json(result)
        else:
            st.warning("ไม่พบชื่อนี้ในระบบ")
            
    st.subheader("ตารางข้อมูลนักศึกษาทั้งหมด")
    st.table(pd.DataFrame(sorted_data).head(15))