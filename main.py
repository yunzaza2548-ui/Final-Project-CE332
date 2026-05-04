import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

# --- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredic AI Pro", page_icon="🧠", layout="wide")

# --- ALGORITHMS (Merge Sort & Binary Search) ---
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
        if left[i][key] <= right[j][key]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

def binary_search(data, target_name):
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid]['name'] == target_name: return data[mid]
        elif data[mid]['name'] < target_name: low = mid + 1
        else: high = mid - 1
    return None

# --- DYNAMIC MOCK DATA GENERATOR ---
subjects = [
    "Computer Programming", "Data Structures", "Digital Logic", 
    "Embedded Systems", "Operating Systems", "Software Engineering",
    "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"
]

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ", "พีรพล", "วิชุดา", "ภาณุ", "อรวรรณ"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์", "ปัญญาดี", "สุขสวัสดิ์"]
    universities = ["Bangkok University", "Chulalongkorn", "Kasetsart", "Mahidol", "Thammasat", "CMU", "KKU"]
    
    data = []
    for _ in range(n):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(0, 30)
        total = mid + att + work + final
        data.append({
            "name": name, "uni": random.choice(universities),
            "year": random.randint(1, 4), "subject": random.choice(subjects),
            "midterm": mid, "attendance": att, "assignment": work, "final": final,
            "total": total, "gpa": round(random.uniform(2.0, 4.0), 2)
        })
    return data

# Initialize Session State
if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(100)

# --- SIDEBAR ---
st.sidebar.title("🎓 EduPredic AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["หน้าแรก & พยากรณ์", "Dashboard วิเคราะห์ข้อมูล", "ระบบจัดการฐานข้อมูล"])

# --- PAGE 1: PREDICTION & DATA ENTRY ---
if page == "หน้าแรก & พยากรณ์":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน")
    
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_uni = st.selectbox("มหาวิทยาลัย", ["Bangkok University", "อื่นๆ"])
            u_year = st.slider("ชั้นปี", 1, 4)
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน/โปรเจกต์ (0-20)", 0, 20)
        
        consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อนำไปพัฒนาระบบ AI พยากรณ์ต่อ")
        submit = st.form_submit_button("เริ่มการพยากรณ์")

    if submit:
        current_total = mid + att + work
        chance = (current_total / 70) * 100
        needed = max(0, 50 - current_total)
        
        st.subheader("📊 ผลการวิเคราะห์")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("ต้องทำ Final อีก", f"{needed} คะแนน")

        if consent:
            new_data = {
                "name": u_name if u_name else "Anonymous", "uni": u_uni, "year": u_year,
                "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work,
                "final": 0, "total": current_total, "gpa": 0.0
            }
            st.session_state.student_db.append(new_data)
            st.success("✅ บันทึกข้อมูลเข้าสู่ระบบฐานข้อมูลถาวรแล้ว")
        
        st.info("📺 วิดีโอแนะนำการติววิชานี้")
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# --- PAGE 2: DASHBOARD ---
elif page == "Dashboard วิเคราะห์ข้อมูล":
    st.title("📈 สถิติและข้อมูลภาพรวม (Analytics)")
    df = pd.DataFrame(st.session_state.student_db)
    
    col1, col2 = st.columns(2)
    with col1:
        # กราฟแสดงจำนวนนักศึกษาแยกตามมหาวิทยาลัย
        fig1 = px.pie(df, names='uni', title='สัดส่วนนักศึกษาตามมหาวิทยาลัย', hole=0.4)
        st.plotly_chart(fig1)
        
        # กราฟแสดงคะแนนเฉลี่ยแยกตามชั้นปี
        fig2 = px.line(df.groupby('year')['total'].mean().reset_index(), x='year', y='total', title='แนวโน้มคะแนนเฉลี่ยตามชั้นปี')
        st.plotly_chart(fig2)

    with col2:
        # กราฟความสัมพันธ์ระหว่างคะแนนเก็บและคะแนนรวม
        fig3 = px.scatter(df, x='midterm', y='total', color='subject', title='ความสัมพันธ์ Midterm vs Total')
        st.plotly_chart(fig3)
        
        # Histogram กระจายเกรดเฉลี่ย
        fig4 = px.histogram(df, x='gpa', nbins=10, title='การกระจายตัวของ GPA ในระบบ', color_discrete_sequence=['indianred'])
        st.plotly_chart(fig4)

# --- PAGE 3: DATABASE MANAGEMENT ---
elif page == "ระบบจัดการฐานข้อมูล":
    st.title("📂 ระบบจัดการข้อมูล (Merge Sort & Binary Search)")
    df = pd.DataFrame(st.session_state.student_db)
    
    # ส่วนการค้นหา
    search_q = st.text_input("🔍 ค้นหาชื่อนักศึกษา (Binary Search)")
    if search_q:
        # Sort ข้อมูลก่อนทำ Binary Search
        sorted_for_search = merge_sort(st.session_state.student_db, 'name')
        res = binary_search(sorted_for_search, search_q)
        if res: st.success(f"พบข้อมูล: {res['name']} จาก {res['uni']} เกรด: {res['gpa']}")
        else: st.error("ไม่พบข้อมูลนักศึกษาท่านนี้")

    # ส่วนการจัดเรียง
    st.subheader("ตารางข้อมูลนักศึกษาทั้งหมด")
    sort_option = st.selectbox("เรียงข้อมูลโดย:", ["name", "total", "gpa", "year"])
    sorted_table = merge_sort(st.session_state.student_db, sort_option)
    
    st.dataframe(pd.DataFrame(sorted_table), use_container_width=True, height=400)
    st.caption(f"จำนวนข้อมูลทั้งหมดในระบบ: {len(st.session_state.student_db)} รายการ")
