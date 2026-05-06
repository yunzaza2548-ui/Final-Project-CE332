import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

# --- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredic AI Pro", page_icon="🧠", layout="wide")

# --- ALGORITHMS ---
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

def binary_search(data, target_name):
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid]['name'] == target_name: return data[mid]
        elif data[mid]['name'] < target_name: low = mid + 1
        else: high = mid - 1
    return None

# --- CONSTANTS & MOCK DATA ---
subjects = [
    "Computer Programming", "Data Structures", "Digital Logic", 
    "Embedded Systems", "Operating Systems", "Software Engineering",
    "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"
]

#  (ระบบแนะนำคลิปทุกวิชา)
study_resources = {
    "Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E",
    "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc",
    "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60",
    "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A",
    "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8",
    "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78",
    "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY",
    "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8",
    "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg",
    "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"
}

uni_options = [
    "Bangkok University", "Chulalongkorn University", "Kasetsart University", 
    "Mahidol University", "Thammasat University", "Chiang Mai University", 
    "Khon Kaen University", "Prince of Songkla University", "KMUTT", "KMITL", "อื่นๆ"
]

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ", "พีรพล", "วิชุดา", "ภาณุ", "อรวรรณ"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์", "ปัญญาดี", "สุขสวัสดิ์"]
    
    data = []
    for _ in range(n):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        total = mid + att + work + final
        data.append({
            "name": name, "uni": random.choice(uni_options[:-1]),
            "year": random.randint(1, 4), "subject": random.choice(subjects),
            "midterm": mid, "attendance": att, "assignment": work, "final": final,
            "total": total, "gpa": round(random.uniform(2.0, 4.0), 2)
        })
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(100)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 EduPredic AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- PAGE 1: PREDICTION ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_uni = st.selectbox("มหาวิทยาลัย", uni_options)
            u_year = st.slider("ชั้นปี", 1, 4)
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน/โปรเจกต์ (0-20)", 0, 20)

        consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อนำไปพัฒนาระบบ AI")
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
            st.session_state.student_db.append({
                "name": u_name if u_name else "Guest", "uni": u_uni, "year": u_year,
                "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work,
                "final": 0, "total": current_total, "gpa": 0.0
            })
            st.success("✅ บันทึกข้อมูลเข้าฐานข้อมูลแล้ว")

        # ✅ เพิ่มตรงนี้ (แนะนำคลิปตามวิชา)
        st.subheader(f"📚 แนะนำคลิปเรียน: {u_sub}")
        st.video(study_resources.get(u_sub))
