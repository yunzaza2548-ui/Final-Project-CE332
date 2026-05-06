import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression

# ==========================================================
# ส่วนที่ 0: CONFIGURATION & SETUP
# หน้าที่: ตั้งค่าพื้นฐานของหน้าเว็บ เช่น ชื่อโปรเจกต์ และ Layout
# ==========================================================
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================================
# ส่วนที่ 1: ALGORITHMS (DATA STRUCTURE & SEARCH)
# หน้าที่: จัดการโครงสร้างข้อมูลและการเข้าถึงข้อมูลอย่างมีประสิทธิภาพ
# ==========================================================

# [1.1] Merge Sort Algorithm (O(n log n))
# ขั้นตอน: แบ่งข้อมูลเป็นส่วนย่อย (Divide) แล้วนำกลับมารวมกันแบบเรียงลำดับ (Conquer)
def merge_sort(data, key):
    # ถ้าข้อมูลเหลือ 1 หรือ 0 ไม่ต้องเรียงต่อ
    if len(data) <= 1: return data
    mid = len(data) // 2
    # แบ่งฝั่งซ้ายและขวา
    left = merge_sort(data[:mid], key)
    right = merge_sort(data[mid:], key)
    # นำมาประกอบร่างกันใหม่โดยการเปรียบเทียบค่า
    return merge(left, right, key)

def merge(left, right, key):
    result = []
    i = j = 0
    # เปรียบเทียบตัวแรกของทั้งสองฝั่ง ตัวไหนน้อยกว่าเอาลงตารางก่อน
    while i < len(left) and j < len(right):
        if left[i][key].lower() <= right[j][key].lower():
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    # เก็บตกส่วนที่เหลือ
    result.extend(left[i:]); result.extend(right[j:])
    return result

# [1.2] Binary Search Algorithm (O(log n))
# ขั้นตอน: ค้นหาข้อมูลโดยการ "แบ่งครึ่ง" (ข้อมูลต้องถูก Sort มาก่อนเท่านั้น)
def binary_search_all(data, target_name):
    low = 0
    high = len(data) - 1
    results = []
    
    while low <= high:
        mid = (low + high) // 2
        # ถ้าชื่อที่หา อยู่ในชื่อลำดับกลาง (ใช้ .lower เพื่อให้พิมพ์เล็กใหญ่หาเจอเหมือนกัน)
        if target_name.lower() in data[mid]['name'].lower():
            results.append(data[mid])
            # ตรวจสอบตัวที่อยู่ติดกัน เผื่อมีคนชื่อซ้ำหรือใกล้เคียง
            l = mid - 1
            while l >= 0 and target_name.lower() in data[l]['name'].lower():
                results.append(data[l]); l -= 1
            r = mid + 1
            while r < len(data) and target_name.lower() in data[r]['name'].lower():
                results.append(data[r]); r += 1
            return results
        # ถ้าเป้าหมาย "มากกว่า" ค่ากลาง ให้ตัดฝั่งซ้ายทิ้ง ไปหาฝั่งขวาต่อ
        elif data[mid]['name'].lower() < target_name.lower():
            low = mid + 1
        # ถ้าเป้าหมาย "น้อยกว่า" ค่ากลาง ให้ตัดฝั่งขวาทิ้ง ไปหาฝั่งซ้ายต่อ
        else:
            high = mid - 1
    return results

# ==========================================================
# ส่วนที่ 2: MACHINE LEARNING LOGIC
# หน้าที่: ใช้ปัญญาประดิษฐ์พยากรณ์คะแนนปลายภาคจากข้อมูลในอดีต
# ==========================================================

# [2.1] Linear Regression Model
# ขั้นตอน: นำคะแนน Midterm, Attendance, Work มาคำนวณหาความสัมพันธ์กับคะแนน Final
def predict_with_ml(mid, att, work, db):
    # เปลี่ยนข้อมูล List เป็น DataFrame เพื่อใช้กับ ML
    df = pd.DataFrame(db)
    # กรองเอาเฉพาะข้อมูลที่มีคะแนนสอบจริง (ที่มีค่า final > 0)
    train_df = df[(df['entry_type'] == 'subject_only') & (df['final'] > 0)]
    
    current_total = mid + att + work
    passing_score = 50
    needed = max(0, passing_score - current_total)
    
    # เงื่อนไข: ต้องมีข้อมูลย้อนหลังอย่างน้อย 10 ชุด ถึงจะใช้ AI ได้
    if len(train_df) > 10:
        # กำหนด X (ตัวแปรต้น) และ y (ตัวแปรตาม คือคะแนน Final)
        X = train_df[['midterm', 'attendance', 'assignment']]
        y = train_df['final']
        # สร้างและสอนโมเดล (Train)
        model = LinearRegression()
        model.fit(X, y)
        # พยากรณ์คะแนน Final ของผู้ใช้ปัจจุบัน
        pred_final = model.predict([[mid, att, work]])[0]
        # ตรวจสอบค่าให้อยู่ในช่วง 0-30 คะแนน
        pred_final = max(0, min(30, pred_final)) 
        chance = ((current_total + pred_final) / 100) * 100
        accuracy = model.score(X, y) # ค่า R-Square (ความแม่นยำ)
        return current_total, chance, needed, pred_final, accuracy
    else:
        # ถ้าข้อมูลไม่พอ ให้คำนวณแบบ Simple Logic (โอกาสผ่านเทียบกับคะแนนเต็ม)
        chance = (current_total / 70) * 100
        return current_total, chance, needed, 0, 0.0

# ==========================================================
# ส่วนที่ 3: DATA PREPARATION (MOCK DATA)
# หน้าที่: สร้างข้อมูลจำลองเพื่อใช้ในการสาธิตและ Train AI
# ==========================================================

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
# ลิงก์แหล่งเรียนรู้ภายนอก
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์"]
    data = []
    for _ in range(n):
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({"name": f"{random.choice(first_names)} {random.choice(last_names)}", "uni": "University", "year": random.randint(1, 4), "subject": random.choice(subjects), "midterm": mid, "attendance": att, "assignment": work, "final": final, "total": mid+att+work+final, "gpa": round(random.uniform(2.0, 4.0), 2), "entry_type": etype})
    return data

# ใช้ Session State เพื่อให้ข้อมูล "ไม่หาย" เมื่อกด Refresh หน้าจอ
if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(150)

# ==========================================================
# ส่วนที่ 4: USER INTERFACE (UI)
# หน้าที่: ส่วนแสดงผลและการรับค่าจากผู้ใช้งาน
# ==========================================================

# [4.1] Sidebar Menu
st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- หน้าที่ 1: การพยากรณ์คะแนน ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน (AI Powered)")
    # สร้าง Form เพื่อรับข้อมูลคะแนนจากนักศึกษา
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน/โปรเจกต์ (0-20)", 0, 20)
            consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อ Train AI")
        submit = st.form_submit_button("เริ่มการพยากรณ์ด้วย AI")

    if submit:
        # เรียกใช้ฟังก์ชัน ML พยากรณ์ผล
        current_total, chance, needed, pred_final, accuracy = predict_with_ml(mid, att, work, st.session_state.student_db)
        
        # แสดงผลลัพธ์ผ่าน Metrics
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("เป้าหมาย Final", f"{needed} คะแนน")
        
        # แสดงวิดีโอแนะนำการเรียน
        st.divider()
        st.subheader(f"📚 ทบทวนเนื้อหาวิชา {u_sub}")
        st.video(study_resources.get(u_sub))
        
        # บันทึกข้อมูลลงฐานข้อมูล (ถ้าผู้ใช้ยินยอม)
        if consent:
            st.session_state.student_db.append({"name": u_name if u_name else "Guest", "uni": "University", "year": 1, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("✅ บันทึกข้อมูลสำเร็จ")

# --- หน้าที่ 3: ระบบจัดการข้อมูลและวิเคราะห์ ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    tab1, tab2 = st.tabs(["🔍 ค้นหาด้วย Binary Search", "📊 ภาพรวม Analytics"])
    
    with tab1:
        st.header("📊 ค้นหาคะแนนรายวิชา")
        search_name = st.text_input("ระบุชื่อที่ต้องการค้นหา (Binary Search)")
        
        # ขั้นตอนการทำงานของระบบค้นหา:
        # 1. กรองข้อมูลเฉพาะวิชา
        db_sub = [item for item in st.session_state.student_db if item['entry_type'] == 'subject_only']
        # 2. จัดเรียงข้อมูล (Merge Sort) เพื่อให้ใช้ Binary Search ได้
        sorted_db = merge_sort(db_sub, 'name')
        
        if search_name:
            # 3. ค้นหาแบบแบ่งครึ่ง (Binary Search)
            results = binary_search_all(sorted_db, search_name)
            if results:
                st.success(f"พบข้อมูล {len(results)} รายการ")
                st.dataframe(pd.DataFrame(results))
            else:
                # แจ้งเตือนเมื่อไม่พบข้อมูล
                st.error(f"❌ ไม่พบข้อมูลชื่อ '{search_name}' ในระบบ กรุณาตรวจสอบการสะกด")
        else:
            st.dataframe(pd.DataFrame(sorted_db))

    with tab2:
        # แสดงกราฟวิเคราะห์ข้อมูล
        st.header("📈 สถิติคะแนนเฉลี่ย")
        df_plot = pd.DataFrame([item for item in st.session_state.student_db if item['entry_type'] == 'subject_only'])
        if not df_plot.empty:
            avg_chart = px.bar(df_plot.groupby('subject')['total'].mean().reset_index(), x='subject', y='total', color='subject')
            st.plotly_chart(avg_chart, use_container_width=True)

