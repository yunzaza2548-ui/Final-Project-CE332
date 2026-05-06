import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
# นำเข้า Library สำหรับ Machine Learning
from sklearn.linear_model import LinearRegression

# ==========================================================
# ส่วนที่ 0: CONFIGURATION & SETUP
# หน้าที่: ตั้งค่าพื้นฐานแอปพลิเคชัน
# ==========================================================
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================================
# ส่วนที่ 1: ALGORITHMS (DATA STRUCTURE & SEARCH)
# ==========================================================

# [1.1] Merge Sort: จัดเรียงข้อมูลแบบแบ่งครึ่ง (O(n log n))
# ขั้นตอน: ข้อมูลต้องถูก Sort ก่อนเสมอเพื่อให้ Binary Search ทำงานได้
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

# [1.2] Binary Search: ค้นหาข้อมูลแบบแบ่งครึ่ง (O(log n))
# ขั้นตอน: กระโดดไปดูค่ากลาง ถ้าไม่ใช่ก็ตัดออกทีละครึ่ง ทำให้หาข้อมูลได้เร็วมาก
def binary_search_all(data, target_name):
    low = 0
    high = len(data) - 1
    results = []
    
    while low <= high:
        mid = (low + high) // 2
        # ตรวจสอบว่า target อยู่ในชื่อลำดับที่ mid หรือไม่
        if target_name.lower() in data[mid]['name'].lower():
            results.append(data[mid])
            # ตรวจสอบหาชื่อที่อาจซ้ำกันในตำแหน่งข้างเคียง
            l = mid - 1
            while l >= 0 and target_name.lower() in data[l]['name'].lower():
                results.append(data[l]); l -= 1
            r = mid + 1
            while r < len(data) and target_name.lower() in data[r]['name'].lower():
                results.append(data[r]); r += 1
            return results
        elif data[mid]['name'].lower() < target_name.lower():
            low = mid + 1
        else:
            high = mid - 1
    return results

# ==========================================================
# ส่วนที่ 2: AI & MACHINE LEARNING LOGIC
# ==========================================================

# [2.1] Linear Regression: พยากรณ์คะแนนปลายภาค
# ขั้นตอน: นำข้อมูล Midterm, Attendance, Work ไปคำนวณหาค่า Final ที่น่าจะเป็นไปได้
def predict_with_ml(mid, att, work, db):
    df = pd.DataFrame(db)
    # ใช้เฉพาะข้อมูลที่มีคะแนน Final จริงๆ มาทำการ Train
    train_df = df[(df['entry_type'] == 'subject_only') & (df['final'] > 0)]
    
    current_total = mid + att + work
    passing_score = 50
    needed = max(0, passing_score - current_total)
    
    if len(train_df) > 10:
        X = train_df[['midterm', 'attendance', 'assignment']]
        y = train_df['final']
        model = LinearRegression()
        model.fit(X, y)
        
        # ทำนายผล
        pred_final = model.predict([[mid, att, work]])[0]
        pred_final = max(0, min(30, pred_final)) 
        chance = ((current_total + pred_final) / 100) * 100
        accuracy = model.score(X, y) # ค่าความแม่นยำ R-Squared
        return current_total, chance, needed, pred_final, accuracy
    else:
        # Fallback Logic ถ้าข้อมูลมีไม่พอสำหรับ AI
        chance = (current_total / 70) * 100
        return current_total, chance, needed, 0, 0.0

# ==========================================================
# ส่วนที่ 3: CONSTANTS & MOCK DATA
# ==========================================================

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}
uni_options = ["Bangkok University", "Chulalongkorn University", "Kasetsart University", "Mahidol University", "Thammasat University", "KMUTT", "KMITL", "อื่นๆ"]

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ", "วิชุดา", "ภาณุ"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์", "ปัญญาดี"]
    data = []
    for _ in range(n):
        mid, att, work = random.randint(10, 40), random.randint(5, 10), random.randint(5, 20)
        final = random.randint(10, 30)
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({"name": f"{random.choice(first_names)} {random.choice(last_names)}", "uni": random.choice(uni_options[:-1]), "year": random.randint(1, 4), "subject": random.choice(subjects), "midterm": mid, "attendance": att, "assignment": work, "final": final, "total": mid+att+work+final, "gpa": round(random.uniform(2.0, 4.0), 2), "entry_type": etype})
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(150)

# ==========================================================
# ส่วนที่ 4: USER INTERFACE (UI)
# ==========================================================

st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- หน้า 1: พยากรณ์ผลการเรียน ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน (AI Powered)")
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
        # ขั้นตอนที่ 1: เรียกใช้ AI Predict
        current_total, chance, needed, pred_final, accuracy = predict_with_ml(mid, att, work, st.session_state.student_db)
        
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("ต้องทำ Final อีก", f"{needed} คะแนน")
        
        if accuracy > 0:
            st.caption(f"💡 AI คาดการณ์คะแนนปลายภาคของคุณที่: {pred_final:.2f} (Model Accuracy R²: {accuracy:.2f})")
        
        st.divider()
        st.video(study_resources.get(u_sub))
        
        if consent:
            # ขั้นตอนที่ 2: บันทึกข้อมูลเข้า DB
            st.session_state.student_db.append({"name": u_name if u_name else "Guest", "uni": "University", "year": 1, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("✅ บันทึกข้อมูลเรียบร้อย")

# --- หน้า 2: วิเคราะห์เกรดเฉลี่ย ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    with st.form("gpa_form"):
        u_name_gpa = st.text_input("ชื่อ-นามสกุล")
        cols = st.columns(2); all_scores = []
        for i, sub in enumerate(subjects):
            with cols[i % 2]:
                all_scores.append(st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_{i}"))
        gpa_btn = st.form_submit_button("คำนวณ GPA")

    if gpa_btn:
        final_gpa = round((sum(all_scores) / len(all_scores) / 100) * 4, 2)
        st.metric("เกรดเฉลี่ยพยากรณ์ (GPA)", f"{final_gpa}")
        st.session_state.student_db.append({"name": u_name_gpa if u_name_gpa else "Guest", "uni": "University", "year": 1, "subject": "GPA Record", "midterm": 0, "attendance": 0, "assignment": 0, "final": 0, "total": 0, "gpa": final_gpa, "entry_type": "gpa_only"})

# --- หน้า 3: ฐานข้อมูล & Analytics ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    tab1, tab2 = st.tabs(["🔍 ค้นหาด้วย Binary Search", "📈 ภาพรวมสถิติ"])
    
    with tab1:
        st.header("🔎 ค้นหาชื่อนักศึกษา")
        search_query = st.text_input("พิมพ์ชื่อเพื่อค้นหา...", placeholder="ชื่อหรือนามสกุล")
        
        # ขั้นตอนที่ 1: กรองข้อมูลเฉพาะส่วนรายวิชา
        db_sub = [item for item in st.session_state.student_db if item['entry_type'] == 'subject_only']
        # ขั้นตอนที่ 2: Merge Sort ก่อนค้นหา (สำคัญ!)
        sorted_db = merge_sort(db_sub, 'name')
        
        if search_query:
            # ขั้นตอนที่ 3: Binary Search
            results = binary_search_all(sorted_db, search_query)
            if results:
                st.success(f"พบข้อมูลจำนวน {len(results)} รายการ")
                st.dataframe(pd.DataFrame(results).drop(columns=['gpa', 'entry_type']), use_container_width=True)
            else:
                # ขั้นตอนที่ 4: Error Message เมื่อไม่พบข้อมูล
                st.error(f"❌ ไม่พบชื่อ '{search_query}' ในระบบ กรุณาตรวจสอบการสะกดและลองอีกครั้ง")
        else:
            st.dataframe(pd.DataFrame(sorted_db).drop(columns=['gpa', 'entry_type']), use_container_width=True)

    with tab2:
        st.header("📈 วิเคราะห์คะแนนรวมเฉลี่ยต่อวิชา")
        df_plot = pd.DataFrame([item for item in st.session_state.student_db if item['entry_type'] == 'subject_only'])
        if not df_plot.empty:
            avg_fig = px.bar(df_plot.groupby('subject')['total'].mean().reset_index(), x='subject', y='total', color='subject')
            st.plotly_chart(avg_fig, use_container_width=True)
    
