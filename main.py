import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression

#--- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================================
# PART 1: ALGORITHMS (SORTING & SEARCHING)
# ==========================================================

# [SORTING]: Merge Sort (O(n log n)) 
# จำเป็นมากสำหรับ Binary Search เพราะข้อมูลต้องเรียงลำดับก่อนค้นหา
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

# [SEARCHING]: Binary Search (O(log n))
# ทำงานโดยการแบ่งครึ่งข้อมูลเพื่อค้นหา (เร็วกว่าการหาทีละตัว)
def binary_search_all(data, target_name):
    low = 0
    high = len(data) - 1
    results = []
    
    while low <= high:
        mid = (low + high) // 2
        # ใช้ .lower() เพื่อให้ค้นหาได้โดยไม่สนใจตัวพิมพ์เล็ก-ใหญ่
        if target_name.lower() in data[mid]['name'].lower():
            # เมื่อพบชื่อที่ใกล้เคียง ให้ขยายขอบเขตหาตัวข้างเคียงที่อาจมีชื่อซ้ำกัน
            results.append(data[mid])
            # ตรวจสอบตัวก่อนหน้า
            l = mid - 1
            while l >= 0 and target_name.lower() in data[l]['name'].lower():
                results.append(data[l]); l -= 1
            # ตรวจสอบตัวถัดไป
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
# PART 2: MACHINE LEARNING LOGIC
# ==========================================================

# [ML]: Linear Regression Model
def predict_with_ml(mid, att, work, db):
    df = pd.DataFrame(db)
    train_df = df[(df['entry_type'] == 'subject_only') & (df['final'] > 0)]
    
    current_total = mid + att + work
    passing_score = 50
    needed = max(0, passing_score - current_total)
    
    if len(train_df) > 10:
        X = train_df[['midterm', 'attendance', 'assignment']]
        y = train_df['final']
        model = LinearRegression()
        model.fit(X, y)
        pred_final = model.predict([[mid, att, work]])[0]
        pred_final = max(0, min(30, pred_final)) 
        chance = ((current_total + pred_final) / 100) * 100
        accuracy = model.score(X, y)
        return current_total, chance, needed, pred_final, accuracy
    else:
        chance = (current_total / 70) * 100
        return current_total, chance, needed, 0, 0.0

# ==========================================================
# PART 3: DATA PREPARATION & MOCK DATA
# ==========================================================

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]
study_resources = {"Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E", "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc", "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60", "Embedded Systems": "https://www.youtube.com/watch?v=B6ofL_S_X6A", "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8", "Software Engineering": "https://www.youtube.com/watch?v=pETh_as6Y78", "Database Systems": "https://www.youtube.com/watch?v=HXV3zeQKqGY", "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8", "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg", "Robotics Design": "https://www.youtube.com/watch?v=0yG-fMHeM6Y"}
uni_options = ["Bangkok University", "Chulalongkorn University", "Kasetsart University", "Mahidol University", "Thammasat University", "KMUTT", "KMITL", "อื่นๆ"]

@st.cache_data
def generate_enhanced_mock_data(n=100):
    first_names = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ", "พีรพล", "วิชุดา", "ภาณุ", "อรวรรณ"]
    last_names = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์", "ปัญญาดี", "สุขสวัสดิ์"]
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
# PART 4: USER INTERFACE (UI)
# ==========================================================

st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- PAGE 1: PREDICTION ---
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
        current_total, chance, needed, pred_final, accuracy = predict_with_ml(mid, att, work, st.session_state.student_db)
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("เป้าหมาย Final", f"{needed} คะแนน")
        st.divider()
        st.video(study_resources.get(u_sub))
        if consent:
            st.session_state.student_db.append({"name": u_name, "uni": "Guest", "year": 1, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("✅ บันทึกข้อมูลแล้ว")

# --- PAGE 3: DB & ANALYTICS (Using Binary Search) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    tab1, tab2 = st.tabs(["🔍 ค้นหาคะแนนรายวิชา", "🎓 วิเคราะห์เกรดเฉลี่ย (GPA)"])
    
    with tab1:
        st.header("📊 รายงานผลการเรียนรายวิชา")
        search_sub = st.text_input("🔍 ค้นชื่อนักศึกษาด้วย Binary Search", placeholder="พิมพ์ชื่อ...")
        
        # 1. เตรียมข้อมูลเฉพาะส่วน subject_only
        db_sub = [item for item in st.session_state.student_db if item['entry_type'] == 'subject_only']
        
        # 2. ต้องทำการ Sort ก่อนใช้ Binary Search
        sorted_db_sub = merge_sort(db_sub, 'name')
        
        if search_sub:
            # 3. เรียกใช้ Binary Search
            search_results = binary_search_all(sorted_db_sub, search_sub)
            
            if search_results:
                st.success(f"พบข้อมูลของ '{search_sub}' จำนวน {len(search_results)} รายการ")
                st.dataframe(pd.DataFrame(search_results).drop(columns=['gpa', 'entry_type']), use_container_width=True)
            else:
                st.error(f"❌ ไม่พบชื่อ '{search_sub}' ในระบบ (Binary Search: Not Found)")
        else:
            st.dataframe(pd.DataFrame(sorted_db_sub).drop(columns=['gpa', 'entry_type']), use_container_width=True)

    with tab2:
        st.header("🏆 รายงานเกรดเฉลี่ยสะสม (GPA)")
        search_gpa = st.text_input("🔍 ค้นชื่อนักศึกษา (GPA)", key="gpa_search")
        db_gpa = [item for item in st.session_state.student_db if item['entry_type'] == 'gpa_only']
        sorted_db_gpa = merge_sort(db_gpa, 'name')

        if search_gpa:
            results_gpa = binary_search_all(sorted_db_gpa, search_gpa)
            if results_gpa:
                st.success(f"พบข้อมูล GPA ของ '{search_gpa}'")
                st.dataframe(pd.DataFrame(results_gpa)[['name', 'uni', 'year', 'gpa']])
            else:
                st.error("❌ ไม่พบข้อมูล")
        else:
            st.dataframe(pd.DataFrame(sorted_db_gpa)[['name', 'uni', 'year', 'gpa']])
        
