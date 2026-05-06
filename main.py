import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
# นำเข้า Library สำหรับ Machine Learning
from sklearn.linear_model import LinearRegression

#--- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================
# 1. DATA STRUCTURE, ALGORITHMS & ML LOGIC
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

# [AI/ML LOGIC] Linear Regression Prediction
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

# ==========================================
# 2. CONSTANTS & MOCK DATA
# ==========================================

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
        data.append({
            "name": f"{random.choice(first_names)} {random.choice(last_names)}", 
            "uni": random.choice(uni_options[:-1]), "year": random.randint(1, 4), 
            "subject": random.choice(subjects), "midterm": mid, "attendance": att, 
            "assignment": work, "final": final, "total": mid+att+work+final, 
            "gpa": round(random.uniform(2.0, 4.0), 2), "entry_type": etype
        })
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(150)

# ==========================================
# 3. USER INTERFACE (UI)
# ==========================================

st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- PAGE 1: PREDICTION ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน (AI Powered)")
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
        submit = st.form_submit_button("เริ่มการพยากรณ์ด้วย AI")

    if submit:
        current_total, chance, needed, pred_final, accuracy = predict_with_ml(mid, att, work, st.session_state.student_db)
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน (Estimated)", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("ต้องทำ Final อย่างน้อย", f"{needed} คะแนน")
        if accuracy > 0: st.caption(f"💡 AI คาดการณ์คะแนนปลายภาค: {pred_final:.2f} (Model Accuracy R²: {accuracy:.2f})")
        st.divider()
        v_col, t_col = st.columns([3, 2])
        with v_col: st.video(study_resources.get(u_sub))
        with t_col: 
            st.info("**EduPredict AI Advice:**")
            if chance < 50: st.warning("คะแนนเสี่ยง! แนะนำให้รีบทบทวนเนื้อหา")
            else: st.success("พื้นฐานดีมาก! คว้าเกรด A ได้แน่นอน")
        if consent:
            st.session_state.student_db.append({"name": u_name if u_name else "Guest", "uni": u_uni, "year": u_year, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("✅ บันทึกข้อมูลแล้ว")

# --- PAGE 2: GPA ANALYSIS ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    with st.form("gpa_form"):
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1: u_name_gpa = st.text_input("ชื่อ-นามสกุล")
        with col_u2: u_uni_gpa = st.selectbox("มหาวิทยาลัย", uni_options)
        with col_u3: u_year_gpa = st.selectbox("ชั้นปี", [1, 2, 3, 4])
        st.divider()
        st.write("### 📝 กรอกคะแนนรายวิชา (0-100)")
        cols = st.columns(2); all_scores = []
        for i, sub in enumerate(subjects):
            with cols[i % 2]:
                score = st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_input_{i}")
                all_scores.append(score)
        st.divider()
        gpa_consent = st.checkbox("ยินยอมให้บันทึกข้อมูล")
        calc_btn = st.form_submit_button("คำนวณและบันทึกข้อมูล")

    if calc_btn:
        final_gpa = round((sum(all_scores) / len(all_scores) / 100) * 4, 2)
        st.subheader("📊 ผลการวิเคราะห์")
        st.metric("เกรดเฉลี่ยพยากรณ์ (GPA)", f"{final_gpa}")
        if gpa_consent:
            st.session_state.student_db.append({"name": u_name_gpa if u_name_gpa else "Guest Student", "uni": u_uni_gpa, "year": u_year_gpa, "subject": "เฉลี่ยทุกรายวิชา", "midterm": 0, "attendance": 0, "assignment": 0, "final": 0, "total": int(sum(all_scores)/len(all_scores)), "gpa": final_gpa, "entry_type": "gpa_only"})
            st.success("✅ บันทึกข้อมูลแล้ว")

# --- PAGE 3: DB & ANALYTICS (เพิ่มระบบแจ้งเตือนการค้นหา) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    tab1, tab2 = st.tabs(["🔍 ค้นหาคะแนนรายวิชา", "🎓 วิเคราะห์เกรดเฉลี่ย (GPA)"])
    full_df = pd.DataFrame(st.session_state.student_db)

    with tab1:
        st.header("📊 รายงานผลการเรียนรายวิชา")
        search_sub = st.text_input("🔍 ค้นชื่อนักศึกษา (รายวิชา)", placeholder="พิมพ์ชื่อเพื่อค้นหา...", key="search_s1")
        df_sub = full_df[full_df['entry_type'] == 'subject_only'].drop(columns=['gpa', 'entry_type'])
        
        if search_sub:
            # กรองข้อมูล
            search_result = df_sub[df_sub['name'].str.contains(search_sub, case=False, na=False)]
            if not search_result.empty:
                st.success(f"พบข้อมูลของ '{search_sub}' จำนวน {len(search_result)} รายการ")
                st.dataframe(search_result, use_container_width=True)
            else:
                st.error(f"❌ ไม่พบข้อมูลนักศึกษาที่ชื่อ '{search_sub}' ในระบบรายวิชา กรุณาตรวจสอบการสะกดชื่อ")
        else:
            st.dataframe(df_sub, use_container_width=True)

    with tab2:
        st.header("🏆 รายงานเกรดเฉลี่ยสะสม (GPA)")
        search_gpa = st.text_input("🔍 ค้นชื่อนักศึกษา (GPA)", placeholder="พิมพ์ชื่อเพื่อค้นหา...", key="search_s2")
        df_gpa = full_df[full_df['entry_type'] == 'gpa_only'][['name', 'uni', 'year', 'gpa']]
        
        if search_gpa:
            # กรองข้อมูล
            search_result_gpa = df_gpa[df_gpa['name'].str.contains(search_gpa, case=False, na=False)]
            if not search_result_gpa.empty:
                st.success(f"พบข้อมูล GPA ของ '{search_gpa}'")
                st.dataframe(search_result_gpa, use_container_width=True)
            else:
                st.error(f"❌ ไม่พบข้อมูล GPA ของนักศึกษาชื่อ '{search_gpa}' ในระบบ")
        else:
            st.dataframe(df_gpa.sort_values(by='gpa', ascending=False), use_container_width=True)

