import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

# --- CONFIG & STYLING ---
st.set_page_config(page_title="EduPredict AI Pro", page_icon="🧠", layout="wide")

# ==========================================
# 1. DATA STRUCTURE & ALGORITHMS
# ==========================================

# SORTING: Merge Sort (O(n log n))
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

# SEARCHING: Binary Search (O(log n))
def binary_search(data, target_name):
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid]['name'] == target_name: 
            return data[mid]
        elif data[mid]['name'] < target_name: 
            low = mid + 1
        else: 
            high = mid - 1
    return None

# AI/ML LOGIC: Performance Prediction
def predict_performance(mid, att, work):
    current_total = mid + att + work
    passing_score = 50
    chance = (current_total / 70) * 100
    needed = max(0, passing_score - current_total)
    return current_total, chance, needed

# ==========================================
# 2. CONSTANTS & MOCK DATA
# ==========================================
subjects = [
    "Computer Programming", "Data Structures", "Digital Logic", 
    "Embedded Systems", "Operating Systems", "Software Engineering",
    "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"
]

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

uni_options = ["Bangkok University", "Chulalongkorn University", "Kasetsart University", "Mahidol University", "Thammasat University", "KMUTT", "KMITL", "อื่นๆ"]

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
        # สุ่มประเภทเพื่อกระจายข้อมูลตัวอย่างลงทั้ง 2 รายงาน
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({
            "name": name, "uni": random.choice(uni_options[:-1]),
            "year": random.randint(1, 4), "subject": random.choice(subjects),
            "midterm": mid, "attendance": att, "assignment": work, "final": final,
            "total": total, "gpa": round(random.uniform(2.0, 4.0), 2),
            "entry_type": etype
        })
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(100)

# ==========================================
# 3. USER INTERFACE (UI)
# ==========================================
st.sidebar.title("🎓 EduPredict AI Navigation")
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
        current_total, chance, needed = predict_performance(mid, att, work)
        
        st.subheader("📊 ผลการวิเคราะห์")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(min(chance, 100))}%")
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")
        c3.metric("ต้องทำ Final อีก", f"{needed} คะแนน")

        st.divider()
        st.subheader(f"📚 แนะนำเนื้อหาสำหรับศึกษาเพิ่มเติม: วิชา {u_sub}")
        v_col, t_col = st.columns([3, 2])
        with v_col:
            st.video(study_resources.get(u_sub, "https://www.youtube.com"))
        with t_col:
            st.info("**EduPredict AI Advice:**")
            if chance < 50:
                st.warning(f"คะแนนปัจจุบันของคุณค่อนข้างเสี่ยง แนะนำให้ทบทวนวิดีโอนี้เพื่อเก็บ Final ให้ได้ {needed} คะแนน!")
            else:
                st.success(f"คุณมีพื้นฐานที่ดีมาก! ศึกษาเพิ่มเติมเพื่อคว้าเกรด A ในวิชา {u_sub} ได้เลย")

        if consent:
            # บันทึกเฉพาะ entry_type = "subject_only"
            st.session_state.student_db.append({
                "name": u_name if u_name else "Guest", "uni": u_uni, "year": u_year,
                "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work,
                "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"
            })
            st.success("✅ บันทึกข้อมูลลงใน 'รายงานผลการเรียนรายวิชา' แล้ว")

# --- PAGE 2: GPA ANALYSIS ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    
    with st.form("gpa_form"):
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            u_name_gpa = st.text_input("ชื่อ-นามสกุล")
        with col_u2:
            u_uni_gpa = st.selectbox("มหาวิทยาลัย", uni_options)
        with col_u3:
            u_year_gpa = st.selectbox("ชั้นปี", [1, 2, 3, 4])
            
        st.divider()
        st.write("### 📝 กรอกคะแนนรายวิชา (0-100)")
        
        cols = st.columns(2)
        all_scores = []
        for i, sub in enumerate(subjects):
            with cols[i % 2]:
                score = st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_input_{i}")
                all_scores.append(score)
        
        st.divider()
        gpa_consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อนำไปใช้ในระบบจัดการฐานข้อมูล & Analytics")
        calc_btn = st.form_submit_button("คำนวณและบันทึกข้อมูลลงระบบ")

    if calc_btn:
        avg_score = sum(all_scores) / len(all_scores)
        final_gpa = round((avg_score / 100) * 4, 2)
        
        st.subheader("📊 ผลการวิเคราะห์")
        st.info(f"คุณ **{u_name_gpa if u_name_gpa else 'นักศึกษา'}** มหาวิทยาลัย **{u_uni_gpa}** ชั้นปีที่ **{u_year_gpa}**")
        st.metric("เกรดเฉลี่ยพยากรณ์ (GPA)", f"{final_gpa}")

        if gpa_consent:
            # บันทึกเฉพาะ entry_type = "gpa_only"
            new_data = {
                "name": u_name_gpa if u_name_gpa else "Guest Student",
                "uni": u_uni_gpa,
                "year": u_year_gpa,
                "subject": "เฉลี่ยทุกรายวิชา",
                "midterm": 0,
                "attendance": 0,
                "assignment": 0,
                "final": 0,
                "total": int(avg_score),
                "gpa": final_gpa,
                "entry_type": "gpa_only"
            }
            st.session_state.student_db.append(new_data)
            st.success("✅ บันทึกข้อมูลลงใน 'รายงานเกรดเฉลี่ยสะสม (GPA)' เรียบร้อยแล้ว")
        else:
            st.warning("⚠️ ไม่ได้บันทึกข้อมูล เนื่องจากคุณไม่กดยินยอม")

# --- PAGE 3: DB & ANALYTICS (RESTRUCTURED) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    
    tab1, tab2 = st.tabs(["🔍 ค้นหาคะแนนรายวิชา", "🎓 วิเคราะห์เกรดเฉลี่ย (GPA)"])

    full_df = pd.DataFrame(st.session_state.student_db)

    # --- TAB 1: ค้นหาคะแนนรายวิชา (แสดงเฉพาะข้อมูลจากหน้าพยากรณ์) ---
    with tab1:
        st.header("📊 รายงานผลการเรียนรายวิชา")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            search_sub = st.text_input("🔍 ค้นชื่อนักศึกษา (ดูคะแนนรายวิชา)", key="search_s1")
        with c2:
            sort_sub = st.selectbox("เรียงตามคะแนน(จากมากไปน้อย):", ["total", "midterm", "final"], key="sort_s1")

        # กรองเฉพาะข้อมูลรายวิชา (subject_only) และซ่อน GPA
        df_sub = full_df[full_df['entry_type'] == 'subject_only'].drop(columns=['gpa', 'entry_type'])
        
        if search_sub:
            df_sub = df_sub[df_sub['name'].str.contains(search_sub)]
        
        st.dataframe(df_sub.sort_values(by=sort_sub, ascending=False), use_container_width=True)

        st.divider()
        st.subheader("📈 Analytics: สถิติคะแนน")
        if not df_sub.empty:
            g1, g2 = st.columns(2)
            with g1:
                avg_sub = df_sub.groupby('subject')['total'].mean().reset_index()
                st.plotly_chart(px.bar(avg_sub, x='subject', y='total', color='subject', title='คะแนนเฉลี่ยรวมในแต่ละรายวิชา'), use_container_width=True)
            with g2:
                st.plotly_chart(px.scatter(df_sub, x='midterm', y='final', color='subject', size='total', title='ความสัมพันธ์คะแนนกลางภาค และ ปลายภาค'), use_container_width=True)
        else:
            st.info("ยังไม่มีข้อมูลรายวิชา")

    # --- TAB 2: วิเคราะห์เกรดเฉลี่ย (แสดงเฉพาะข้อมูลจากหน้าวิเคราะห์เกรดเฉลี่ยรายปี) ---
    with tab2:
        st.header("🏆 รายงานเกรดเฉลี่ยสะสม (GPA)")
        
        c3, c4 = st.columns([2, 1])
        with c3:
            search_gpa = st.text_input("🔍 ค้นชื่อนักศึกษา (ดู GPA)", key="search_s2")
        with c4:
            sort_gpa = st.radio("ลำดับ GPA:", ["มากไปน้อย", "น้อยไปมาก"], horizontal=True)

        # กรองเฉพาะข้อมูลเกรดเฉลี่ย (gpa_only)
        df_gpa = full_df[full_df['entry_type'] == 'gpa_only'][['name', 'uni', 'year', 'gpa']]
        
        if search_gpa:
            df_gpa = df_gpa[df_gpa['name'].str.contains(search_gpa)]
        
        is_asc = True if sort_gpa == "น้อยไปมาก" else False
        st.dataframe(df_gpa.sort_values(by='gpa', ascending=is_asc), use_container_width=True)

        st.divider()
        st.subheader("📉 Analytics: วิเคราะห์เกรด")
        if not df_gpa.empty:
            g3, g4 = st.columns(2)
            with g3:
                df_gpa['grade_range'] = pd.cut(df_gpa['gpa'], bins=[0, 2, 3, 3.5, 4], labels=['< 2.0', '2.0-3.0', '3.0-3.5', '3.5-4.0'])
                st.plotly_chart(px.pie(df_gpa, names='grade_range', title='สัดส่วนกลุ่มเกรดเฉลี่ยนักศึกษาทั้งหมด', hole=0.4), use_container_width=True)
            with g4:
                st.plotly_chart(px.box(df_gpa, x='uni', y='gpa', color='uni', title='การกระจายตัวของเกรดแยกตามมหาวิทยาลัย'), use_container_width=True)
            
            avg_year = df_gpa.groupby('year')['gpa'].mean().reset_index()
            st.plotly_chart(px.line(avg_year, x='year', y='gpa', markers=True, title='แนวโน้มเกรดเฉลี่ยเฉลี่ยตามชั้นปี (1-4)'), use_container_width=True)
        else:
            st.info("ยังไม่มีข้อมูล GPA")
