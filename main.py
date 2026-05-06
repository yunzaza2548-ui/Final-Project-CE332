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
def merge_sort(data, key, reverse=True): 
    if len(data) <= 1: return data
    mid = len(data) // 2
    left = merge_sort(data[:mid], key, reverse)
    right = merge_sort(data[mid:], key, reverse)
    return merge(left, right, key, reverse)

def merge(left, right, key, reverse):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        val_l = left[i][key]
        val_r = right[j][key]
        condition = val_l >= val_r if reverse else val_l <= val_r
        if condition:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:]); result.extend(right[j:])
    return result

# [SEARCHING]: Binary Search (O(log n))
def binary_search_all(data, key, target):
    low, high = 0, len(data) - 1
    results = []
    target_val = str(target).lower()
    while low <= high:
        mid = (low + high) // 2
        current_val = str(data[mid][key]).lower()
        if target_val in current_val:
            results.append(data[mid])
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
# 3. MOCK DATA & CONSTANTS (THAI NAMES & VARIOUS UNIS)
# ==========================================================

universities = [
    "มหาวิทยาลัยกรุงเทพ", "จุฬาลงกรณ์มหาวิทยาลัย", "มหาวิทยาลัยธรรมศาสตร์", 
    "มหาวิทยาลัยเกษตรศาสตร์", "มหาวิทยาลัยมหิดล", "มหาวิทยาลัยเชียงใหม่", 
    "มหาวิทยาลัยรังสิต", "มหาวิทยาลัยอัสสัมชัญ", "ม.เทคโนโลยีพระจอมเกล้าลาดกระบัง"
]

subjects = ["Computer Programming", "Data Structures", "Digital Logic", "Embedded Systems", "Operating Systems", "Software Engineering", "Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"]

# ลิงก์วิดีโอแนะนำการเรียนต่อยอดในแต่ละวิชา
study_resources = {
    "Computer Programming": "https://www.youtube.com/watch?v=zOjov-2OZ0E",
    "Data Structures": "https://www.youtube.com/watch?v=zg9ih6SVACc",
    "Digital Logic": "https://www.youtube.com/watch?v=M0mx8S05v60",
    "Embedded Systems": "https://youtu.be/xaCAIZKu_zQ?si=ZB-KtJbTPTeemAXX",
    "Operating Systems": "https://www.youtube.com/watch?v=26QPDBe-NB8",
    "Software Engineering": "https://youtu.be/WOPIoZuD1og?si=cl8N86JuOpPwjqQS",
    "Database Systems": "https://youtu.be/6Iu45VZGQDk?si=9K35V44fJsswV0t9",
    "Computer Networks": "https://www.youtube.com/watch?v=IPvYjXCsTg8",
    "Artificial Intelligence": "https://www.youtube.com/watch?v=ad79nYk2keg",
    "Robotics Design": "https://youtu.be/GFLa1b1juUo?si=PYGzvfhmZB4Hx158"
}

if 'student_db' not in st.session_state:
    fnames = ["ทัตเทพ", "ณัฐพงษ์", "สิรินธร", "วรวุฒิ", "กิตติพงษ์", "ชลลดา", "ธนพล", "เบญจมาศ", "วิชุดา", "ภาณุ"]
    lnames = ["ทนันชัย", "ทองดี", "รุ่งเรือง", "สวัสดิ์รักษา", "เจริญพร", "มณีรัตน์", "ปัญญาดี"]
    data = []
    for _ in range(200):
        mid, att, work = random.randint(15, 40), random.randint(7, 10), random.randint(10, 20)
        final = random.randint(10, 30)
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({
            "name": f"{random.choice(fnames)} {random.choice(lnames)}", 
            "uni": random.choice(universities), 
            "year": random.randint(2563, 2568),
            "grade_level": random.randint(1, 4),
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

# --- หน้าที่ 1: พยากรณ์ ---
if page == "พยากรณ์ผลการเรียน":
    st.title("🎯 ระบบพยากรณ์ผลการเรียน (AI Powered)")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            u_name = st.text_input("ชื่อ-นามสกุล")
            u_uni = st.selectbox("มหาวิทยาลัย", universities)
            u_year = st.number_input("ปีการศึกษา", 2560, 2580, 2567)
            u_level = st.slider("ชั้นปี", 1, 4, 1)
            u_sub = st.selectbox("วิชาที่ต้องการพยากรณ์", subjects)
        with col2:
            mid = st.number_input("Midterm (0-40)", 0, 40)
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน/โปรเจกต์ (0-20)", 0, 20)
            consent = st.checkbox("ยินยอมให้บันทึกข้อมูลเพื่อนำไปพัฒนาระบบ AI")
        submit = st.form_submit_button("เริ่มการพยากรณ์ด้วย AI")

    if submit:
        curr, chance, need, pred, acc = predict_with_ml(mid, att, work, st.session_state.student_db)
        st.subheader("📊 ผลการวิเคราะห์และคำแนะนำ")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน", f"{int(chance)}%")
        c2.metric("คะแนนปัจจุบัน", f"{curr}/70")
        c3.metric("ต้องทำ Final", f"{need} คะแนน")
        
        st.info(f"💡 AI แนะนำ: เพื่อผลการเรียนที่ดีขึ้นในวิชา {u_sub} ควรศึกษาเพิ่มเติมจากวิดีโอด้านล่างนี้")
        st.video(study_resources.get(u_sub)) # แสดงวิดีโอตามวิชาที่เลือก
        
        if consent:
            st.session_state.student_db.append({"name": u_name, "uni": u_uni, "year": u_year, "grade_level": u_level, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": curr, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("บันทึกข้อมูลเรียบร้อย")

# --- หน้าที่ 2: วิเคราะห์เกรดเฉลี่ย ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    with st.form("gpa_form"):
        u_name_gpa = st.text_input("ชื่อ-นามสกุล")
        u_uni_gpa = st.selectbox("มหาวิทยาลัย", universities)
        u_year_gpa = st.number_input("ปีการศึกษา", 2560, 2580, 2567)
        u_level_gpa = st.slider("ชั้นปี", 1, 4, 1)
        cols = st.columns(2); all_scores = []
        for i, sub in enumerate(subjects):
            with cols[i % 2]:
                all_scores.append(st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_{i}"))
        gpa_consent = st.checkbox("ยินยอมให้บันทึกข้อมูล GPA เพื่อไปแสดงในหน้า Analytics")
        if st.form_submit_button("คำนวณและบันทึก"):
            gpa = round((sum(all_scores)/len(all_scores)/100)*4, 2)
            st.metric("GPA คาดการณ์", gpa)
            if gpa_consent:
                st.session_state.student_db.append({"name": u_name_gpa, "uni": u_uni_gpa, "year": u_year_gpa, "grade_level": u_level_gpa, "subject": "Overall GPA", "midterm": 0, "attendance": 0, "assignment": 0, "final": 0, "total": 0, "gpa": gpa, "entry_type": "gpa_only"})
                st.success("บันทึกข้อมูล GPA แล้ว")

# --- หน้าที่ 3: Analytics (จัดเรียงได้ทุกคอลัมน์ มากไปน้อย) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    t1, t2 = st.tabs(["🔍 รายชื่อนักศึกษา (รายวิชา)", "🏆 ค้นหา GPA"])

    with t1:
        st.header("📊 รายงานคะแนนรายวิชา")
        sort_key = st.selectbox("จัดเรียงจากมากไปน้อยตาม:", ["year", "total", "midterm", "attendance", "assignment", "grade_level"], key="sort_sub")
        db_sub = [i for i in st.session_state.student_db if i['entry_type'] == 'subject_only']
        sorted_sub = merge_sort(db_sub, sort_key, reverse=True)
        
        search_n = st.text_input("ค้นหาชื่อนักศึกษา (Binary Search)", key="sn1")
        if search_n:
            sorted_for_search = merge_sort(db_sub, 'name', reverse=False)
            res = binary_search_all(sorted_for_search, 'name', search_n)
            if res: st.dataframe(pd.DataFrame(res).drop(columns=['gpa', 'entry_type']), use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        else:
            st.dataframe(pd.DataFrame(sorted_sub).drop(columns=['gpa', 'entry_type']), use_container_width=True)
        
        # --- เพิ่มกราฟในหน้าที่ 3 (Tab 1) ---
        st.divider()
        st.subheader("📈 Visualization: ข้อมูลคะแนนรายวิชา")
        if db_sub:
            df_sub = pd.DataFrame(db_sub)
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # กราฟแท่ง: คะแนนเฉลี่ยแยกตามวิชา
                fig_bar = px.bar(df_sub.groupby('subject')['total'].mean().reset_index(), 
                                 x='subject', y='total', color='subject', title='คะแนนเฉลี่ยแยกตามรายวิชา')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # กราฟกระจาย: Midterm vs Final
                fig_scatter = px.scatter(df_sub, x='midterm', y='final', color='subject', 
                                         size='total', title='ความสัมพันธ์ของ Midterm และ Final')
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_chart2:
                # กราฟวงกลม: สัดส่วนนักศึกษาตามมหาวิทยาลัย
                fig_pie = px.pie(df_sub, names='uni', title='สัดส่วนจำนวนนักศึกษาตามมหาวิทยาลัย')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # กราฟแท่ง: คะแนนเฉลี่ยแยกตามชั้นปี
                fig_year_bar = px.bar(df_sub.groupby('grade_level')['total'].mean().reset_index(), 
                                      x='grade_level', y='total', title='คะแนนเฉลี่ยแยกตามชั้นปี')
                st.plotly_chart(fig_year_bar, use_container_width=True)

    with t2:
        st.header("🏆 รายงานเกรดเฉลี่ย (GPA)")
        sort_key_g = st.selectbox("จัดเรียงจากมากไปน้อยตาม:", ["gpa", "year", "grade_level"], key="sort_gpa")
        db_gpa = [i for i in st.session_state.student_db if i['entry_type'] == 'gpa_only']
        sorted_gpa = merge_sort(db_gpa, sort_key_g, reverse=True)
        
        search_gn = st.text_input("ค้นหาชื่อนักศึกษา (Binary Search)", key="sn2")
        if search_gn:
            sorted_for_search_g = merge_sort(db_gpa, 'name', reverse=False)
            res = binary_search_all(sorted_for_search_g, 'name', search_gn)
            if res: st.dataframe(pd.DataFrame(res)[['name', 'uni', 'year', 'grade_level', 'gpa']], use_container_width=True)
            else: st.error("ไม่พบข้อมูล")
        else:
            st.dataframe(pd.DataFrame(sorted_gpa)[['name', 'uni', 'year', 'grade_level', 'gpa']], use_container_width=True)

        # --- เพิ่มกราฟในหน้าที่ 3 (Tab 2) ---
        st.divider()
        st.subheader("📉 Visualization: ข้อมูลเกรดเฉลี่ย (GPA)")
        if db_gpa:
            df_gpa = pd.DataFrame(db_gpa)
            col_gpa1, col_gpa2 = st.columns(2)
            
            with col_gpa1:
                # กราฟแท่ง: GPA เฉลี่ยตามชั้นปี
                fig_gpa_bar = px.bar(df_gpa.groupby('grade_level')['gpa'].mean().reset_index(), 
                                     x='grade_level', y='gpa', title='GPA เฉลี่ยแยกตามชั้นปี', color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_gpa_bar, use_container_width=True)
                
                # กราฟฮิสโตแกรม: การกระจายตัวของ GPA
                fig_hist = px.histogram(df_gpa, x='gpa', nbins=10, title='การกระจายตัวของเกรดเฉลี่ย (Histogram)', 
                                        labels={'gpa':'เกรดเฉลี่ย'}, color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_gpa2:
                # กราฟกล่อง: ดูการกระจายตัวของ GPA ในแต่ละมหาลัย
                fig_box = px.box(df_gpa, x='uni', y='gpa', title='การกระจายตัวของ GPA แยกตามมหาวิทยาลัย', color='uni')
                st.plotly_chart(fig_box, use_container_width=True)
                
                # กราฟเส้น: แนวโน้ม GPA เฉลี่ยตามปีการศึกษา
                fig_line = px.line(df_gpa.groupby('year')['gpa'].mean().reset_index(), 
                                   x='year', y='gpa', title='แนวโน้ม GPA เฉลี่ยตามปีการศึกษา', markers=True)
                st.plotly_chart(fig_line, use_container_width=True)

