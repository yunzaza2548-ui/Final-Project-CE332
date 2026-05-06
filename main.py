import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
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
    # กรองเฉพาะข้อมูลที่มีคะแนนครบ (Entry Type เป็น subject_only และมีคะแนน final)
    train_df = df[(df['entry_type'] == 'subject_only') & (df['final'] > 0)]
    
    current_total = mid + att + work
    passing_score = 50
    needed = max(0, passing_score - current_total)
    
    if len(train_df) > 10:  # ถ้ามีข้อมูลพอให้ Train AI
        X = train_df[['midterm', 'attendance', 'assignment']]
        y = train_df['final']
        model = LinearRegression()
        model.fit(X, y)
        
        # ทำนายคะแนน Final
        pred_final = model.predict([[mid, att, work]])[0]
        pred_final = max(0, min(30, pred_final)) # บีบคะแนนให้อยู่ในกลุ่ม 0-30
        
        chance = ((current_total + pred_final) / 100) * 100
        accuracy = model.score(X, y)
        return current_total, chance, needed, pred_final, accuracy
    else:
        # Fallback เป็น Logic พื้นฐานถ้าข้อมูลไม่พอ
        chance = (current_total / 70) * 100
        return current_total, chance, needed, 0, 0.0

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
        etype = random.choice(["subject_only", "gpa_only"])
        data.append({
            "name": name, "uni": random.choice(uni_options[:-1]),
            "year": random.randint(1, 4), "subject": random.choice(subjects),
            "midterm": mid, "attendance": att, "assignment": work, "final": final,
            "total": mid + att + work + final, "gpa": round(random.uniform(2.0, 4.0), 2),
            "entry_type": etype
        })
    return data

if 'student_db' not in st.session_state:
    st.session_state.student_db = generate_enhanced_mock_data(150)

# ==========================================
# 3. USER INTERFACE (UI)
# ==========================================

st.sidebar.title("🎓 EduPredict AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

# --- PAGE 1: PREDICTION (UI เดิม + ML Logic) ---
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
        # เรียกใช้ AI Prediction
        current_total, chance, needed, pred_final, accuracy = predict_with_ml(mid, att, work, st.session_state.student_db)
        
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")  
        c1, c2, c3 = st.columns(3)  
        c1.metric("โอกาสผ่าน (Estimated)", f"{int(min(chance, 100))}%")  
        c2.metric("คะแนนปัจจุบัน", f"{current_total}/70")  
        c3.metric("ต้องทำ Final อย่างน้อย", f"{needed} คะแนน")  

        if accuracy > 0:
            st.caption(f"💡 AI คาดการณ์คะแนนปลายภาคของคุณที่: {pred_final:.2f} คะแนน (Model Accuracy R²: {accuracy:.2f})")

        st.divider()  
        st.subheader(f"📚 แนะนำเนื้อหาสำหรับศึกษาเพิ่มเติม: วิชา {u_sub}")  
        v_col, t_col = st.columns([3, 2])  
        with v_col:  
            st.video(study_resources.get(u_sub, "https://www.youtube.com"))  
        with t_col:  
            st.info("**EduPredict AI Advice:**")  
            if chance < 50:  
                st.warning(f"คะแนนปัจจุบันค่อนข้างเสี่ยง AI แนะนำให้ทบทวนวิดีโอนี้เพื่อเก็บ Final ให้ได้ {needed} คะแนน!")  
            else:  
                st.success(f"คุณมีพื้นฐานที่ดี! AI คาดว่าคุณจะผ่านได้ไม่ยาก ศึกษาเพิ่มเพื่อคว้าเกรด A ได้เลย")  

        if consent:  
            st.session_state.student_db.append({  
                "name": u_name if u_name else "Guest", "uni": u_uni, "year": u_year,  
                "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work,  
                "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"  
            })  
            st.success("✅ บันทึกข้อมูลลงฐานข้อมูล Training เรียบร้อยแล้ว")

# --- PAGE 2: GPA ANALYSIS (คงเดิม) ---
elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
    st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
    with st.form("gpa_form"):  
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1: u_name_gpa = st.text_input("ชื่อ-นามสกุล")  
        with col_u2: u_uni_gpa = st.selectbox("มหาวิทยาลัย", uni_options)  
        with col_u3: u_year_gpa = st.selectbox("ชั้นปี", [1, 2, 3, 4])  
        
        st.divider()  
        st.write("### 📝 กรอกคะแนนรายวิชา (0-100)")  
        cols = st.columns(2)  
        all_scores = []  
        for i, sub in enumerate(subjects):  
            with cols[i % 2]:  
                score = st.number_input(f"วิชา {sub}", 0, 100, 50, key=f"gpa_input_{i}")  
                all_scores.append(score)  
        
        st.divider()  
        gpa_consent = st.checkbox("ยินยอมให้บันทึกข้อมูลลง Analytics")  
        calc_btn = st.form_submit_button("คำนวณและบันทึกข้อมูล")  

    if calc_btn:  
        avg_score = sum(all_scores) / len(all_scores)  
        final_gpa = round((avg_score / 100) * 4, 2)  
        st.subheader("📊 ผลการวิเคราะห์")  
        st.metric("เกรดเฉลี่ยพยากรณ์ (GPA)", f"{final_gpa}")  

        if gpa_consent:  
            st.session_state.student_db.append({  
                "name": u_name_gpa if u_name_gpa else "Guest Student", "uni": u_uni_gpa, "year": u_year_gpa,  
                "subject": "เฉลี่ยทุกรายวิชา", "midterm": 0, "attendance": 0, "assignment": 0,  
                "final": 0, "total": int(avg_score), "gpa": final_gpa, "entry_type": "gpa_only"  
            })  
            st.success("✅ บันทึกข้อมูล GPA เรียบร้อยแล้ว")

# --- PAGE 3: DB & ANALYTICS (คงเดิม) ---
elif page == "ระบบจัดการฐานข้อมูล & Analytics":
    st.title("📂 ระบบจัดการฐานข้อมูล & Analytics")
    tab1, tab2 = st.tabs(["🔍 ค้นหาคะแนนรายวิชา", "🎓 วิเคราะห์เกรดเฉลี่ย (GPA)"])  
    full_df = pd.DataFrame(st.session_state.student_db)  

    with tab1:  
        st.header("📊 รายงานผลการเรียนรายวิชา")  
        search_sub = st.text_input("🔍 ค้นชื่อนักศึกษา", key="search_s1")  
        df_sub = full_df[full_df['entry_type'] == 'subject_only'].drop(columns=['gpa', 'entry_type'])  
        if search_sub: 
            df_sub = df_sub[df_sub['name'].str.contains(search_sub)]  
        st.dataframe(df_sub, use_container_width=True)  
        
        if not df_sub.empty:
            st.plotly_chart(px.bar(df_sub.groupby('subject')['total'].mean().reset_index(), 
                                   x='subject', y='total', color='subject', title='คะแนนเฉลี่ยแต่ละวิชา'))

    with tab2:  
        st.header("🏆 รายงานเกรดเฉลี่ยสะสม (GPA)")  
        df_gpa = full_df[full_df['entry_type'] == 'gpa_only'][['name', 'uni', 'year', 'gpa']]  
        search_gpa = st.text_input("🔍 ค้นชื่อนักศึกษา (GPA)", key="search_s2")
        if search_gpa:
            df_gpa = df_gpa[df_gpa['name'].str.contains(search_gpa)]
        
        st.dataframe(df_gpa.sort_values(by='gpa', ascending=False), use_container_width=True)  
        
        if not df_gpa.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df_gpa, names='uni', title='สัดส่วนนักศึกษาแยกตามมหาวิทยาลัย'))
            with c2:
                st.plotly_chart(px.box(df_gpa, x='year', y='gpa', title='การกระจายเกรดเฉลี่ยตามชั้นปี'))
            att = st.number_input("เข้าเรียน (0-10)", 0, 10)
            work = st.number_input("งาน (0-20)", 0, 20)
            consent = st.checkbox("บันทึกข้อมูลเพื่อ Train AI")
        submit = st.form_submit_button("Run AI Prediction")

    if submit:
        # เรียกใช้ฟังก์ชัน ML
        current_total, chance, needed, pred_final, accuracy = train_and_predict(mid, att, work, st.session_state.student_db)
        
        st.subheader("📊 ผลการวิเคราะห์จาก AI Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("โอกาสผ่าน (Estimate)", f"{int(min(chance, 100))}%")
        c2.metric("คะแนน Final ที่คาดการณ์", f"{pred_final:.2f}")
        c3.metric("Model Accuracy (R²)", f"{accuracy:.2f}")

        with st.expander("🔍 อธิบายการทำงานของ AI"):
            st.write(f"ระบบใช้ **Linear Regression** ในการเรียนรู้จากข้อมูลนักศึกษาเก่า {len(st.session_state.student_db)} คน")
            st.write(f"โดยวิเคราะห์ว่าคะแนน Midterm และงานส่ง มีผลต่อคะแนน Final อย่างไร")
            st.progress(accuracy if accuracy > 0 else 0, text="ความแม่นยำของ Model")

        st.divider()
        v_col, t_col = st.columns([3, 2])
        with v_col: st.video(study_resources.get(u_sub))
        with t_col: 
            st.info(f"AI แนะนำ: เพื่อให้ผ่านเกณฑ์ 50 คะแนน คุณต้องทำคะแนนสอบปลายภาคให้ได้อย่างน้อย {needed} คะแนน")

        if consent:
            st.session_state.student_db.append({"name": u_name, "uni": "Guest", "year": 1, "subject": u_sub, "midterm": mid, "attendance": att, "assignment": work, "final": 0, "total": current_total, "gpa": 0.0, "entry_type": "subject_only"})
            st.success("บันทึกข้อมูลเข้าสู่ Training Set แล้ว")

# (ส่วน Page 2 และ Page 3 คงเดิมตามโค้ดก่อนหน้าได้เลย)

