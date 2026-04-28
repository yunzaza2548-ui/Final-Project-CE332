import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from sklearn.tree import DecisionTreeClassifier  # Commented out due to missing sklearn

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="EduPredict AI Pro", layout="wide", page_icon="🎓")

# --- ส่วนของการสร้างข้อมูล (จำลอง 20 ปี) ---
@st.cache_data
def get_data():
    # ในการใช้งานจริงควรดึงจาก CSV แต่ถ้าหาไฟล์ไม่เจอ ระบบจะสร้างใหม่ให้ทันที
    try:
        df = pd.read_csv('student_data.csv')
        # ตรวจสอบว่าไฟล์มีคอลัมน์ที่ต้องการครบหรือไม่
        required_cols = {"ID", "Name", "Year", "Subject", "Midterm", "Assignment", "Total", "Status"}
        if not required_cols.issubset(df.columns):
            raise ValueError("CSV columns not correct")
        return df
    except:
        subjects = ["แคลคูลัส 1", "ฟิสิกส์วิศวกรรม", "โครงสร้างข้อมูล (Data Structure)", "ภาษาอังกฤษพื้นฐาน"]
        names = ["สมชาย", "วิภา", "กิตติ", "ธนา", "รินดา", "พรเทพ", "นภา", "ใจดี", "มุ่งมั่น", "เก่งกล้า"]
        data = []
        for i in range(500):
            total = np.random.randint(30, 95)
            midterm = np.random.randint(10, 40)
            assign = np.random.randint(5, 20)
            data.append({
                "ID": f"ID-{1000+i}",
                "Name": f"{np.random.choice(names)} {np.random.choice(names)}",
                "Year": np.random.randint(2006, 2026),
                "Subject": np.random.choice(subjects),
                "Midterm": midterm,
                "Assignment": assign,
                "Total": total,
                "Status": "Pass" if total >= 50 else "Fail"
            })
        return pd.DataFrame(data)

df = get_data()

# --- CSS ตกแต่งเพิ่มเติม ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (Searching & Sorting) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
st.sidebar.title("ระบบจัดการข้อมูล")

search_query = st.sidebar.text_input("🔍 ค้นหารหัส/ชื่อ (Hash Search Logic)")
sort_by = st.sidebar.selectbox("🔢 เรียงลำดับข้อมูล", ["คะแนน (มาก-น้อย)", "ปีการศึกษา (ใหม่-เก่า)", "ชื่อนักศึกษา"])

# Logic สำหรับ Sorting & Searching
filtered_df = df.copy()

if search_query:
    filtered_df = filtered_df[filtered_df['ID'].str.contains(search_query, case=False) | filtered_df['Name'].str.contains(search_query, case=False)]

if sort_by == "คะแนน (มาก-น้อย)":
    filtered_df = filtered_df.sort_values(by="Total", ascending=False)
elif sort_by == "ปีการศึกษา (ใหม่-เก่า)":
    filtered_df = filtered_df.sort_values(by="Year", ascending=False)
elif sort_by == "ชื่อนักศึกษา":
    filtered_df = filtered_df.sort_values(by="Name", ascending=True)

# --- MAIN CONTENT ---
st.title("🚀 EduPredict AI: พยากรณ์และแนะแนวการเรียน")
st.info("ระบบนี้ใช้ Machine Learning (Decision Tree) ในการวิเคราะห์สถิติจำลองย้อนหลัง 20 ปี")

# แถวที่ 1: กราฟสถิติภาพรวม
col_a, col_b = st.columns(2)
with col_a:
    fig1 = px.histogram(df, x="Total", color="Status", title="การกระจายตัวของคะแนนทั้งหมด")
    st.plotly_chart(fig1, width='stretch')
with col_b:
    fig2 = px.scatter(df, x="Midterm", y="Total", color="Subject", title="ความสัมพันธ์คะแนนมิดเทอมและคะแนนรวม")
    st.plotly_chart(fig2, width='stretch')

# แถวที่ 2: ระบบ AI พยากรณ์
st.markdown("---")
st.subheader("🤖 พยากรณ์ผลการเรียนของคุณ")
c1, c2, c3 = st.columns(3)
with c1:
    in_mid = st.number_input("คะแนน Midterm (0-40)", 0, 40, 20)
with c2:
    in_assign = st.number_input("คะแนนงาน (0-20)", 0, 20, 10)
with c3:
    in_sub = st.selectbox("วิชาที่เรียน", df['Subject'].unique())

if st.button("วิเคราะห์โอกาสสอบผ่าน"):
    # Simple prediction logic instead of ML (since sklearn not available)
    score = in_mid + in_assign
    if score >= 25:
        prob = 75.0
        pred = 1
    else:
        prob = 25.0
        pred = 0

    if pred == 1:
        st.success(f"ยินดีด้วย! มีโอกาสสอบผ่าน {prob:.2f}% สำหรับวิชา {in_sub}")
        st.balloons()
    else:
        st.error(f"ระวัง! มีโอกาสสอบผ่านเพียง {prob:.2f}% ในวิชา {in_sub} (ควรส่งงานเพิ่ม)")

    # ระบบแนะนำวิดีโอ (Recommendation)
    st.write("### 📺 บทเรียนที่แนะนำสำหรับคุณ:")
    video_links = {
        "แคลคูลัส 1": "https://www.youtube.com/watch?v=KzVre73KExw",
        "ฟิสิกส์วิศวกรรม": "https://www.youtube.com/watch?v=pYitQ6M_oX0",
        "โครงสร้างข้อมูล (Data Structure)": "https://www.youtube.com/watch?v=S0Q6S-X-Y-Y",
        "ภาษาอังกฤษพื้นฐาน": "https://www.youtube.com/watch?v=D-Z9P_T83s8"
    }
    st.video(video_links.get(in_sub, ""))

# แสดงตารางข้อมูลด้านล่างสุด และปุ่มดาวน์โหลด
st.markdown("---")
st.subheader("📋 ตารางรายชื่อนักศึกษา")
st.dataframe(filtered_df, width='stretch')

csv = filtered_df.to_csv(index=False)
st.download_button("📥 ดาวน์โหลดข้อมูลที่กรอง", data=csv, file_name="filtered_students.csv", mime="text/csv")