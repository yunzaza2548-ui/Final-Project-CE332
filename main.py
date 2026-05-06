import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random

--- CONFIG & STYLING ---

st.set_page_config(page_title="EduPredic AI Pro", page_icon="🧠", layout="wide")

--- ALGORITHMS ---

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

--- CONSTANTS & MOCK DATA ---

subjects = [
"Computer Programming", "Data Structures", "Digital Logic",
"Embedded Systems", "Operating Systems", "Software Engineering",
"Database Systems", "Computer Networks", "Artificial Intelligence", "Robotics Design"
]

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

--- SIDEBAR NAVIGATION ---

st.sidebar.title("🎓 EduPredic AI Navigation")
page = st.sidebar.radio("เมนูหลัก", ["พยากรณ์ผลการเรียน", "วิเคราะห์เกรดเฉลี่ยรายปี", "ระบบจัดการฐานข้อมูล & Analytics"])

--- PAGE 1: PREDICTION ---

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
st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

--- PAGE 2: GPA CALCULATION ---

elif page == "วิเคราะห์เกรดเฉลี่ยรายปี":
st.title("📉 คำนวณและพยากรณ์เกรดเฉลี่ย (GPA)")
st.info("กรอกคะแนนทั้ง 10 วิชาเพื่อวิเคราะห์เกรดเฉลี่ยรวม")

with st.form("gpa_form"):
u_name_gpa = st.text_input("ชื่อ-นามสกุล")
u_uni_gpa = st.selectbox("มหาวิทยาลัย", uni_options)
u_year_gpa = st.selectbox("ชั้นปี", [1, 2, 3, 4])

st.divider()    
cols = st.columns(2)    
all_scores = []    
for i, sub in enumerate(subjects):    
    with cols[i%2]:    
        score = st.number_input(f"คะแนนวิชา {sub} (0-100)", 0, 100, 50, key=f"sub_{i}")    
        all_scores.append(score)    
    
save_gpa = st.checkbox("บันทึกข้อมูลเกรดเฉลี่ยชุดนี้ลงระบบ")    
calc_btn = st.form_submit_button("คำนวณ GPA")

if calc_btn:
avg_score = sum(all_scores) / 10
final_gpa = round((avg_score / 100) * 4, 2)
st.metric("เกรดเฉลี่ยพยากรณ์", f"{final_gpa}")

if save_gpa:    
    st.session_state.student_db.append({    
        "name": u_name_gpa if u_name_gpa else "Student_New", "uni": u_uni_gpa, "year": u_year_gpa,    
        "subject": "Average (All)", "midterm": 0, "attendance": 0, "assignment": 0,    
        "final": 0, "total": int(avg_score), "gpa": final_gpa    
    })    
    st.success("บันทึกข้อมูลลงระบบเรียบร้อย!")

--- PAGE 3: DATABASE & ANALYTICS ---

elif page == "ระบบจัดการฐานข้อมูล & Analytics":
st.title("📂 ระบบจัดการฐานข้อมูล")

--- Search & Sort Section (Top) ---

st.subheader("📑 ค้นหาและจัดเรียงข้อมูล")

col_s1, col_s2 = st.columns([2, 1])
with col_s1:
search_q = st.text_input("🔍 ค้นหาชื่อนักศึกษา (Binary Search)")

col_o1, col_o2 = st.columns(2)
with col_o1:
sort_opt = st.selectbox("เรียงข้อมูลตาม:", ["name", "total", "gpa", "year"])
with col_o2:
sort_order = st.radio("ลำดับการเรียง:", ["น้อยไปมาก (Ascending)", "มากไปน้อย (Descending)"], horizontal=True)

Process Data

is_reverse = True if "มากไปน้อย" in sort_order else False
sorted_data = merge_sort(st.session_state.student_db, sort_opt, reverse=is_reverse)

if search_q:
search_ready_data = merge_sort(st.session_state.student_db, 'name')
res = binary_search(search_ready_data, search_q)
if res: st.success(f"พบข้อมูล: {res['name']} | มหาวิทยาลัย: {res['uni']} | เกรด: {res['gpa']}")
else: st.error("ไม่พบข้อมูล")

Display Table with Order Index

df_display = pd.DataFrame(sorted_data)
df_display.insert(0, 'ลำดับ', range(1, len(df_display) + 1))

st.dataframe(df_display, use_container_width=True, height=400)
st.caption(f"จำนวนฐานข้อมูลปัจจุบัน: {len(st.session_state.student_db)} รายการ")

st.divider()

--- Analytics Section (Bottom) ---

st.subheader("📈 Analytics Dashboard (สถิติจากข้อมูลในระบบ)")
df_anal = pd.DataFrame(st.session_state.student_db)
col_a, col_b = st.columns(2)
with col_a:
st.plotly_chart(px.pie(df_anal, names='uni', title='สัดส่วนนักศึกษาตามมหาวิทยาลัย', hole=0.4), use_container_width=True)
st.plotly_chart(px.line(df_anal.groupby('year')['total'].mean().reset_index(), x='year', y='total', title='แนวโน้มคะแนนเฉลี่ยตามชั้นปี'), use_container_width=True)
with col_b:
st.plotly_chart(px.scatter(df_anal, x='midterm', y='total', color='subject', title='ความสัมพันธ์ Midterm vs Total Score'), use_container_width=True)
st.plotly_chart(px.histogram(df_anal, x='gpa', title='การกระจายตัวของเกรดเฉลี่ย (GPA Distribution)', color_discrete_sequence=['lightgreen']), use_container_width=True) เพิ่ม ml ลงในนี้เพิ่มแค่mlที่เหลือเหมือนเดิม ขอโค้ดสมบูรณ์ที่เพิ่มมาเลย
