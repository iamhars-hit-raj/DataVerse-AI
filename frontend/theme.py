import streamlit as st


def apply_theme():

    st.markdown("""
<style>

/* Hide Streamlit elements */
header {visibility:hidden;}
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#0F172A;
}

/* Main background */
.main{
    background:#F8FAFC;
}

/* Typography */
h1{
    color:#0F172A;
    font-size:42px;
    font-weight:700;
}

h2,h3{
    color:#1E293B;
}

/* Buttons */
.stButton>button{
    border-radius:12px;
    height:45px;
    font-weight:600;
}

/* Metric Cards */
.metric-card{
    background:white;
    border-radius:18px;
    padding:20px;
    border:1px solid #E2E8F0;
    box-shadow:0 4px 15px rgba(0,0,0,.08);
    transition:.25s;
    height:180px;
}

.metric-card:hover{
    transform:translateY(-5px);
    box-shadow:0 10px 25px rgba(0,0,0,.15);
}

.metric-icon{
    font-size:30px;
}

.metric-title{
    color:#64748B;
    font-size:16px;
}

.metric-value{
    font-size:40px;
    font-weight:bold;
    color:#2563EB;
    margin-top:15px;
}

.metric-footer{
    color:#94A3B8;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)