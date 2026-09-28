import streamlit as st

st.set_page_config(
    page_title="SIH26142 | Sentinel-2 Super-Resolution",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/1_Home.py", title="Home"),
    st.Page("pages/2_Playground.py", title="Playground"),
    st.Page("pages/3_Models.py", title="Models"),
    st.Page("pages/4_Applications.py", title="Applications"),
    st.Page("pages/5_Future_Scope.py", title="Future Scope"),
    st.Page("pages/6_References.py", title="References"),
    st.Page("pages/7_Members.py", title="Members"),
    st.Page("pages/8_Results.py", title="Results")
])

pg.run()
