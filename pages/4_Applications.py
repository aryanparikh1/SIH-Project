import streamlit as st
from utils.page_config import setup_page, render_footer

setup_page("Applications")

st.markdown("""
<div class="reveal">
  <p class="section-heading">Real-World Applications</p>
  <p class="section-sub">Deploying super-resolution for actionable intelligence across sectors.</p>
</div>
""", unsafe_allow_html=True)

# Primary Applications
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="app-card">
        <h3>Crop Monitoring</h3>
        <p style="text-align:left; color:#94a3b8; font-size:0.9rem;">
        Enhancing Sentinel-2's 10m bands allows for precise delineation of field boundaries and early detection of crop stress. 
        High-resolution vegetative indices (like NDVI) enable precision agriculture, optimizing fertilizer application and forecasting yields with greater accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="app-card">
        <h3>Urban Analysis</h3>
        <p style="text-align:left; color:#94a3b8; font-size:0.9rem;">
        Urban planners require high-fidelity data to track infrastructure development and urban sprawl. 
        Super-resolution bridges the gap between free satellite data and expensive aerial imagery, facilitating automated building footprint extraction and road network mapping.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="app-card">
        <h3>Disaster Assessment</h3>
        <p style="text-align:left; color:#94a3b8; font-size:0.9rem;">
        During floods, earthquakes, or wildfires, rapid assessment is critical. 
        Upscaled imagery provides emergency responders with sharper visual data to identify damaged infrastructure, blocked supply routes, and safe evacuation zones faster.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="reveal">
  <p class="section-heading">Impact Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Environmental Impact")
st.write("By providing continuous, high-resolution monitoring without launching new hardware, this solution enables better tracking of deforestation, coastal erosion, and water body health. It allows environmental agencies to detect illegal logging or mining activities that would otherwise remain hidden in low-resolution 10m pixels.")

st.markdown("### Economic Impact")
st.write("Commercial sub-meter satellite imagery is prohibitively expensive for many NGOs, researchers, and developing nations. By mathematically upscaling free open-source Sentinel-2 data, we democratize access to high-quality geospatial intelligence, resulting in massive cost savings and enabling startups to build downstream Earth Observation products.")

st.markdown("### Social Impact")
st.write("Equitable access to data empowers local governments to make informed decisions regarding land rights, informal settlement upgrading, and resource allocation. In rural areas, better agricultural monitoring directly translates to improved food security and livelihood protection for smallholder farmers.")

render_footer()
