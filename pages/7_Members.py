import streamlit as st
from utils.page_config import setup_page, render_footer

setup_page("Team Members")

st.markdown("""
<div class="reveal">
  <p class="section-heading">The Team</p>
  <p class="section-sub">Minds behind the SIH26142 Super-Resolution project.</p>
</div>
""", unsafe_allow_html=True)

# Grid layout for members
c1, c2, c3 = st.columns(3, gap="large")

# Placeholder member data
members = [
    {"name": "Sasank Bulusu", "role": "ICT", "bio": "24BIT094"},
    {"name": "Aryan Parikh", "role": "CSE", "bio": "24BCP042"},
    {"name": "Aadi Patel", "role": "CSE", "bio": "24BCP020"},
    {"name": "Neel Patel", "role": "ICT", "bio": "24BIT109"},
    {"name": "Prem Murjani", "role": "ICT", "bio": "24BIT091"},
    {"name": "Flora Patidar", "role": "CSE", "bio": "25BCP029"}
]

# Render members
for i, member in enumerate(members):
    col = [c1, c2, c3][i % 3]
    with col:
        # Determine initials for the avatar
        name_parts = member['name'].split()
        initials = (name_parts[0][0] + (name_parts[-1][0] if len(name_parts) > 1 else "")).upper()
        
        role_html = f'<p style="color: #38bdf8; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.2rem;">{member["role"]}</p>' if member["role"] else ''
        
        st.markdown(f"""
        <div class="member-card" style="margin-bottom: 2rem; text-align: center;">
            <div style="width: 80px; height: 80px; background: rgba(56,189,248,.15); border: 2px solid #38bdf8; border-radius: 50%; margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #38bdf8; font-weight: bold;">
                {initials}
            </div>
            <h3 style="margin-bottom: 0.2rem; font-size: 1.1rem; color: #f8fafc;">{member['name']}</h3>
            {role_html}
            <p style="color: #94a3b8; font-size: 0.95rem; font-family: monospace;">{member['bio']}</p>
        </div>
        """, unsafe_allow_html=True)

render_footer()
