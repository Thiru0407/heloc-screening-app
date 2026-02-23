import streamlit as st
import pandas as pd
import joblib


# Page config 

st.set_page_config(
    page_title="Simon Bank HELOC Screening Tool",
    layout="centered"
)

# Optional: light custom CSS to make it look cleaner (safe)
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 860px; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); padding: 14px 16px; border-radius: 12px; }
      .stButton button { width: 100%; padding: 0.6rem 1rem; border-radius: 10px; }
      .small-note { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
      .section-title { margin-top: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
pipe_lr = joblib.load("heloc_model.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("Simon Bank HELOC Screening Tool")
st.markdown(
    '<div class="small-note">Decision support prototype for preliminary HELOC screening. Final decisions are made by a loan officer.</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Inputs in sidebar 
# -----------------------------
with st.sidebar:
    st.header("Applicant Inputs")

    ExternalRiskEstimate = st.number_input(
        "External Risk Estimate",
        min_value=0,
        max_value=100,
        value=50
    )

    NumInqLast6M = st.number_input(
        "Inquiries (Last 6 Months)",
        min_value=0,
        max_value=100,
        value=1
    )

    NetFractionRevolvingBurden = st.number_input(
        "Revolving Utilization Ratio",
        min_value=0,
        max_value=300,
        value=50
    )

    NumSatisfactoryTrades = st.number_input(
        "Satisfactory Trades",
        min_value=0,
        max_value=200,
        value=10
    )

    AverageMInFile = st.number_input(
        "Average Months in File",
        min_value=0,
        max_value=600,
        value=100
    )

    st.caption("Tip: Try a low-risk case (90, 0, 10, 25, 120) and a high-risk case (45, 8, 85, 5, 20).")

# -----------------------------
# Main action
# -----------------------------
st.subheader("Screening Result", anchor=False)

if st.button("Check Eligibility"):
    input_data = pd.DataFrame(
        [[ExternalRiskEstimate, NumInqLast6M, NetFractionRevolvingBurden, NumSatisfactoryTrades, AverageMInFile]],
        columns=[
            "ExternalRiskEstimate",
            "NumInqLast6M",
            "NetFractionRevolvingBurden",
            "NumSatisfactoryTrades",
            "AverageMInFile",
        ],
    )

    prob = float(pipe_lr.predict_proba(input_data)[0, 1])
    pred = int(pipe_lr.predict(input_data)[0])

    # Metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Probability", f"{prob:.3f}")
    with col2:
        st.metric("Decision", "Forward to Loan Officer" if pred == 1 else "Not Forwarded")

    # Progress bar 
    st.progress(min(max(prob, 0.0), 1.0))

    st.divider()

    # Explanation logic (aligned with monotonic direction from data dictionary)
    reasons = []
    tips = []

    if ExternalRiskEstimate < 60:
        reasons.append("External risk estimate is low")
        tips.append("Improve credit score over time")

    if NumInqLast6M >= 3:
        reasons.append("Too many recent credit inquiries (last 6 months)")
        tips.append("Avoid applying for new credit for a few months")

    if NetFractionRevolvingBurden >= 50:
        reasons.append("High revolving utilization ratio")
        tips.append("Reduce credit utilization (aim below 30%)")

    if NumSatisfactoryTrades < 10:
        reasons.append("Low number of satisfactory trades")
        tips.append("Build more on-time payment history")

    if AverageMInFile < 60:
        reasons.append("Short credit history (average months in file is low)")
        tips.append("Maintain accounts longer to build credit history")

    # Decision message
    if pred == 1:
        st.success("Application forwarded to loan officer.")
        
        with st.expander("View contributing factors"):
            st.write("This applicant shows relatively favorable risk indicators based on the provided inputs.")
    else:
        st.error("Application not forwarded to loan officer (preliminary screening).")

        
        with st.expander("Main reasons"):
            if reasons:
                for r in reasons:
                    st.write("•", r)
            else:
                st.write("No specific rule-based reasons were triggered for the selected thresholds.")

        with st.expander("How to improve"):
            if tips:
                for t in tips:
                    st.write("•", t)
            else:
                st.write("No improvement suggestions were generated for the selected thresholds.")

    st.info("Note: This tool is for preliminary screening only. Final approval decisions are made by a loan officer.")
else:
    st.caption("Enter applicant inputs in the sidebar, then click 'Check Eligibility'.")
