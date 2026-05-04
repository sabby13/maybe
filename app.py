import streamlit as st
import tempfile
from pipeline import process_video

st.set_page_config(page_title="Surveillance Analytics", layout="wide")

st.title("🎥 Task-Aware Surveillance Analytics")

uploaded_file = st.file_uploader("Upload a 20-sec surveillance video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.success("Video uploaded successfully!")

    if st.button("Run Analysis"):
        with st.spinner("Processing video..."):
            output_video, log_df = process_video(tfile.name)

        st.subheader("📹 Processed Video")
        st.video(output_video)

        st.subheader("📊 Analytics")
        st.dataframe(log_df)

        # Quick stats
        st.subheader("📌 Summary")
        st.metric("Total People", len(log_df))
        st.metric("Max Duration", f"{log_df['Duration'].max()} sec")