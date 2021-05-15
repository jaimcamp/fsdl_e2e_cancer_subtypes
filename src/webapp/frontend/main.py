"""
File: source.py
Author: yourname
Email: yourname@email.com
Github: https://github.com/yourname
Description:
"""

import streamlit as st
import pandas as pd
import requests as re
import altair as alt

URL="http://backend:8080"

def get_data():
    dataset = pd.read_parquet("/storage/tsne_df.parquet")
    return(dataset)

def main():
    st.title("Genotype explorer")
    st.header("Description")
    st.subheader("This Web app shows the clustering of TCGA cancer data using a DL methodology.")
    st.sidebar.header("Sections")
    selection = st.sidebar.radio(label="Select the page", options=["Explore Data", "Upload Data"])
    if selection == "Explore Data":
        show_explore()
    elif selection == "Upload Data":
        show_upload()
    else:
        pass


def show_explore():
    dataset = get_data()
    st.header("Exploring transcriptomic data")
    chart = alt.Chart(dataset).mark_circle().encode(
        x='First Dimension',
        y='Second Dimension',
        color='Project',
        tooltip=['Project']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def show_upload():
    st.header("Upload your own transcriptomic data")
    file_uploaded = st.file_uploader(label="Choose a file")
    if file_uploaded is not None:
        bytes_data = file_uploaded.getvalue()
        st.write(f"Uploaded file: {file_uploaded.name} ")
        res = re.post(f"{URL}/addData", files={"file": bytes_data})
        st.write(res)
        result = res.json()
        st.write(result)

if __name__=='__main__':
    main()
