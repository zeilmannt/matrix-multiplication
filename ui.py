import streamlit as st
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os

st.title("MPI Matrix Multiplication")

os.makedirs("data", exist_ok=True)

N = st.number_input("Matrix size (NxN)", min_value=2, max_value=64, value=4, step=1)
show_heatmap = st.checkbox("Show Heatmap", value=True)
num_procs = st.slider("Number of MPI Processes", min_value=1, max_value=8, value=4)

st.write("Enter Matrix A:")
matrix_a = st.data_editor(pd.DataFrame(np.zeros((N, N))), num_rows="fixed", key="matrix_a")

st.write("Enter Matrix B:")
matrix_b = st.data_editor(pd.DataFrame(np.zeros((N, N))), num_rows="fixed", key="matrix_b")

if st.button("Run MPI Multiplication"):
    # Save matrices
    matrix_a.to_csv("data/matrix_a.csv", index=False, header=False)
    matrix_b.to_csv("data/matrix_b.csv", index=False, header=False)

    try:
        # Run the MPI C program
        result = subprocess.run(
            ["mpirun", "-np", str(num_procs), "./matrix_mpi", str(N)],
            capture_output=True, text=True
        )

        # Result outputs
        #st.text("STDOUT:")
        #st.text(result.stdout)
        #st.text("STDERR:")
        #st.text(result.stderr)

        # Read result
        df = pd.read_csv("data/matrix_result.csv", header=None, usecols=range(N))

        # Show result matrix like inputs
        st.write("Result Matrix:")
        st.data_editor(df, disabled=True, key="matrix_result")

        # Optional heatmap
        if show_heatmap:
            st.write("Heatmap:")
            fig, ax = plt.subplots()
            cax = ax.matshow(df.values, cmap="viridis")
            fig.colorbar(cax)
            st.pyplot(fig)

    except FileNotFoundError:
        st.error("Result file not found.")
