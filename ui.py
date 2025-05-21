import streamlit as st
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np

st.title("MPI Matrix Multiplication")

N = st.number_input("Matrix size (NxN)", min_value=2, max_value=64, value=4, step=1)

st.write("Enter Matrix A:")
matrix_a = st.data_editor(pd.DataFrame(np.zeros((N, N))), num_rows="fixed", key="matrix_a")

st.write("Enter Matrix B:")
matrix_b = st.data_editor(pd.DataFrame(np.zeros((N, N))), num_rows="fixed", key="matrix_b")

if st.button("Run MPI Multiplication"):
    matrix_a.to_csv("matrix_a.csv", index=False, header=False)
    matrix_b.to_csv("matrix_b.csv", index=False, header=False)

    try:
        result = subprocess.run(["mpirun", "-np", "4", "./matrix_mpi", str(N)],
                                capture_output=True, text=True)
        st.text(result.stdout)

        df = pd.read_csv("matrix_result.csv", header=None)
        st.write("Result Matrix:")
        st.dataframe(df)

        st.write("Heatmap:")
        fig, ax = plt.subplots()
        cax = ax.matshow(df.values, cmap="viridis")
        fig.colorbar(cax)
        st.pyplot(fig)
    except FileNotFoundError:
        st.error("Result file not found.")
