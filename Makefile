all: compile run

compile:
	mpicc -o matrix_mpi main.c

run:
	streamlit run ui.py
