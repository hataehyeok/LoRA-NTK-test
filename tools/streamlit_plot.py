import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def get_existing_directories(base_dir):
    """Get a list of existing subdirectories in the base directory."""
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        return subdirs
    except FileNotFoundError:
        st.error(f"Directory {base_dir} not found.")
        return []

def get_files_in_directory(selected_dir, file_extension=".csv"):
    """Get a list of files with the specified extension in the selected directory."""
    try:
        files = [f for f in os.listdir(selected_dir) if f.endswith(file_extension)]
        return files
    except FileNotFoundError:
        st.error(f"Directory {selected_dir} not found.")
        return []

def draw_selected_plots(selected_dir, selected_files, max_epochs):
    st.subheader(f"Selected Plots from Directory: {selected_dir}")

    base_dir = os.path.join("exp_results", "csv", selected_dir)
    if not os.path.exists(base_dir):
        st.error(f"Directory {base_dir} does not exist.")
        return

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for file, color in zip(selected_files, colors):
        try:
            file_path = os.path.join(base_dir, file)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            # Plot loss or accuracy based on the file type
            if "loss" in file.lower():
                epoch_data = df["loss"].tolist()[:max_epochs]
                ax.plot(range(1, len(epoch_data) + 1), epoch_data, label=file, color=color, linestyle='--')
            elif "accu" in file.lower() or "accuracy" in file.lower():
                epoch_data = df["accuracy"].tolist()[:max_epochs]
                ax.plot(range(1, len(epoch_data) + 1), epoch_data, label=file, color=color, linestyle='-')

        except FileNotFoundError:
            st.warning(f"File {file} not found in {selected_dir}. Skipping...")
        except KeyError:
            st.error(f"Key Error in {file}. Skipping...")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Metric Value")
    ax.set_title("Plots for Selected Files")
    ax.legend()
    st.pyplot(fig)


st.title("Enhanced CSV Data Plotter")

base_dir = "exp_results/csv"
available_dirs = get_existing_directories(base_dir)

if available_dirs:
    selected_dir = st.selectbox("Select a Directory:", available_dirs)

    if selected_dir:
        selected_path = os.path.join(base_dir, selected_dir)
        files_in_dir = get_files_in_directory(selected_path)

        if files_in_dir:
            selected_files = st.multiselect("Select Files to Plot:", files_in_dir)
            max_epochs = st.number_input("Maximum Epochs to Display:", min_value=1, max_value=1000, value=500)

            if st.button("Generate Selected Plots"):
                draw_selected_plots(selected_dir, selected_files, max_epochs)
        else:
            st.error(f"No CSV files found in {selected_dir}. Please check the directory.")
else:
    st.error(f"No directories found in {base_dir}. Please ensure the data is available.")