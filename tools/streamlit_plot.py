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

def draw_multi_plot(selected_dir, ranks, max_epochs):
    st.subheader(f"Training Loss and Accuracy Curves for Directory: {selected_dir}")

    base_dir = os.path.join("exp_results", "csv", selected_dir)
    if not os.path.exists(base_dir):
        st.error(f"Directory {base_dir} does not exist.")
        return

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_accu, ax_accu = plt.subplots(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        train_file = os.path.join(base_dir, f"loss_{rank}.csv")
        eval_file = os.path.join(base_dir, f"accu_{rank}.csv")

        try:
            df_loss = pd.read_csv(train_file)
            df_loss.columns = df_loss.columns.str.strip()
            epoch_loss = df_loss["loss"].tolist()[:max_epochs]
            ax_loss.plot(range(1, len(epoch_loss) + 1), epoch_loss, label=f"Rank {rank}", color=color, linestyle='--')

            df_accu = pd.read_csv(eval_file)
            df_accu.columns = df_accu.columns.str.strip()
            epoch_accu = df_accu["accuracy"].tolist()[:max_epochs]
            ax_accu.plot(range(1, len(epoch_accu) + 1), epoch_accu, label=f"Rank {rank}", color=color, linestyle='--')

        except FileNotFoundError:
            st.warning(f"File for Rank {rank} not found in {selected_dir}. Skipping...")
        except KeyError:
            st.error(f"Key Error in Rank {rank}'s files. Skipping...")

    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    ax_accu.set_xlabel("Epochs")
    ax_accu.set_ylabel("Accuracy")
    ax_accu.set_title("Evaluation Accuracy")
    ax_accu.legend()
    st.pyplot(fig_accu)


st.title("CSV Data Plotter for Multi-Plot")

base_dir = "exp_results/csv"
available_dirs = get_existing_directories(base_dir)

if available_dirs:
    selected_dir = st.selectbox("Select a Directory:", available_dirs)
    ranks = st.multiselect("Select Ranks:", [2, 4, 8, 16, 32, 64, 128, 256, '2_wd0', '4_wd0', '8_wd0', '16_wd0', '32_wd0', '64_wd0', '128_wd0', '256_wd0'])
    max_epochs = st.slider("Maximum Epochs to Display:", 100, 1000, 500)

    if st.button("Generate Multi Plot"):
        draw_multi_plot(selected_dir, ranks, max_epochs)
else:
    st.error(f"No directories found in {base_dir}. Please ensure the data is available.")
