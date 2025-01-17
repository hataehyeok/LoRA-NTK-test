import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Helper functions
def draw_single_plot(eval_file, train_file, rank):
    try:
        # Read and clean data
        df_accu = pd.read_csv(eval_file)
        df_accu.columns = df_accu.columns.str.strip()
        epoch_accu = df_accu["accuracy"].tolist()

        df_loss = pd.read_csv(train_file)
        df_loss.columns = df_loss.columns.str.strip()
        epoch_loss = df_loss["loss"].tolist()

        # Plot Accuracy
        st.subheader(f"Test Accuracy Curve (Rank {rank})")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(epoch_accu) + 1), epoch_accu, label="Test Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy")
        ax.legend()
        st.pyplot(fig)

        # Plot Loss
        st.subheader(f"Training Loss Curve (Rank {rank})")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        st.pyplot(fig)

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except KeyError as e:
        st.error(f"Error: Missing column in CSV file: {e}")

def draw_multi_plot(csv_dir, ranks, max_epochs):
    st.subheader("Training Loss and Accuracy Curves for Different Ranks")

    # Colors for plots
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_accu, ax_accu = plt.subplots(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        train_file = os.path.join(csv_dir, f"loss_{rank}.csv")
        eval_file = os.path.join(csv_dir, f"accu_{rank}.csv")

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
            st.warning(f"File for Rank {rank} not found. Skipping...")
        except KeyError:
            st.error(f"Key Error in Rank {rank}'s files. Skipping...")

    # Customize and show plots
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

# Streamlit UI
st.title("CSV Data Plotter")

# User selects mode
mode = st.radio("Select Plot Mode:", ["Single Plot", "Multi Plot"])

if mode == "Single Plot":
    eval_file = st.file_uploader("Upload Evaluation CSV:", type="csv")
    train_file = st.file_uploader("Upload Training CSV:", type="csv")
    rank = st.number_input("Enter Rank:", min_value=0, step=1, value=0)

    if eval_file and train_file:
        draw_single_plot(eval_file, train_file, rank)

elif mode == "Multi Plot":
    csv_dir = st.text_input("Enter Directory for CSV files:", "test/csv/qnli")
    ranks = st.multiselect("Select Ranks:", [2, 4, 8, 16, 32, 64, 128, 256])
    max_epochs = st.slider("Maximum Epochs to Display:", 100, 1000, 500)

    if st.button("Generate Multi Plot"):
        draw_multi_plot(csv_dir, ranks, max_epochs)