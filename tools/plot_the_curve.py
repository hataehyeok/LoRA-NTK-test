import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def draw_single_plot(mode, rank):
    if rank != 0:
        eval_file = f"test/csv/qnli/accu_{rank}.csv"
        train_file = f"test/csv/qnli/loss_{rank}.csv"
    else:
        eval_file = f"test/csv/qnli/accu_full.csv"
        train_file = f"test/csv/qnli/loss_full.csv"
    
    eval_dir = os.path.dirname(eval_file)
    train_dir = os.path.dirname(train_file)
    pic_dir = f"test/pic_test/qnli"

    for directory in [eval_dir, train_dir, pic_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")

    try:
        df_accu = pd.read_csv(eval_file)
        df_accu.columns = df_accu.columns.str.strip()
        print("Evaluation CSV columns after stripping:", df_accu.columns)
        epoch_accu = df_accu["accuracy"].tolist()
        
        df_loss = pd.read_csv(train_file)
        df_loss.columns = df_loss.columns.str.strip()
        print("Training CSV columns after stripping:", df_loss.columns)
        epoch_loss = df_loss["loss"].tolist()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing column in CSV file: {e}")
        exit(1)

    plt.figure()
    plt.plot(range(1, len(epoch_accu) + 1), epoch_accu, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Test Curve (Rank {rank})")
    plt.legend()
    accu_output_path = os.path.join(pic_dir, f"accu_{rank}.png")
    plt.savefig(accu_output_path)
    print(f"Saved Test Accuracy plot: {accu_output_path}")
    plt.show()

    plt.figure()
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Curve (Rank {rank})")
    plt.legend()
    loss_output_path = os.path.join(pic_dir, f"loss_{rank}.png")
    plt.savefig(loss_output_path)
    print(f"Saved Training Loss plot: {loss_output_path}")
    plt.show()

def draw_multi_plot():
    # ranks = [2, 4, 8, 16, 32, 64, 128, 256]
    # colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    ranks = [2, 4]
    colors = ["red", "blue"]
    max_epochs = 500

    plt.figure(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        if rank == 777:
            rank = "full"
        train_file = f"test/csv/cr/loss_{rank}.csv"
        try:
            df_loss = pd.read_csv(train_file)
            df_loss.columns = df_loss.columns.str.strip()
            print("Training CSV columns after stripping:", df_loss.columns)
            epoch_loss = df_loss["loss"].tolist()[:max_epochs]
            plt.plot(
                range(1, len(epoch_loss) + 1),
                epoch_loss,
                label=f"Rank {rank}",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {train_file} not found. Skipping...")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves for Different Ranks")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_all_ranks.png")
    plt.show()

    plt.figure(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        if rank == 777:
            rank = "full"
        eval_file = f"test/csv/cr/accu_{rank}.csv"
        try:
            df_accu = pd.read_csv(eval_file)
            df_accu.columns = df_accu.columns.str.strip()
            print("Evaluation CSV columns after stripping:", df_accu.columns)
            epoch_accu = df_accu["accuracy"].tolist()[:max_epochs]

            plt.plot(
                range(1, len(epoch_accu) + 1),
                epoch_accu,
                label=f"Rank {rank}",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {eval_file} not found. Skipping...")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Curves for Different Ranks")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluate_accuracy_all_ranks.png")
    plt.show()

def draw_wd_multi_plot():
    # ranks = [2, 4, 8, 16, 32, 64, 128, 256]
    # colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    ranks = [2]
    colors = ["red", "blue"]
    wd_colors = ["green", "orange"]
    max_epochs = 500

    plt.figure(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        if rank == 777:
            rank = "full"
        train_file = f"fin_data/csv/subj/loss_{rank}.csv"
        try:
            df_loss = pd.read_csv(train_file)
            df_loss.columns = df_loss.columns.str.strip()
            print("Training CSV columns after stripping:", df_loss.columns)
            epoch_loss = df_loss["loss"].tolist()[:max_epochs]
            plt.plot(
                range(1, len(epoch_loss) + 1),
                epoch_loss,
                label=f"Rank {rank}",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {train_file} not found. Skipping...")

    for rank, color in zip(ranks, wd_colors):
        if rank == 777:
            rank = "full"
        train_file = f"fin_data/csv/subj/loss_{rank}_wd0.csv"
        try:
            df_loss = pd.read_csv(train_file)
            df_loss.columns = df_loss.columns.str.strip()
            print("Training CSV columns after stripping:", df_loss.columns)
            epoch_loss = df_loss["loss"].tolist()[:max_epochs]
            plt.plot(
                range(1, len(epoch_loss) + 1),
                epoch_loss,
                label=f"Rank {rank}_wd0",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {train_file} not found. Skipping...")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves for Different Ranks")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_all_ranks.png")
    plt.show()

    plt.figure(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        if rank == 777:
            rank = "full"
        eval_file = f"fin_data/csv/subj/accu_{rank}.csv"
        try:
            df_accu = pd.read_csv(eval_file)
            df_accu.columns = df_accu.columns.str.strip()
            print("Evaluation CSV columns after stripping:", df_accu.columns)
            epoch_accu = df_accu["accuracy"].tolist()[:max_epochs]

            plt.plot(
                range(1, len(epoch_accu) + 1),
                epoch_accu,
                label=f"Rank {rank}",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {eval_file} not found. Skipping...")

    for rank, color in zip(ranks, wd_colors):
        if rank == 777:
            rank = "full"
        eval_file = f"fin_data/csv/subj/accu_{rank}_wd0.csv"
        try:
            df_accu = pd.read_csv(eval_file)
            df_accu.columns = df_accu.columns.str.strip()
            print("Evaluation CSV columns after stripping:", df_accu.columns)
            epoch_accu = df_accu["accuracy"].tolist()[:max_epochs]

            plt.plot(
                range(1, len(epoch_accu) + 1),
                epoch_accu,
                label=f"Rank {rank}_wd0",
                color=color,
                linestyle='--'
            )
        except FileNotFoundError:
            print(f"File {eval_file} not found. Skipping...")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Curves for Different Ranks")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluate_accuracy_all_ranks.png")
    plt.show()


def parser():
    parser = argparse.ArgumentParser(description="Plot training and evaluation curves based on user-specified rank.")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the script (s for single plot, m for multi plot).")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the files to plot (e.g., 4, 16).")
    args = parser.parse_args()

    if args.mode == "s":
        draw_single_plot(args.mode, args.rank)
    elif args.mode == "m":
        # draw_multi_plot()
        draw_wd_multi_plot()
    else:
        print("Invalid mode. Please specify 's' for single plot or 'm' for multi plot.")


def dub():    
    file_path = "accu_2.csv"
    df = pd.read_csv(file_path)
    df_cleaned = df.drop_duplicates(subset=["epoch"], keep="last")
    df_cleaned.to_csv(file_path, index=False)
    print(f"Cleaned data saved to {file_path}")

if __name__ == "__main__":
    parser()