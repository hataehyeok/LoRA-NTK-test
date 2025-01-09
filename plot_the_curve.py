import os
import json
import matplotlib.pyplot as plt
import argparse

def draw_single_plot(rank):
    eval_file = f"json_test/qnli/accu_{rank}.json"
    train_file = f"json_test/qnli/loss_{rank}.json"
    eval_dir = os.path.dirname(eval_file)
    train_dir = os.path.dirname(train_file)
    pic_dir = f"pic_test/qnli"

    for directory in [eval_dir, train_dir, pic_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")

    try:
        with open(eval_file, "r") as f:
            epoch_accu = json.load(f)
        with open(train_file, "r") as f:
            epoch_loss = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
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
    # ranks = [0, 4, 16, 64, 512]
    # colors = ["red", "blue", "green", "orange", "purple"]

    ranks = [2, 16]
    colors = ["red", "blue"]

    plt.figure(figsize=(10, 6))

    for rank, color in zip(ranks, colors):
        train_file = f"json/qnli_1000_{rank}.json"
        try:
            with open(train_file, "r") as f:
                epoch_loss = json.load(f)
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
        eval_file = f"json/eval_qnli_1000_{rank}.json"
        try:
            with open(eval_file, "r") as f:
                epoch_accuracy = json.load(f)
            plt.plot(
                range(1, len(epoch_accuracy) + 1),
                epoch_accuracy,
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

def parser():
    parser = argparse.ArgumentParser(description="Plot training and evaluation curves based on user-specified rank.")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the script (s for single plot, m for multi plot).")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the files to plot (e.g., 4, 15).")
    args = parser.parse_args()

    if args.mode == "s":
        draw_single_plot(args.rank)
    elif args.mode == "m":
        draw_multi_plot()
    else:
        print("Invalid mode. Please specify 's' for single plot or 'm' for multi plot.")

if __name__ == "__main__":
    parser()
