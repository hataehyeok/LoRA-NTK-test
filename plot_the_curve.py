import json
import matplotlib.pyplot as plt
import argparse

def draw_single_plot():
    parser = argparse.ArgumentParser(description="Plot training and evaluation curves based on user-specified rank.")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the files to plot (e.g., 4, 15).")
    args = parser.parse_args()

    rank = args.rank

    eval_file = f"json/eval_qnli_1000_{rank}.json"
    train_file = f"json/qnli_1000_{rank}.json"

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
    plt.savefig(f"pic/eval_qnli_1000_{rank}.png")
    plt.show()

    plt.figure()
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Curve (Rank {rank})")
    plt.legend()
    plt.savefig(f"pic/qnli_1000_{rank}.png")
    plt.show()

def draw_multi_plot():
    ranks = [2, 4, 8, 16]
    colors = ['blue', 'green', 'orange', 'red']

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

if __name__ == "__main__":
    # draw_single_plot()
    draw_multi_plot()