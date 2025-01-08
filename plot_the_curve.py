import json
import matplotlib.pyplot as plt
import argparse

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
