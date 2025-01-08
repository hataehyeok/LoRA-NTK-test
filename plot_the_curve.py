import json
import matplotlib.pyplot as plt

# Load the training losses
with open("training_losses.json", "r") as f:
    epoch_losses = json.load(f)

# Plot the curve
plt.figure()
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.savefig("training_curve.png")
plt.show()