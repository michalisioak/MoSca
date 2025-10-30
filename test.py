import torch
import matplotlib.pyplot as plt

# Parameters
delta = 1.0
errors = torch.linspace(-3, 3, 500)

# Define Huber loss manually
def huber_loss(x, delta):
    abs_x = torch.abs(x)
    quadratic = torch.minimum(abs_x, torch.tensor(delta))
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear

# Compute losses
huber_vals = huber_loss(errors, delta)
l1_vals = torch.abs(errors)
mse_vals = 0.5 * errors**2  # scaled to compare visually

# Plot
plt.figure(figsize=(8, 5))
plt.plot(errors, huber_vals, label=f'Huber Loss', linewidth=2, color='blue')
plt.plot(errors, l1_vals, '--', label='L1 Loss (MAE)', color='orange')
plt.plot(errors, mse_vals, '--', label='0.5*L2 Loss (MSE)', color='green')

# Horizontal lines for ±delta
y_delta = huber_loss(torch.tensor(delta), delta).item()
plt.axhline(y=y_delta, color='gray', linestyle='--', linewidth=1)

# Annotate the delta points
plt.text(delta + 1, y_delta + 0.05, 'robust_huber_delta', color='gray', fontsize=10)


# Labels and title
plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.title('Huber Loss')
plt.legend()
plt.grid(True)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
# Save the figure
plt.tight_layout()
plt.savefig("huber_loss_with_delta.png", dpi=300)
plt.show()

