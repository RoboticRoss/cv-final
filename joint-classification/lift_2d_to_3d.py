import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ABANDONDED THIS CAUSE IT AINT WORKIN, JUST USE THE DEPTH CAMERA LOL ;-;

# mlp model
class PoseLifter(nn.Module):
    def __init__(self, input_dim=6, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# load points
data = np.load('pose2d_mediapipe.npz', allow_pickle=True)['positions_2d'].item()
sequence_2d = data['S1']['Walking']  # shape (T, 3, 2)
T = sequence_2d.shape[0]

# flatten
sequence_2d_flat = sequence_2d.reshape(T, -1)
inputs = torch.tensor(sequence_2d_flat, dtype=torch.float32)

model = PoseLifter()
# model.load_state_dict(torch.load('pose_lifter_weights.pth', map_location='cpu'))
print("Using randomly initialized weights — expect nonsense, but good for testing.")

model.eval()

with torch.no_grad():
    pred_3d_flat = model(inputs)  # shape: (T, 9)
    pred_3d = pred_3d_flat.view(T, 3, 3)  # (T, 3 joints, 3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'green', 'red']
labels = ['Shoulder', 'Elbow', 'Wrist']
for i in range(3):
    joint = pred_3d[:, i]
    ax.plot(joint[:, 0], joint[:, 1], joint[:, 2], label=labels[i], color=colors[i])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lifted 3D Pose (Right Arm)")
ax.legend()
plt.show()
