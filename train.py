import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from vit import Embed, Block


class ViT(nn.Module):
    """The full Vision Transformer: embed -> 4 blocks -> read the class token -> pick 1 of 47 classes."""
    def __init__(self, dim=64, depth=4, n_classes=47):
        super().__init__()
        # Image -> 50 tokens of 64 numbers (49 patches + class token)
        self.embed = Embed(dim)
        # 4 Transformer blocks run one after another
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(depth)])
        # A final LayerNorm before the head
        self.norm = nn.LayerNorm(dim)
        # Classifier head: 64 numbers -> 47 scores, one per class (digits + letters in EMNIST balanced)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.embed(x)
        # (B, 50, 64)
        x = self.blocks(x)
        # (B, 50, 64): through attention, the class token now holds a summary of the whole image
        x = self.norm(x[:, 0])
        # x[:, 0] = only the class token (position 0) -> (B, 64); the 49 patch tokens are dropped here
        # head -> (B, 47): one score (logit) per class; the highest score is the prediction
        return self.head(x)


# --- Data ---
# EMNIST stores images transposed (letters lying on their side); swapping height and width makes them upright.
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.transpose(1, 2))])
# "balanced" split: 47 classes, 112,800 training and 18,800 test images. Downloads into data/ on the first run.
train_ds = torchvision.datasets.EMNIST("data", split="balanced", train=True, download=True, transform=tf)
test_ds = torchvision.datasets.EMNIST("data", split="balanced", train=False, download=True, transform=tf)
# DataLoaders cut the data into batches: 128 images per training step, shuffled every epoch so the order varies.
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
# Test batches can be bigger (512) since no gradients are stored during evaluation.
test_dl = DataLoader(test_ds, batch_size=512)

# --- Model, optimizer, loss ---
# Fixed random seed, so the run is reproducible
torch.manual_seed(0)
model = ViT()
# Counts all learnable numbers in the model (about 207k)
print("parameters:", sum(p.numel() for p in model.parameters()))
# AdamW: the optimizer that updates the weights. lr = step size; weight_decay gently pulls weights toward 0
# to reduce overfitting.
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
# Cross-entropy: the standard classification loss. It's low when the model gives the correct class a high
# score. At the start it's about ln(47) = 3.85, i.e. random guessing among 47 classes.
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 10
# One-cycle schedule: the learning rate warms up to 1e-3, then slowly decays to almost 0 by the last step.
# Big steps early to learn fast, small steps at the end to settle.
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(train_dl))


def accuracy():
    """Fraction of test images classified correctly."""
    # eval mode: switches off training-only behaviour (here: dropout inside the attention layer)
    model.eval()
    correct = 0
    # no_grad: don't track gradients, since we're only measuring, not learning; faster and uses less memory
    with torch.no_grad():
        for x, y in test_dl:
            # argmax(1) = the class with the highest score; count how many match the true label y
            correct += (model(x).argmax(1) == y).sum().item()
    # back to training mode
    model.train()
    return correct / len(test_ds)


# --- Training loop ---
for epoch in range(EPOCHS):
    # one epoch = one full pass over all training images (882 batches of up to 128)
    t0 = time.time()
    for i, (x, y) in enumerate(train_dl):
        # 1) Forward pass: predict scores for the batch and compare them with the true labels y
        loss = loss_fn(model(x), y)
        # 2) Clear the gradients left over from the previous step
        opt.zero_grad()
        # 3) Backpropagation: compute how each weight should change to lower the loss
        loss.backward()
        # 4) Update the weights a small step in that direction
        opt.step()
        # 5) Move the learning rate along the one-cycle schedule
        sched.step()
        # Print progress every 100 batches
        if i % 100 == 0:
            print(f"epoch {epoch+1} step {i}/{len(train_dl)} loss {loss.item():.3f}")
    # After each epoch, measure accuracy on test images the model has never trained on
    acc = accuracy()
    print(f"=== epoch {epoch+1} done in {time.time()-t0:.0f}s, test accuracy {acc:.4f} ===")

# Save the trained weights to a file so the model can be reloaded later without retraining
torch.save(model.state_dict(), "vit_emnist.pt")
print("saved vit_emnist.pt")
