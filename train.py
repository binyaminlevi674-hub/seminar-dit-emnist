import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from vit import Embed, Block


class ViT(nn.Module):
    def __init__(self, dim=64, depth=4, n_classes=47):
        super().__init__()
        self.embed = Embed(dim)
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# EMNIST is stored transposed; fix it so letters are upright.
tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.transpose(1, 2))])
train_ds = torchvision.datasets.EMNIST("data", split="balanced", train=True, download=True, transform=tf)
test_ds = torchvision.datasets.EMNIST("data", split="balanced", train=False, download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512)

torch.manual_seed(0)
model = ViT()
print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 10
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(train_dl))


def accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
    model.train()
    return correct / len(test_ds)


for epoch in range(EPOCHS):
    t0 = time.time()
    for i, (x, y) in enumerate(train_dl):
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if i % 100 == 0:
            print(f"epoch {epoch+1} step {i}/{len(train_dl)} loss {loss.item():.3f}")
    acc = accuracy()
    print(f"=== epoch {epoch+1} done in {time.time()-t0:.0f}s, test accuracy {acc:.4f} ===")

torch.save(model.state_dict(), "vit_emnist.pt")
print("saved vit_emnist.pt")
