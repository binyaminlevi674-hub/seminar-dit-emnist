import torch
import torch.nn as nn

class Embed(nn.Module):
	def __init__(self, dim=64):
		super().__init__()
		self.proj= nn.Conv2d(1, dim, kernel_size = 4, stride=4)
		self.cls = nn.Parameter(torch.zeros(1, 1, dim))
		self.pos = nn.Parameter(torch.zeros(1, 50, dim))
	def forward(self, x):
		x = self.proj(x)
		x = x.flatten(2)
		x = x.transpose(1, 2)
		cls = self.cls.expand(x.shape[0], -1, -1)
		x = torch.cat([cls, x], dim=1)
		x = x + self.pos
		return x
class Block(nn.Module):
	def __init__(self, dim=64, heads=4):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
		self.norm2 = nn.LayerNorm(dim)
		self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
	def forward(self, x):
		h = self.norm1(x)
		x = x + self.attn(h, h, h)[0]
		x = x + self.ff(self.norm2(x))
		return x

