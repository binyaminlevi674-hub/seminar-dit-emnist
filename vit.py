import torch
import torch.nn as nn

# Shapes below use B = batch size (number of images processed at once).


class Embed(nn.Module):
	"""Turns an image into a sequence of tokens the Transformer can read.

	28x28 image -> 49 patch tokens (a 7x7 grid of 4x4 patches) -> + 1 class token = 50 tokens of 64 numbers each.
	"""
	def __init__(self, dim=64):
		super().__init__()
		# Patching: a convolution with kernel 4 and stride 4 looks at each 4x4 patch exactly once
		# (no overlap) and maps its 16 pixels to a 64-number vector. This is the same as cutting the
		# image into patches and putting each through one shared linear layer.
		self.proj= nn.Conv2d(1, dim, kernel_size = 4, stride=4)
		# Class token: one extra learnable token, the "summary slot". It starts as zeros, and through
		# attention it collects information from all patches. The classifier head later reads only this token.
		self.cls = nn.Parameter(torch.zeros(1, 1, dim))
		# Positional embedding: one learnable 64-number vector per position (49 patches + 1 class token = 50).
		# Attention itself has no sense of order, so without this the model wouldn't know where a patch came from.
		self.pos = nn.Parameter(torch.zeros(1, 50, dim))
	def forward(self, x):
		# x: (B, 1, 28, 28), one grayscale channel
		x = self.proj(x)
		# (B, 64, 7, 7): a 7x7 grid of patches, each now a 64-number vector
		x = x.flatten(2)
		# (B, 64, 49): the 7x7 grid flattened into a row of 49 patches
		x = x.transpose(1, 2)
		# (B, 49, 64): swap the axes so it's "49 tokens, each with 64 numbers", the format attention expects
		cls = self.cls.expand(x.shape[0], -1, -1)
		# (B, 1, 64): the same class token copied once for every image in the batch
		x = torch.cat([cls, x], dim=1)
		# (B, 50, 64): class token put in front of the 49 patch tokens
		x = x + self.pos
		# (B, 50, 64): add the position info; the shape stays the same
		return x


class Block(nn.Module):
	"""One Transformer block (pre-norm). The model stacks 4 of these.

	Step 1, attention: every token looks at all the other tokens and mixes in information from them.
	Step 2, feed-forward (MLP): each token is processed on its own, to "think about" what it just collected.
	Both steps are residual: x = x + change, so a block only adds a correction on top of its input.
	Input and output shape are the same: (B, 50, 64).
	"""
	def __init__(self, dim=64, heads=4):
		super().__init__()
		# LayerNorm rescales each token's 64 numbers to mean 0 and spread 1 (with learnable scale gamma and
		# shift beta). It keeps the values in a stable range so training doesn't blow up.
		self.norm1 = nn.LayerNorm(dim)
		# Multi-head self-attention with 4 heads: 4 parallel attention patterns, each working on 16 of the
		# 64 numbers, so different heads can focus on different relationships between patches.
		self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
		self.norm2 = nn.LayerNorm(dim)
		# Feed-forward network: widen each token 64 -> 256, apply GELU (a smooth non-linearity), shrink back 256 -> 64.
		self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
	def forward(self, x):
		# x: (B, 50, 64)
		h = self.norm1(x)
		# Self-attention: query, key and value all come from the same tokens h. [0] takes the output and
		# ignores the attention weights PyTorch also returns. Residual: add the result onto the original x.
		x = x + self.attn(h, h, h)[0]
		# Normalize again, run the feed-forward network, and add the result back on (second residual).
		x = x + self.ff(self.norm2(x))
		# (B, 50, 64): same shape as the input, so blocks can be stacked
		return x
