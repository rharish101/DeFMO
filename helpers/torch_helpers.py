import torch
from torch.nn import Module, functional as F

def get_images(encoder: Module, rendering: Module, device: torch.device, val_batch: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		latent = encoder(val_batch)
		times = torch.linspace(0,1,2).to(device)
		renders = rendering(latent,times[None])

	renders = renders.cpu().numpy()
	renders = renders[:,:,3:4]*(renders[:,:,:3]-1)+1
	return renders

def normalized_cross_correlation_channels(image1, image2):
	mean1 = image1.mean([2,3,4],keepdims=True)
	mean2 = image2.mean([2,3,4],keepdims=True) 
	std1 = image1.std([2,3,4],unbiased=False,keepdims=True)
	std2 = image2.std([2,3,4],unbiased=False,keepdims=True)
	eps=1e-8
	bs, ts, *sh = image1.shape
	N = sh[0]*sh[1]*sh[2]
	im1b = ((image1-mean1)/(std1*N+eps)).view(bs*ts, sh[0], sh[1], sh[2])
	im2b = ((image2-mean2)/(std2+eps)).reshape(bs*ts, sh[0], sh[1], sh[2])
	padding = tuple(side // 10 for side in sh[:2]) + (0,)
	result = F.conv3d(im1b[None], im2b[:,None], padding=padding, bias=None, groups=bs*ts)
	ncc = result.view(bs*ts, -1).max(1)[0].view(bs, ts)
	return ncc
