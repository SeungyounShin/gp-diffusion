from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch 
import matplotlib.pyplot as plt
from celluloid import Camera
from PIL import Image
import numpy as np
import math 

model = Unet(
    dim = 64,
    channels = 1, 
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 28,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'mnist',
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False                       # turn on mixed precision
)

trainer.load(47)

_, recon_list = trainer.model.ddim_sample_animate(batch_size=64) # [len :: 250]

recon_list = [i.detach().squeeze().cpu().numpy() for i in recon_list]

fig, ax = plt.subplots()
camera = Camera(fig)

for i in range(0,len(recon_list)+1,5):
    print(i)
    batch_img_t = recon_list[max(0,i-1)]
    nrow = int(math.sqrt(batch_img_t.shape[0]))
    img_size = batch_img_t.shape[-1]

    grid = np.zeros((img_size*nrow,img_size*nrow))
    cnt = 0 
    for n in range(nrow):
        for m in range(nrow):
            grid[n*img_size:n*img_size+img_size, m*img_size:m*img_size+img_size] = batch_img_t[cnt]
            cnt += 1

    #sample = batch_img_t[0]

    ax.imshow(grid, cmap='gray')
    ax.axis('off')

    camera.snap()

animation = camera.animate(interval=10, repeat=True)
animation.save('./results/animate_diffusion.gif')

