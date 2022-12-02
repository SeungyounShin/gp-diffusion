from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch 

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

import matplotlib.pyplot as plt

x_start = trainer.ds[0][0].unsqueeze(0)

x_t_list = list()
i = 1
plt.figure(figsize=(16,4))
for t in range(0,1001,100):
    print("noise : ", max(t-1,0))
    x_t = diffusion.q_sample(x_start.cuda(), t=torch.tensor([max(t-1,0)]).cuda()).squeeze().detach().cpu()
    
    plt.subplot(1,11,i)
    plt.title(str(max(t-1,0)))
    plt.imshow(x_t, cmap='gray')
    plt.axis('off')

    i += 1

plt.savefig("./results/forward_process.png")