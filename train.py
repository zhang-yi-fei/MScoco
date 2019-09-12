from model import *
from pycocotools.coco import COCO
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate_g = 0.0001
learning_rate_d = 0.0001
epoch = 200

# penalty coefficient
lamb = 10

# train discriminator k times before training generator
k = 5

# ADAM solver
beta_1 = 0.0
beta_2 = 0.9

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1

# plot the reference heatmap
x = np.array([5, 6, 4, 7, 3, 7, 3, 8, 2, 9, 1, 7, 3, 8, 2, 9, 1]) / 10 * heatmap_size
y = np.array([2, 1, 1, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]) / 10 * heatmap_size
reference_heatmap = np.empty((total_keypoints, heatmap_size, heatmap_size), dtype='float32')
for i in range(total_keypoints):
    reference_heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / sigma ** 2, dtype='float32')
plt.figure()
plot_heatmap(reference_heatmap, skeleton)
plt.title(None)
plt.xticks([])
plt.yticks([])
plt.savefig('reference_heatmap' + '.png')
plt.close()

# get the dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

net_g = Generator().to(device)
net_d = Discriminator().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))

# fixed noise to see the progression
fixed_h = 4
fixed_w = 4
fixed_size = fixed_h * fixed_w
fixed_noise = get_noise_tensor(fixed_size).to(device)

# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()
iteration = 1
writer = SummaryWriter()
loss_g = torch.tensor(0)
loss_d = torch.tensor(0)

# log
writer.add_graph(JoinGAN().to(device), fixed_noise)

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
        optimizer_d.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # first, optimize discriminator
        net_d.zero_grad()

        # get heatmaps and noises
        heatmap_real = batch.get('heatmap')
        current_batch_size = len(heatmap_real)
        noise = get_noise_tensor(current_batch_size)

        heatmap_real = heatmap_real.to(device)
        noise = noise.to(device)

        # discriminate heatmpaps
        score_right = net_d(heatmap_real)

        # generate heatmaps
        heatmap_fake = net_g(noise).detach()

        # discriminate heatmpaps
        score_fake = net_d(heatmap_fake)

        # random sample
        epsilon = np.random.rand(current_batch_size)
        heatmap_sample = torch.empty_like(heatmap_real)
        for j in range(current_batch_size):
            heatmap_sample[j] = epsilon[j] * heatmap_real[j] + (1 - epsilon[j]) * heatmap_fake[j]
        heatmap_sample.requires_grad = True

        # calculate gradient penalty
        score_sample = net_d(heatmap_sample)
        gradient, = grad(score_sample, heatmap_sample, torch.ones_like(score_sample), create_graph=True)
        gradient_norm = gradient.pow(2).sum((1, 2, 3)).sqrt()

        # calculate losses and update
        loss_d = (score_fake - score_right + lamb * ((gradient_norm - 1).pow(2))).mean()
        loss_d.backward()
        optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * e + i)
        writer.add_histogram('score/real', score_right, batch_number * e + i)
        writer.add_histogram('score/fake', score_fake, batch_number * e + i)
        writer.add_histogram('gradient_norm', gradient_norm, batch_number * e + i)

        # second, optimize generator
        if iteration == k:
            net_g.zero_grad()
            iteration = 0

            # get noises
            noise = get_noise_tensor(current_batch_size)
            noise = noise.to(device)

            # generate heatmaps
            heatmap_fake = net_g(noise)

            # discriminate heatmpaps
            score_fake = net_d(heatmap_fake)

            # discriminate losses and update
            loss_g = -score_fake.mean()
            loss_g.backward()
            optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_g, batch_number * e + i)
            writer.add_histogram('score/fake_2', score_fake, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:03d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:03d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise)
    net_g.train()
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    f = plt.figure(figsize=(12.8, 9.6))
    for sample in range(fixed_size):
        plt.subplot(fixed_h, fixed_w, sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton)
        plt.title(None)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_noise_samples_' + f'{e + 1:03d}' + '.png')

    # log
    writer.add_images('heatmap', np.amax(fixed_fake, 1, keepdims=True), e, dataformats='NCHW')
    writer.add_figure('heatmaps', f, e)

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
