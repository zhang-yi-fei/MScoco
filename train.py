from model import *
from pycocotools.coco import COCO
import torch.optim as optim
# import fastText
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

workers = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 64
learning_rate_g = 0.00005
learning_rate_d = 0.00005
rate_decay_g = 1
rate_decay_d = 1
rate_step_g = 4
rate_step_d = 4
epoch = 500

# weight clipping
c = 0.01

# train discriminator k times before training generator
k = 5

# RMSprop smoothing constant
alpha = 0.99

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton'), dtype='int32') - 1

# load text encoding model
# text_model = fastText.load_model(text_model_path)

# get the dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

net_g = Generator().to(device)
net_d = Discriminator().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.RMSprop(net_g.parameters(), lr=learning_rate_g, alpha=alpha)
optimizer_d = optim.RMSprop(net_d.parameters(), lr=learning_rate_d, alpha=alpha)
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=rate_step_g, gamma=rate_decay_g)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=rate_step_d, gamma=rate_decay_d)

# fixed noise to see the progression
fixed_noise = get_noise_tensor(4 * 4).to(device)

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
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' c ' + str(
        optimizer_d.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # first, optimize discriminator
        net_d.zero_grad()

        # get heatmaps, sentence vectors and noises
        heatmap_real = batch.get('heatmap')
        current_batch_size = len(heatmap_real)
        # text_mismatch = dataset.get_random_caption_tensor(current_batch_size)
        # text_interpolated = dataset.get_interpolated_caption_tensor(current_batch_size)
        noise = get_noise_tensor(current_batch_size)
        # noise2 = get_noise_tensor(current_batch_size)

        heatmap_real = heatmap_real.to(device)
        # text_match = text_match.to(device)
        # text_mismatch = text_mismatch.to(device)
        # text_interpolated = text_interpolated.to(device)
        noise = noise.to(device)
        # noise2 = noise2.to(device)

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real)
        # score_wrong = net_d(heatmap_real, text_mismatch).view(-1)

        # generate heatmaps
        heatmap_fake = net_g(noise).detach()

        # discriminate heatmpap-text pairs
        score_fake = net_d(heatmap_fake)

        # calculate losses and update
        loss_d = -(score_right.mean() - score_fake.mean())
        loss_d.backward()
        optimizer_d.step()

        # clipping
        for p in net_d.parameters():
            p.data.clamp_(-c, c)

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * e + i)
        writer.add_histogram('score/real', score_right, batch_number * e + i)
        writer.add_histogram('score/fake', score_fake, batch_number * e + i)

        # second, optimize generator
        if iteration == k:
            net_g.zero_grad()
            iteration = 0

            # get noises
            noise = get_noise_tensor(current_batch_size)
            noise = noise.to(device)

            # generate heatmaps
            heatmap_fake = net_g(noise)
            # heatmap_interpolated = net_g(noise2, text_interpolated)

            # discriminate heatmpap-text pairs
            score_fake = net_d(heatmap_fake)
            # score_interpolated = net_d(heatmap_interpolated, text_interpolated).view(-1)

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

    # learning rate scheduling
    scheduler_g.step()
    scheduler_d.step()

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
    for sample in range(4 * 4):
        plt.subplot(4, 4, sample + 1)
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
