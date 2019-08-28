from model import *
from pycocotools.coco import COCO
import torch.optim as optim
# import fastText
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

workers = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 256
learning_rate_g = 0.0002
learning_rate_d = 0.0002
rate_decay_g = 1
rate_decay_d = 1
rate_step_g = 4
rate_step_d = 4
epoch = 30
real_label = (1, 1)
fake_label = (0, 0)

# train discriminator k times before training generator
k = 1

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton'), dtype='int32') - 1

# load text encoding model
# text_model = fastText.load_model(text_model_path)

# get the dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption)

# data loader, containing image ids
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

net_g = Generator().to(device)
net_d = Discriminator().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(first_momentum, second_momentum))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(first_momentum, second_momentum))
criterion = nn.BCEWithLogitsLoss()
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
score_fake2 = None
loss_fake2 = None

# log
writer.add_graph(JoinGAN().to(device), fixed_noise)

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
        optimizer_d.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
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

        # first, optimize discriminator
        net_d.zero_grad()

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real).view(-1)
        # score_wrong = net_d(heatmap_real, text_mismatch).view(-1)

        # calculate losses and update
        label = torch.empty((current_batch_size,), dtype=torch.float32, device=device).uniform_(*real_label)
        loss_right = criterion(score_right, label)
        loss_right.backward()
        # label.fill_(0)
        # criterion(score_wrong, label).mul(0.5).backward()

        # generate heatmaps
        heatmap_fake = net_g(noise)

        # discriminate heatmpap-text pairs
        score_fake = net_d(heatmap_fake.detach()).view(-1)

        # calculate losses and update
        label.uniform_(*fake_label)
        loss_fake = criterion(score_fake, label)
        loss_fake.backward()
        optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_right + loss_fake, batch_number * e + i)
        writer.add_scalar('batch_mean_score/real', score_right.sigmoid().mean(), batch_number * e + i)
        writer.add_scalar('batch_mean_score/fake', score_fake.sigmoid().mean(), batch_number * e + i)
        writer.add_histogram('batch_score/real', score_right.sigmoid(), batch_number * e + i)
        writer.add_histogram('batch_score/fake', score_fake.sigmoid(), batch_number * e + i)

        # second, optimize generator
        if iteration == k:
            iteration = 0

            # get noises
            noise = get_noise_tensor(current_batch_size)
            noise = noise.to(device)

            # generate heatmaps
            heatmap_fake = net_g(noise)
            # heatmap_interpolated = net_g(noise2, text_interpolated)

            net_g.zero_grad()

            # discriminate heatmpap-text pairs
            score_fake2 = net_d(heatmap_fake).view(-1)
            # score_interpolated = net_d(heatmap_interpolated, text_interpolated).view(-1)

            # calculate losses and update
            label.uniform_(*real_label)
            loss_fake2 = criterion(score_fake2, label)
            loss_fake2.backward()
            # label.fill_(1)
            # criterion(score_interpolated, label).backward()
            optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_fake2, batch_number * e + i)
            writer.add_scalar('batch_mean_score/fake_after_d_updated', score_fake2.sigmoid().mean(),
                              batch_number * e + i)
            writer.add_histogram('batch_score/fake_after_d_updated', score_fake2.sigmoid(), batch_number * e + i)

        # print progress
        if score_fake2 is not None:
            print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
                batch_number) + ' score_right: ' + str(
                score_right.sigmoid().mean().item()) + ' score_fake(before): ' + str(
                score_fake.sigmoid().mean().item()) + ' score_fake(after): ' + str(
                score_fake2.sigmoid().mean().item()) + ' g loss: ' + str(
                loss_right.item() + loss_fake.item()) + ' d loss: ' + str(loss_fake2.item()))

        iteration = iteration + 1

    # learning rate scheduling
    scheduler_g.step()
    scheduler_d.step()

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:03d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:03d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    net_d.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise)
        fixed_score = net_d(fixed_fake)
    net_g.train()
    net_d.train()
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    fixed_score = fixed_score.squeeze().sigmoid().tolist()
    f = plt.figure(figsize=(12.8, 9.6))
    for sample in range(4 * 4):
        plt.subplot(4, 4, sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton)
        plt.title(f'{fixed_score[sample]:.3f}')
    plt.savefig('figures/fixed_noise_samples_' + f'{e + 1:03d}' + '.png')

    # log
    writer.add_images('heatmap', np.amax(fixed_fake, 1, keepdims=True), e, dataformats='NCHW')
    writer.add_figure('heatmaps', f, e)

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
