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

# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0005, epoch = 50
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0001, c = 0.01, epoch = 200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0001, lamb = 10, epoch = 200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0001, lamb = 10, epoch = 200
algorithm = 'wgan-lp'

# weight clipping (WGAN)
c = 0.01

# penalty coefficient (Lipschitz Penalty or Gradient Penalty)
lamb = 10

# train discriminator k times before training generator
k = 5

# ADAM solver
beta_1 = 0.0
beta_2 = 0.9

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)
coco_caption_val = COCO(caption_path_val)
coco_keypoint_val = COCO(keypoint_path_val)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1

# get the dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# data to validate
data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
heatmap_real_val = data_val.get('heatmap').to('cpu')
label_val = torch.full((len(dataset_val),), 1, dtype=torch.float32, device='cpu')

net_g = Generator().to(device)
if algorithm == 'gan':
    net_d = Discriminator(bn=True, sigmoid=True).to(device)
elif algorithm == 'wgan':
    net_d = Discriminator(bn=True).to(device)
else:
    net_d = Discriminator().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))
criterion = nn.BCELoss()

# fixed noise to see the progression
fixed_h = 4
fixed_w = 6
fixed_size = fixed_h * fixed_w
fixed_noise = get_noise_tensor(fixed_size).to(device)

# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()
iteration = 1
writer = SummaryWriter(comment='_pose_' + algorithm)
loss_g = torch.tensor(0)
loss_d = torch.tensor(0)

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

        if algorithm == 'gan':
            label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
            loss_right = criterion(score_right.view(-1), label)
            loss_right.backward()

            label.fill_(0)
            loss_fake = criterion(score_fake.view(-1), label)
            loss_fake.backward()

            # calculate losses and update
            loss_d = loss_right + loss_fake
            optimizer_d.step()
        elif algorithm == 'wgan':
            # calculate losses and update
            loss_d = (score_fake - score_right).mean()
            loss_d.backward()
            optimizer_d.step()

            # clipping
            for p in net_d.parameters():
                p.data.clamp_(-c, c)
        else:
            # 'wgan-gp' and 'wgan-lp'
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
            if algorithm == 'wgan-gp':
                loss_d = (score_fake - score_right + lamb * ((gradient_norm - 1).pow(2))).mean()
            else:
                # 'wgan-lp'
                loss_d = (score_fake - score_right + lamb * (
                    torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()
            loss_d.backward()
            optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * e + i)

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

            if algorithm == 'gan':
                label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
                loss_g = criterion(score_fake.view(-1), label)

                # calculate losses and update
                loss_g.backward()
                optimizer_g.step()
            else:
                # 'wgan', 'wgan-gp' and 'wgan-lp'
                # calculate losses and update
                loss_g = -score_fake.mean()
                loss_g.backward()
                optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_g, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:05d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise)
    net_g.train()
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_size):
        plt.subplot(fixed_h, fixed_w, sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton)
        plt.title(None)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_noise_samples_' + f'{e + 1:05d}' + '.png')

    # validate
    net_g.eval()
    net_d.eval()

    net_g.to('cpu')
    net_d.to('cpu')

    # calculate d loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to('cpu')
    with torch.no_grad():
        score_right_val = net_d(heatmap_real_val).detach()
        heatmap_fake_val = net_g(noise_val).detach()
        score_fake_val = net_d(heatmap_fake_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_right_val = criterion(score_right_val.view(-1), label_val)
        label_val.fill_(0)
        loss_fake_val = criterion(score_fake_val.view(-1), label_val)
        loss_d_val = loss_right_val + loss_fake_val
    elif algorithm == 'wgan':
        loss_d_val = (score_fake_val - score_right_val).mean()
    else:
        # 'wgan-gp' and 'wgan-lp'
        epsilon_val = np.random.rand(dataset_val.__len__())
        heatmap_sample_val = torch.empty_like(heatmap_real_val)
        for j in range(dataset_val.__len__()):
            heatmap_sample_val[j] = epsilon_val[j] * heatmap_real_val[j] + (1 - epsilon_val[j]) * heatmap_fake_val[j]
        heatmap_sample_val.requires_grad = True
        score_sample_val = net_d(heatmap_sample_val)
        gradient_val, = grad(score_sample_val, heatmap_sample_val, torch.ones_like(score_sample_val), create_graph=True)
        gradient_norm_val = gradient_val.pow(2).sum((1, 2, 3)).sqrt()
        if algorithm == 'wgan-gp':
            loss_d_val = (score_fake_val - score_right_val + lamb * ((gradient_norm_val - 1).pow(2))).mean()
        else:
            # 'wgan-lp
            loss_d_val = (score_fake_val - score_right_val + lamb * (
                torch.max(torch.tensor(0, dtype=torch.float32, device='cpu'), gradient_norm_val - 1).pow(2))).mean()

    # calculate g loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to('cpu')
    with torch.no_grad():
        heatmap_fake_val = net_g(noise_val).detach()
        score_fake_val = net_d(heatmap_fake_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_g_val = criterion(score_fake_val.view(-1), label_val)
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        loss_g_val = -score_fake_val.mean()

    # print and log
    print(
        'epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val g loss: ' + str(loss_g_val.item()) + ' val d loss: ' + str(
            loss_d_val.item()))
    writer.add_scalar('loss_val/g', loss_g_val, e)
    writer.add_scalar('loss_val/d', loss_d_val, e)

    net_g.train()
    net_d.train()

    net_g.to(device)
    net_d.to(device)

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
