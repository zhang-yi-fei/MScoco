from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# whether multi-person
multi = False

# training parameters
batch_size = 128
learning_rate_g = 0.0004
learning_rate_d = 0.0004
start_from_epoch = 200
end_in_epoch = 1200

# algorithms: gan, wgan, wgan-gp, wgan-lp
# gan: k = 1, beta_1 = 0.5, beta_2 = 0.999, lr = 0.0001, epoch = 50~300
# wgan: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.001, c = 0.01, epoch = 200~1200
# wgan-gp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 20, epoch = 200~1200
# wgan-lp: k = 5, beta_1 = 0, beta_2 = 0.9, lr = 0.0004, lamb = 150, epoch = 200~1200
algorithm = 'wgan-gp'

# weight clipping (WGAN)
c = 0.01

# penalty coefficient (Lipschitz Penalty or Gradient Penalty)
lamb = 10

# level of text-image matching
alpha = 1

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

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get the dataset (single person, with captions)
dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=not multi, text_model=text_model, full_image=multi)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, single_person=not multi, text_model=text_model,
                             full_image=multi)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# data to validate
data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
text_match_val = data_val.get('vector').to(device)
heatmap_real_val = data_val.get('heatmap').to(device)
label_val = torch.full((len(dataset_val),), 1, dtype=torch.float32, device=device)

net_g = Generator2().to(device)
if algorithm == 'gan':
    net_d = Discriminator2(bn=True, sigmoid=True).to(device)
elif algorithm == 'wgan':
    net_d = Discriminator2(bn=True).to(device)
else:
    net_d = Discriminator2().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

# load first step (without captions) trained weights if available
if start_from_epoch > 0:
    net_g.load_state_dict(torch.load(generator_path + '_' + f'{start_from_epoch:05d}'), False)
    net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{start_from_epoch:05d}'), False)
    net_g.first2.weight.data[0:noise_size] = net_g.first.weight.data
    net_d.second2.weight.data[:, 0:convolution_channel_d[-1], :, :] = net_d.second.weight.data
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))
criterion = nn.BCELoss()

# fixed training data (from validation set), noise and sentence vectors to see the progression
fixed_h = 6
fixed_w = 5
fixed_size = fixed_h * fixed_w
fixed_train = dataset_val.get_random_heatmap_with_caption(fixed_w)
fixed_real = fixed_train.get('heatmap').to(device)
fixed_real_array = np.array(fixed_real.tolist()) * 0.5 + 0.5
fixed_caption = fixed_train.get('caption')
fixed_noise = get_noise_tensor(fixed_h).to(device)
fixed_text = torch.tensor([get_caption_vector(text_model, caption) for caption in fixed_caption], dtype=torch.float32,
                          device=device).unsqueeze(-1).unsqueeze(-1)

# save models before training
torch.save(net_g.state_dict(), generator_path + '_' + f'{start_from_epoch:05d}' + '_new')
torch.save(net_d.state_dict(), discriminator_path + '_' + f'{start_from_epoch:05d}' + '_new')

# plot and save generated samples from fixed noise (before training begins)
net_g.eval()
with torch.no_grad():
    fixed_fake = net_g(fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1, 1, 1))
net_g.train()
fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
f = plt.figure(figsize=(19.2, 12))
for sample in range(fixed_w):
    plt.subplot(fixed_h + 1, fixed_w, sample + 1)
    plot_heatmap(fixed_real_array[sample], skeleton=(None if multi else skeleton))
    plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
    plt.xlabel('(real)')
    plt.xticks([])
    plt.yticks([])
for sample in range(fixed_size):
    plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
    plot_heatmap(fixed_fake[sample], skeleton=(None if multi else skeleton))
    plt.title(None)
    plt.xlabel('(fake)')
    plt.xticks([])
    plt.yticks([])
plt.savefig('figures/fixed_noise_samples_' + f'{start_from_epoch:05d}' + '_new.png')
plt.close()

# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()
iteration = 1
writer = SummaryWriter(comment='_caption_' + ('multi_' if multi else '') + algorithm)
loss_g = torch.tensor(0)
loss_d = torch.tensor(0)

# number of batches
batch_number = len(data_loader)

for e in range(start_from_epoch, end_in_epoch):
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
        optimizer_d.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # first, optimize discriminator
        net_d.zero_grad()

        # get heatmaps, sentence vectors and noises
        heatmap_real = batch.get('heatmap')
        text_match = batch.get('vector')
        current_batch_size = len(heatmap_real)
        text_mismatch = dataset.get_random_caption_tensor(current_batch_size)
        noise = get_noise_tensor(current_batch_size)

        heatmap_real = heatmap_real.to(device)
        text_match = text_match.to(device)
        text_mismatch = text_mismatch.to(device)
        noise = noise.to(device)

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real, text_match)
        score_wrong = net_d(heatmap_real, text_mismatch)

        # generate heatmaps
        heatmap_fake = net_g(noise, text_match).detach()

        # discriminate heatmpap-text pairs
        score_fake = net_d(heatmap_fake, text_match)

        if algorithm == 'gan':
            label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
            loss_right = criterion(score_right.view(-1), label) * (1 + alpha)
            loss_right.backward()

            label.fill_(0)
            loss_fake = criterion(score_fake.view(-1), label)
            loss_fake.backward()

            label.fill_(0)
            loss_wrong = criterion(score_wrong.view(-1), label) * alpha
            loss_wrong.backward()

            # calculate losses and update
            loss_d = loss_right + loss_fake + loss_wrong
            optimizer_d.step()
        elif algorithm == 'wgan':
            # calculate losses and update
            loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right).mean()
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
            text_match.requires_grad = True

            # calculate gradient penalty
            score_sample = net_d(heatmap_sample, text_match)
            gradient_h, gradient_t = grad(score_sample, [heatmap_sample, text_match], torch.ones_like(score_sample),
                                          create_graph=True)
            gradient_norm = (gradient_h.pow(2).sum((1, 2, 3)) + gradient_t.pow(2).sum((1, 2, 3))).sqrt()

            # calculate losses and update
            if algorithm == 'wgan-gp':
                loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
                    (gradient_norm - 1).pow(2))).mean()
            else:
                # 'wgan-lp'
                loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
                    torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()

            loss_d.backward()
            optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * (e - start_from_epoch) + i)

        # second, optimize generator
        if iteration == k:
            net_g.zero_grad()
            iteration = 0

            # get sentence vectors and noises
            text_interpolated = dataset.get_interpolated_caption_tensor(current_batch_size)
            noise = get_noise_tensor(current_batch_size)
            noise2 = get_noise_tensor(current_batch_size)
            text_interpolated = text_interpolated.to(device)
            noise = noise.to(device)
            noise2 = noise2.to(device)

            # generate heatmaps
            heatmap_fake = net_g(noise, text_match)
            heatmap_interpolated = net_g(noise2, text_interpolated)

            # discriminate heatmpap-text pairs
            score_fake = net_d(heatmap_fake, text_match)
            score_interpolated = net_d(heatmap_interpolated, text_interpolated)

            if algorithm == 'gan':
                label = torch.full((current_batch_size,), 1, dtype=torch.float32, device=device)
                loss_g = criterion(score_fake.view(-1), label) + criterion(score_interpolated.view(-1), label)

                # calculate losses and update
                loss_g.backward()
                optimizer_g.step()
            else:
                # 'wgan', 'wgan-gp' and 'wgan-lp'
                # calculate losses and update
                loss_g = -(score_fake + score_interpolated).mean()
                loss_g.backward()
                optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_g, batch_number * (e - start_from_epoch) + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:05d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1, 1, 1))
    net_g.train()
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(fixed_h + 1, fixed_w, sample + 1)
        plot_heatmap(fixed_real_array[sample], skeleton=(None if multi else skeleton))
        plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
        plt.xlabel('(real)')
        plt.xticks([])
        plt.yticks([])
    for sample in range(fixed_size):
        plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton=(None if multi else skeleton))
        plt.title(None)
        plt.xlabel('(fake)')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_noise_samples_' + f'{e + 1:05d}' + '.png')
    plt.close()

    # validate
    net_g.eval()
    net_d.eval()

    # calculate d loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    text_mismatch_val = dataset_val.get_random_caption_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        score_right_val = net_d(heatmap_real_val, text_match_val).detach()
        score_wrong_val = net_d(heatmap_real_val, text_mismatch_val).detach()
        heatmap_fake_val = net_g(noise_val, text_match_val).detach()
        score_fake_val = net_d(heatmap_fake_val, text_match_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_right_val = criterion(score_right_val.view(-1), label_val) * (1 + alpha)
        label_val.fill_(0)
        loss_fake_val = criterion(score_fake_val.view(-1), label_val)
        label_val.fill_(0)
        loss_wrong_val = criterion(score_wrong_val.view(-1), label_val) * alpha
        loss_d_val = loss_right_val + loss_fake_val + loss_wrong_val
    elif algorithm == 'wgan':
        loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val).mean()
    else:
        # 'wgan-gp' and 'wgan-lp'
        epsilon_val = np.random.rand(dataset_val.__len__())
        heatmap_sample_val = torch.empty_like(heatmap_real_val)
        for j in range(dataset_val.__len__()):
            heatmap_sample_val[j] = epsilon_val[j] * heatmap_real_val[j] + (1 - epsilon_val[j]) * heatmap_fake_val[j]
        heatmap_sample_val.requires_grad = True
        text_match_val.requires_grad = True
        score_sample_val = net_d(heatmap_sample_val, text_match_val)
        gradient_h_val, gradient_t_val = grad(score_sample_val, [heatmap_sample_val, text_match_val],
                                              torch.ones_like(score_sample_val), create_graph=True)
        gradient_norm_val = (gradient_h_val.pow(2).sum((1, 2, 3)) + gradient_t_val.pow(2).sum((1, 2, 3))).sqrt()
        if algorithm == 'wgan-gp':
            loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
                (gradient_norm_val - 1).pow(2))).mean()

        else:
            # 'wgan-lp'
            loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
                torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm_val - 1).pow(2))).mean()

    # calculate g loss
    text_interpolated_val = dataset_val.get_interpolated_caption_tensor(dataset_val.__len__()).to(device)
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    noise2_val = get_noise_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        heatmap_fake_val = net_g(noise_val, text_match_val).detach()
        heatmap_interpolated_val = net_g(noise2_val, text_interpolated_val).detach()
        score_fake_val = net_d(heatmap_fake_val, text_match_val).detach()
        score_interpolated_val = net_d(heatmap_interpolated_val, text_interpolated_val).detach()
    if algorithm == 'gan':
        label_val.fill_(1)
        loss_g_val = criterion(score_fake_val.view(-1), label_val) + criterion(score_interpolated_val.view(-1),
                                                                               label_val)
    else:
        # 'wgan', 'wgan-gp' and 'wgan-lp'
        loss_g_val = -(score_fake_val + score_interpolated_val).mean()

    # print and log
    print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' val g loss: ' + str(
        loss_g_val.item()) + ' val d loss: ' + str(loss_d_val.item()))
    writer.add_scalar('loss_val/g', loss_g_val, (e - start_from_epoch))
    writer.add_scalar('loss_val/d', loss_d_val, (e - start_from_epoch))

    net_g.train()
    net_d.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
