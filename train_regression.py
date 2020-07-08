from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate_g = 0.001
learning_rate_d = 0.001
epoch = 2000

# visibility entropy loss weight
w = 10

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
dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=True, text_model=text_model, for_regression=True)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, single_person=True, text_model=text_model,
                             for_regression=True)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# data to validate
data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
text_match_val = data_val.get('vector').to(device)
coordinates_real_val = data_val.get('coordinates').to(device)

net_g = Generator_R().to(device)
net_d = Discriminator_R().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))
criterion = nn.BCELoss()

# fixed training data (from validation set), noise and sentence vectors to see the progression
fixed_h = 6
fixed_w = 5
fixed_size = fixed_h * fixed_w
fixed_train = dataset_val.get_random_coordinates_with_caption(fixed_w)
fixed_real = fixed_train.get('coordinates').to(device)
fixed_real_array = np.array(fixed_real.tolist())
fixed_caption = fixed_train.get('caption')
fixed_noise = get_noise_tensor(fixed_h).to(device).squeeze_()
fixed_text = torch.tensor([get_caption_vector(text_model, caption) for caption in fixed_caption], dtype=torch.float32,
                          device=device)

# save models before training
torch.save(net_g.state_dict(), generator_path + '_' + f'{0:05d}')
torch.save(net_d.state_dict(), discriminator_path + '_' + f'{0:05d}')

# plot and save generated samples from fixed noise (before training begins)
net_g.eval()
with torch.no_grad():
    fixed_fake = net_g(fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1))
net_g.train()
fixed_fake = np.array(fixed_fake.tolist())
f = plt.figure(figsize=(19.2, 12))
for sample in range(fixed_w):
    plt.subplot(fixed_h + 1, fixed_w, sample + 1)
    x, y, v = result_to_coordinates(fixed_real_array[sample])
    plot_pose(x, y, v, skeleton)
    plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
    plt.xlabel('(real)')
    plt.xticks([])
    plt.yticks([])
for sample in range(fixed_size):
    plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
    x, y, v = result_to_coordinates(fixed_fake[sample])
    plot_pose(x, y, v, skeleton)
    plt.title(None)
    plt.xlabel('(fake)')
    plt.xticks([])
    plt.yticks([])
plt.savefig('figures/fixed_noise_samples_' + f'{0:05d}' + '.png')
plt.close()

# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()
iteration = 1
writer = SummaryWriter(comment='_regression')
loss_g = torch.tensor(0)
loss_d = torch.tensor(0)
loss_v = torch.tensor(0)
loss_gv = torch.tensor(0)

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: g ' + str(optimizer_g.param_groups[0].get('lr')) + ' d ' + str(
        optimizer_d.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # first, optimize discriminator
        net_d.zero_grad()

        # get coordinatess, sentence vectors and noises
        coordinates_real = batch.get('coordinates')
        text_match = batch.get('vector')
        current_batch_size = len(coordinates_real)
        text_mismatch = dataset.get_random_caption_tensor(current_batch_size)
        noise = get_noise_tensor(current_batch_size).squeeze_()

        coordinates_real = coordinates_real.to(device)
        text_match = text_match.to(device)
        text_mismatch = text_mismatch.to(device)
        noise = noise.to(device)

        # discriminate coordinates-text pairs
        score_right = net_d(coordinates_real, text_match)
        score_wrong = net_d(coordinates_real, text_mismatch)

        # generate coordinates
        coordinates_fake = net_g(noise, text_match).detach()

        # discriminate heatmpap-text pairs
        score_fake = net_d(coordinates_fake, text_match)

        # random sample
        epsilon = np.random.rand(current_batch_size)
        coordinates_sample = torch.empty_like(coordinates_real)
        for j in range(current_batch_size):
            coordinates_sample[j] = epsilon[j] * coordinates_real[j] + (1 - epsilon[j]) * coordinates_fake[j]
        coordinates_sample.requires_grad = True
        text_match.requires_grad = True

        # calculate gradient penalty
        score_sample = net_d(coordinates_sample, text_match)
        gradient_h, gradient_t = grad(score_sample, [coordinates_sample, text_match], torch.ones_like(score_sample),
                                      create_graph=True)
        gradient_norm = (gradient_h.pow(2).sum(1) + gradient_t.pow(2).sum(1)).sqrt()

        # calculate losses and update
        loss_d = (score_fake + alpha * score_wrong - (1 + alpha) * score_right + lamb * (
            torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm - 1).pow(2))).mean()

        loss_d.backward()
        optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * e + i)

        # second, optimize generator
        if iteration == k:
            net_g.zero_grad()
            iteration = 0

            # get noise
            noise = get_noise_tensor(current_batch_size).squeeze_()
            noise = noise.to(device)

            # generate coordinates
            coordinates_fake = net_g(noise, text_match)

            # discriminate coordinates-text pairs
            score_fake = net_d(coordinates_fake, text_match)

            # calculate losses and update
            loss_g = -score_fake.mean()
            loss_v = criterion(coordinates_fake[:, total_keypoints * 2:total_keypoints * 3],
                               coordinates_real[:, total_keypoints * 2:total_keypoints * 3])
            loss_gv = loss_g + w * loss_v
            loss_gv.backward()
            optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_g, batch_number * e + i)
            writer.add_scalar('loss/v', loss_v, batch_number * e + i)
            writer.add_scalar('loss/g+v', loss_gv, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' v loss: ' + str(loss_v.item()) + ' g+v loss: ' + str(
            loss_gv.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:05d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1))
    net_g.train()
    fixed_fake = np.array(fixed_fake.tolist())
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(fixed_h + 1, fixed_w, sample + 1)
        x, y, v = result_to_coordinates(fixed_real_array[sample])
        plot_pose(x, y, v, skeleton)
        plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
        plt.xlabel('(real)')
        plt.xticks([])
        plt.yticks([])
    for sample in range(fixed_size):
        plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
        x, y, v = result_to_coordinates(fixed_fake[sample])
        plot_pose(x, y, v, skeleton)
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
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device).squeeze_()
    text_mismatch_val = dataset_val.get_random_caption_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        score_right_val = net_d(coordinates_real_val, text_match_val).detach()
        score_wrong_val = net_d(coordinates_real_val, text_mismatch_val).detach()
        coordinates_fake_val = net_g(noise_val, text_match_val).detach()
        score_fake_val = net_d(coordinates_fake_val, text_match_val).detach()

    epsilon_val = np.random.rand(dataset_val.__len__())
    coordinates_sample_val = torch.empty_like(coordinates_real_val)
    for j in range(dataset_val.__len__()):
        coordinates_sample_val[j] = epsilon_val[j] * coordinates_real_val[j] + (1 - epsilon_val[j]) * \
                                    coordinates_fake_val[j]
    coordinates_sample_val.requires_grad = True
    text_match_val.requires_grad = True
    score_sample_val = net_d(coordinates_sample_val, text_match_val)
    gradient_h_val, gradient_t_val = grad(score_sample_val, [coordinates_sample_val, text_match_val],
                                          torch.ones_like(score_sample_val), create_graph=True)
    gradient_norm_val = (gradient_h_val.pow(2).sum(1) + gradient_t_val.pow(2).sum(1)).sqrt()
    loss_d_val = (score_fake_val + alpha * score_wrong_val - (1 + alpha) * score_right_val + lamb * (
        torch.max(torch.tensor(0, dtype=torch.float32, device=device), gradient_norm_val - 1).pow(2))).mean()

    # calculate g loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device).squeeze_()
    with torch.no_grad():
        coordinates_fake_val = net_g(noise_val, text_match_val).detach()
        score_fake_val = net_d(coordinates_fake_val, text_match_val).detach()
    loss_g_val = -score_fake_val.mean()
    loss_v_val = criterion(coordinates_fake_val[:, total_keypoints * 2:total_keypoints * 3],
                           coordinates_real_val[:, total_keypoints * 2:total_keypoints * 3])
    loss_gv_val = loss_g_val + w * loss_v_val

    # print and log
    print(
        'epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val g loss: ' + str(loss_g_val.item()) + ' val v loss: ' + str(
            loss_v_val.item()) + ' val g+v loss: ' + str(loss_gv_val.item()) + ' val d loss: ' + str(loss_d_val.item()))
    writer.add_scalar('loss_val/g', loss_g_val, e)
    writer.add_scalar('loss_val/d', loss_d_val, e)
    writer.add_scalar('loss_val/v', loss_v_val, e)
    writer.add_scalar('loss_val/g+v', loss_gv_val, e)

    net_g.train()
    net_d.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
