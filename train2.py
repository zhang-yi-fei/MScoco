from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fastText
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate_g = 0.0001
learning_rate_d = 0.0001
start_from_epoch = 0
end_in_epoch = 2000

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

# load text encoding model
text_model = fastText.load_model(text_model_path)

# get the dataset (single person, with captions)
dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=True, text_model=text_model)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

net_g = Generator2().to(device)
net_d = Discriminator2().to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)

# load first step (without captions) trained weights
# net_g.load_state_dict(torch.load(generator_path + '_' + f'{start_from_epoch:05d}'), False)
# net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{start_from_epoch:05d}'), False)
# net_g.first2.weight.data[0:noise_size] = net_g.first.weight.data
# net_d.second2.weight.data[:, 0:convolution_channel_d[-1], :, :] = net_d.second.weight.data
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(beta_1, beta_2))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(beta_1, beta_2))

# fixed training data, noise and sentence vectors to see the progression
fixed_h = 6
fixed_w = 4
fixed_size = fixed_h * fixed_w
fixed_train = dataset.get_random_heatmap_with_caption(fixed_w)
fixed_real = fixed_train.get('heatmap').to(device)
fixed_real_array = np.array(fixed_real.tolist()) * 0.5 + 0.5
fixed_caption = fixed_train.get('caption')
fixed_noise = get_noise_tensor(fixed_h).to(device)
fixed_text = torch.tensor([text_model.get_sentence_vector(caption.replace('\n', '')) for caption in fixed_caption],
                          dtype=torch.float32, device=device).unsqueeze(-1).unsqueeze(-1)

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
writer.add_graph(JoinGAN2().to(device),
                 (fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1, 1, 1)))

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
        # text_mismatch = dataset.get_random_caption_tensor(current_batch_size)
        noise = get_noise_tensor(current_batch_size)

        heatmap_real = heatmap_real.to(device)
        text_match = text_match.to(device)
        # text_mismatch = text_mismatch.to(device)
        noise = noise.to(device)

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real, text_match)
        # score_wrong = net_d(heatmap_real, text_mismatch)

        # generate heatmaps
        heatmap_fake = net_g(noise, text_match).detach()

        # discriminate heatmpap-text pairs
        score_fake = net_d(heatmap_fake, text_match)

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
        loss_d = (score_fake - score_right + lamb * ((gradient_norm - 1).pow(2))).mean()
        loss_d.backward()
        optimizer_d.step()

        # log
        writer.add_scalar('loss/d', loss_d, batch_number * e + i)
        writer.add_histogram('score/real', score_right, batch_number * e + i)
        # writer.add_histogram('score/wrong', score_wrong, batch_number * e + i)
        writer.add_histogram('score/fake', score_fake, batch_number * e + i)
        writer.add_histogram('gradient_norm', gradient_norm, batch_number * e + i)

        # second, optimize generator
        if iteration == k:
            net_g.zero_grad()
            iteration = 0

            # get sentence vectors and noises
            # text_interpolated = dataset.get_interpolated_caption_tensor(current_batch_size)
            noise = get_noise_tensor(current_batch_size)
            # noise2 = get_noise_tensor(current_batch_size)
            # text_interpolated = text_interpolated.to(device)
            noise = noise.to(device)
            # noise2 = noise2.to(device)

            # generate heatmaps
            heatmap_fake = net_g(noise, text_match)
            # heatmap_interpolated = net_g(noise2, text_interpolated)

            # discriminate heatmpap-text pairs
            score_fake = net_d(heatmap_fake, text_match)
            # score_interpolated = net_d(heatmap_interpolated, text_interpolated)

            # discriminate losses and update
            loss_g = -score_fake.mean()
            loss_g.backward()
            optimizer_g.step()

            # log
            writer.add_scalar('loss/g', loss_g, batch_number * e + i)
            writer.add_histogram('score/fake_2', score_fake, batch_number * e + i)
            # writer.add_histogram('score/interpolated', score_interpolated, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(end_in_epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' g loss: ' + str(loss_g.item()) + ' d loss: ' + str(loss_d.item()))

        iteration = iteration + 1

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + f'{e + 1:05d}')
    torch.save(net_d.state_dict(), discriminator_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples from fixed noise
    net_g.eval()
    net_d.eval()
    with torch.no_grad():
        fixed_fake = net_g(fixed_noise.repeat_interleave(fixed_w, dim=0), fixed_text.repeat(fixed_h, 1, 1, 1))
        fixed_score_fake = net_d(fixed_fake, fixed_text.repeat(fixed_h, 1, 1, 1))
        fixed_score_real = net_d(fixed_real, fixed_text)
    net_g.train()
    net_d.train()
    fixed_fake = np.array(fixed_fake.tolist()) * 0.5 + 0.5
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(fixed_h + 1, fixed_w, sample + 1)
        plot_heatmap(fixed_real_array[sample], skeleton)
        plt.title(fixed_caption[sample][0:20] + '\n' + fixed_caption[sample][20:])
        plt.xlabel('(real) score = ' + str(fixed_score_real[sample].item()))
        plt.xticks([])
        plt.yticks([])
    for sample in range(fixed_size):
        plt.subplot(fixed_h + 1, fixed_w, fixed_w + sample + 1)
        plot_heatmap(fixed_fake[sample], skeleton)
        plt.title(None)
        plt.xlabel('(fake) score = ' + str(fixed_score_fake[sample].item()))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_noise_samples_' + f'{e + 1:05d}' + '.png')

    # log
    writer.add_images('heatmap', np.amax(fixed_fake, 1, keepdims=True), e, dataformats='NCHW')
    writer.add_figure('heatmaps', f, e)

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
