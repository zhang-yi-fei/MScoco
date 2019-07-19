from model import *
from pycocotools.coco import COCO
import torch.optim as optim
# import fastText
from datetime import datetime

workers = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate_g = 0.0002
learning_rate_d = 0.0002
epoch = 100

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1

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
criterion = nn.BCELoss()
scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, 0.8)
scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, 0.8)

batch_score_right = 0
batch_score_wrong = 0
batch_score_fake_before = 0
batch_score_fake_after = 0
batch_score_interpolated = 0
epoch_score_right = []
epoch_score_wrong = []
epoch_score_fake_before = []
epoch_score_fake_after = []
epoch_score_interpolated = []
loss = []

# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()

for e in range(epoch):
    print('learning rate: g ' + str(learning_rate_g) + ' d ' + str(learning_rate_d))
    batch_score_right = 0
    batch_score_wrong = 0
    batch_score_fake_before = 0
    batch_score_fake_after = 0
    batch_score_interpolated = 0

    # number of batches
    batch_number = len(data_loader)

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
        label = torch.full((current_batch_size,), 1, device=device)
        loss_right = criterion(score_right, label)
        loss_right.backward()
        # label.fill_(0)
        # criterion(score_wrong, label).mul(0.5).backward()

        # generate heatmaps
        heatmap_fake = net_g(noise)

        # discriminate heatmpap-text pairs
        # score_wrong = net_d(heatmap_real, text_mismatch).view(-1)
        score_fake = net_d(heatmap_fake.detach()).view(-1)

        # calculate losses and update
        # label.fill_(0)
        # criterion(score_wrong, label).mul(0.5).backward()
        label.fill_(0)
        loss_fake = criterion(score_fake, label)
        loss_fake.backward()
        optimizer_d.step()

        mean_score_right = score_right.mean().item()
        # mean_score_wrong = score_wrong.mean().item()
        mean_score_fake_before = score_fake.mean().item()

        loss.append(loss_right.detach() + loss_fake.detach())

        # second, optimize generator
        net_g.zero_grad()

        # generate heatmaps
        # heatmap_interpolated = net_g(noise2, text_interpolated)

        # discriminate heatmpap-text pairs
        score_fake2 = net_d(heatmap_fake).view(-1)
        # score_interpolated = net_d(heatmap_interpolated, text_interpolated).view(-1)

        # calculate losses and update
        label.fill_(1)
        loss_fake2 = criterion(score_fake2, label)
        loss_fake2.backward()
        # label.fill_(1)
        # criterion(score_interpolated, label).backward()
        optimizer_g.step()

        mean_score_fake_after = score_fake2.mean().item()
        # mean_score_interpolated = score_interpolated.mean().item()

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(batch_number))
        print('score_right: ' + str(mean_score_right) + ' score_fake(before): ' + str(
            mean_score_fake_before) + ' score_fake(after): ' + str(mean_score_fake_after) + ' loss: ' + str(
            loss[-1].item()))

        # record scores
        batch_score_right = batch_score_right + mean_score_right
        # batch_score_wrong = batch_score_wrong + mean_score_wrong
        batch_score_fake_before = batch_score_fake_before + mean_score_fake_before
        batch_score_fake_after = batch_score_fake_after + mean_score_fake_after
        # batch_score_interpolated = batch_score_interpolated + mean_score_interpolated

    # learning rate scheduling
    scheduler_g.step()
    scheduler_d.step()

    # record scores (epoch average)
    epoch_score_right.append(batch_score_right / batch_number)
    # epoch_score_wrong.append(batch_score_wrong / batch_number)
    epoch_score_fake_before.append(batch_score_fake_before / batch_number)
    epoch_score_fake_after.append(batch_score_fake_after / batch_number)
    # epoch_score_interpolated.append(batch_score_interpolated / batch_number)

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + str(e + 1))
    torch.save(net_d.state_dict(), discriminator_path + '_' + str(e + 1))

    # save traces of scores
    plt.plot(epoch_score_right)
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('matching')
    plt.savefig('Figure_1.png')
    plt.close()

    plt.plot(epoch_score_wrong)
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('mismatching')
    plt.savefig('Figure_2.png')
    plt.close()

    plt.plot(epoch_score_fake_before)
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('generated (before discriminator updated)')
    plt.savefig('Figure_3.png')
    plt.close()

    plt.plot(epoch_score_fake_after)
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('generated (after discriminator updated)')
    plt.savefig('Figure_4.png')
    plt.close()

    plt.plot(epoch_score_interpolated)
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('generated (interpolated)')
    plt.savefig('Figure_5.png')
    plt.close()

    # save trace of loss
    plt.plot(loss)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.savefig('Figure_6.png')
    plt.close()

    # save scores
    torch.save(epoch_score_right, 'epoch_score_right')
    torch.save(epoch_score_wrong, 'epoch_score_wrong')
    torch.save(epoch_score_fake_before, 'epoch_score_fake_before')
    torch.save(epoch_score_fake_after, 'epoch_score_fake_after')
    torch.save(epoch_score_interpolated, 'epoch_score_interpolated')

    # save loss
    torch.save(loss, 'loss')

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
