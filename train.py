from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fastText
from datetime import datetime

# training parameters
batch_size = 64
learning_rate_g = 0.0002
learning_rate_d = 0.0002
epoch = 200

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fastText.load_model(text_model_path)

# get single-person image ids
image_ids = get_one_person_image_ids(coco_keypoint)

# data loader, containing image ids
image_loader = torch.utils.data.DataLoader(image_ids, batch_size=batch_size, shuffle=True)

net_g = Generator()
net_d = Discriminator()
net_g.to(device)
net_d.to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(first_momentum, second_momentum))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(first_momentum, second_momentum))
criterion = nn.BCELoss()

batch_score_right = 0
batch_score_wrong = 0
batch_score_fake = 0
batch_score_interpolated = 0
epoch_score_right = []
epoch_score_wrong = []
epoch_score_fake = []
epoch_score_interpolated = []

# train
print(datetime.now())
print('training')
net_g.train()
net_d.train()

for e in range(epoch):
    batch_score_right = 0
    batch_score_wrong = 0
    batch_score_fake = 0
    batch_score_interpolated = 0

    # number of batches
    batch = len(image_loader)
    for i, data in enumerate(image_loader, 0):
        # get heatmaps, sentence vectors and noises
        heatmap_real, text_match = get_random_view_and_caption_tensor(coco_keypoint, coco_caption, text_model, data)
        text_mismatch = get_random_caption_tensor(coco_caption, text_model, data)
        text_interpolated = get_interpolated_caption_tensor(coco_caption, text_model, data)
        noise = get_noise_tensor(data)
        noise2 = get_noise_tensor(data)

        heatmap_real = heatmap_real.to(device)
        text_match = text_match.to(device)
        text_mismatch = text_mismatch.to(device)
        text_interpolated = text_interpolated.to(device)
        noise = noise.to(device)
        noise2 = noise2.to(device)

        # first, optimize discriminator
        net_d.zero_grad()

        # generate heatmaps
        heatmap_fake = net_g(noise, text_match)

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real, text_match).view(-1)
        score_wrong = net_d(heatmap_real, text_mismatch).view(-1)
        score_fake = net_d(heatmap_fake.detach(), text_match).view(-1)

        label_real = torch.full((data.size(0),), 1, device=device)
        label_fake = torch.full((data.size(0),), 0, device=device)

        # calculate losses and update
        criterion(score_right, label_real).backward()
        criterion(score_wrong, label_fake).mul(0.5).backward()
        criterion(score_fake, label_fake).mul(0.5).backward()
        optimizer_d.step()

        # second, optimize generator
        net_g.zero_grad()

        # generate heatmaps
        heatmap_interpolated = net_g(noise2, text_interpolated)

        # discriminate heatmpap-text pairs
        score_fake = net_d(heatmap_fake, text_match).view(-1)
        score_interpolated = net_d(heatmap_interpolated, text_interpolated).view(-1)

        # calculate losses and update
        criterion(score_fake, label_real).backward()
        criterion(score_interpolated, label_real).backward()
        optimizer_g.step()

        # print progress
        mean_score_right = score_right.mean().item()
        mean_score_wrong = score_wrong.mean().item()
        mean_score_fake = score_fake.mean().item()
        mean_score_interpolated = score_interpolated.mean().item()
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch) + ' score_right: ' + str(mean_score_right) + ' score_wrong: ' + str(
            mean_score_wrong) + ' score_fake: ' + str(mean_score_fake) + ' score_interpolated: ' + str(
            mean_score_interpolated))

        # record scores
        batch_score_right = batch_score_right + mean_score_right
        batch_score_wrong = batch_score_wrong + mean_score_wrong
        batch_score_fake = batch_score_fake + mean_score_fake
        batch_score_interpolated = batch_score_interpolated + mean_score_interpolated

    # record scores
    epoch_score_right.append(batch_score_right / batch)
    epoch_score_wrong.append(batch_score_wrong / batch)
    epoch_score_fake.append(batch_score_fake / batch)
    epoch_score_interpolated.append(batch_score_interpolated / batch)

    # save models
    torch.save(net_g.state_dict(), generator_path + '_' + str(e + 1))
    torch.save(net_d.state_dict(), discriminator_path + '_' + str(e + 1))

print('\nfinished')
print(datetime.now())

# save models
torch.save(net_g.state_dict(), generator_path)
torch.save(net_d.state_dict(), discriminator_path)

# traces of scores
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

plt.plot(epoch_score_fake)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('generated')
plt.savefig('Figure_3.png')
plt.close()

plt.plot(epoch_score_interpolated)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('generated (interpolated)')
plt.savefig('Figure_4.png')
plt.close()

torch.save(epoch_score_right, 'epoch_score_right')
torch.save(epoch_score_wrong, 'epoch_score_wrong')
torch.save(epoch_score_fake, 'epoch_score_fake')
torch.save(epoch_score_interpolated, 'epoch_score_interpolated')
