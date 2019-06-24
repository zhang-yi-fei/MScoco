from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fastText
from datetime import datetime

# training parameters
learning_rate_g = 0.00005
learning_rate_d = 0.00005
epoch = 36

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

# how often to visualize
plot_cycle = 30

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

# one example to visualize
example_caption = 'A little boy is playing Wii in front of a chair.'
example_vector = torch.tensor(text_model.get_sentence_vector(example_caption), dtype=torch.float32, device=device).view(
    1, sentence_vector_size, 1, 1)
example_noise = torch.randn(1, noise_size, 1, 1, device=device)

net_g = Generator()
net_d = Discriminator()
net_g.to(device)
net_d.to(device)
net_g.apply(weights_init)
net_d.apply(weights_init)
optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(first_momentum, second_momentum))
optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(first_momentum, second_momentum))
criterion = nn.BCELoss()

# visualize example first
with torch.no_grad():
    net_g.eval()
    heatmap = np.array(net_g(example_noise, example_vector).squeeze().tolist()) * 0.5 + 0.5
    plt.ion()
    figure, heatmap_plot, skeleton_plot = plot_heatmap(heatmap, skeleton=skeleton, caption=example_caption)
    net_g.train()

# train
print(datetime.now())
print('training')
net_g.train()
net_d.train()

j = 0
for e in range(epoch):

    # number of batches
    batch = len(image_loader)
    for i, data in enumerate(image_loader, 0):
        j = j + 1

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

        # optimize
        net_d.zero_grad()
        net_g.zero_grad()

        # generate heatmaps
        heatmap_fake = net_g(noise, text_match)
        heatmap_interpolated = net_g(noise2, text_interpolated)

        # discriminate heatmpap-text pairs
        score_right = net_d(heatmap_real, text_match).view(-1)
        score_wrong = net_d(heatmap_real, text_mismatch).view(-1)
        score_fake = net_d(heatmap_fake.detach(), text_match).view(-1)
        score_fake_to_back = net_d(heatmap_fake, text_match).view(-1)
        score_interpolated = net_d(heatmap_interpolated, text_interpolated).view(-1)

        label_real = torch.full((data.size(0),), 1, device=device)
        label_fake = torch.full((data.size(0),), 0, device=device)

        # calculate losses
        criterion(score_right, label_real).backward()
        criterion(score_wrong, label_fake).backward()
        criterion(score_fake, label_fake).backward()
        optimizer_d.step()
        criterion(score_fake_to_back, label_real).backward()
        criterion(score_interpolated, label_real).backward()
        optimizer_g.step()
        # update

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch) + ' score_right: ' + str(score_right.mean().item()) + ' score_wrong: ' + str(
            score_wrong.mean().item()) + ' score_fake: ' + str(
            score_fake.mean().item()) + ' score_interpolated: ' + str(score_interpolated.mean().item()))

        # visualize example
        if j % plot_cycle == 0:
            with torch.no_grad():
                net_g.eval()
                heatmap = np.array(net_g(example_noise, example_vector).squeeze().tolist()) * 0.5 + 0.5
                plot_heatmap_redraw(heatmap, figure, heatmap_plot, skeleton_plot, skeleton=skeleton)
                net_g.train()

    # save models
    torch.save(net_g.state_dict(), generator_path)
    torch.save(net_d.state_dict(), discriminator_path)

print('\nfinished')
print(datetime.now())
