from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

workers = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate = 0.01
epoch = 500

# weight of visibility loss part
w = 1000

# ADAM solver
beta_1 = 0.9
beta_2 = 0.999

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

net = Regression().to(device)
net.apply(weights_init)

optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
criterion_xy = nn.MSELoss()
criterion_v = nn.BCEWithLogitsLoss()

# fixed training data (from validation set), noise and sentence vectors to see the progression
fixed_w = 5
fixed_train = dataset_val.get_random_coordinates_with_caption(fixed_w)
fixed_real = np.array(fixed_train.get('coordinates').tolist())
fixed_caption = fixed_train.get('caption')
fixed_text = torch.tensor([get_caption_vector(text_model, caption) for caption in fixed_caption], dtype=torch.float32,
                          device=device)

# save models before training
torch.save(net.state_dict(), regression_path + '_' + f'{0:05d}')

# plot and save generated samples (before training begins)
net.eval()
with torch.no_grad():
    fixed_fake = net(fixed_text)
net.train()
fixed_fake = np.array(fixed_fake.tolist())
f = plt.figure(figsize=(19.2, 12))
for sample in range(fixed_w):
    plt.subplot(2, fixed_w, sample + 1)
    plot_pose(fixed_real[sample][0:total_keypoints], fixed_real[sample][total_keypoints:2 * total_keypoints],
              fixed_real[sample][2 * total_keypoints:3 * total_keypoints], skeleton=skeleton)
    plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
    plt.xlabel('(real)')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, fixed_w, fixed_w + sample + 1)
    x, y, v = result_to_coordinates(fixed_fake[sample])
    plot_pose(x, y, v, skeleton=skeleton)
    plt.title(None)
    plt.xlabel('(fake)')
    plt.xticks([])
    plt.yticks([])
plt.savefig('figures/fixed_samples_start.png')
plt.close()

# train
start = datetime.now()
print(start)
print('training')
net.train()
writer = SummaryWriter(comment='_regression')

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: ' + str(optimizer.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        net.zero_grad()

        # get coordinates, sentence vectors
        coordinates_real = batch.get('coordinates')
        text_match = batch.get('vector')
        current_batch_size = len(coordinates_real)

        coordinates_real = coordinates_real.to(device)
        text_match = text_match.to(device)

        # regression
        coordinates_fake = net(text_match)

        # calculate losses adn update
        loss_xy = criterion_xy(coordinates_fake[:, 0:2 * total_keypoints], coordinates_real[:, 0:2 * total_keypoints])
        loss_v = criterion_v(coordinates_fake[:, 2 * total_keypoints:3 * total_keypoints],
                             coordinates_real[:, 2 * total_keypoints:3 * total_keypoints])
        loss = loss_xy + w * loss_v
        loss.backward()
        optimizer.step()

        # log
        writer.add_scalar('loss/xy', loss_xy, batch_number * e + i)
        writer.add_scalar('loss/v', loss_v, batch_number * e + i)
        writer.add_scalar('loss/xy+v', loss, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' xy loss: ' + str(loss_xy.item()) + ' v loss: ' + str(
            loss_v.item()) + ' overall loss: ' + str(loss.item()))

    # save models
    torch.save(net.state_dict(), regression_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples
    net.eval()
    with torch.no_grad():
        fixed_fake = net(fixed_text)
    net.train()
    fixed_fake = np.array(fixed_fake.tolist())
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(2, fixed_w, sample + 1)
        plot_pose(fixed_real[sample][0:total_keypoints], fixed_real[sample][total_keypoints:2 * total_keypoints],
                  fixed_real[sample][2 * total_keypoints:3 * total_keypoints], skeleton=skeleton)
        plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
        plt.xlabel('(real)')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, fixed_w, fixed_w + sample + 1)
        x, y, v = result_to_coordinates(fixed_fake[sample])
        plot_pose(x, y, v, skeleton=skeleton)
        plt.title(None)
        plt.xlabel('(fake)')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_samples_' + f'{e + 1:05d}' + '.png')
    plt.close()

    # validate
    net.eval()

    # calculate losses
    with torch.no_grad():
        coordinates_fake_val = net(text_match_val).detach()
        loss_xy_val = criterion_xy(coordinates_fake_val[:, 0:2 * total_keypoints],
                                   coordinates_real_val[:, 0:2 * total_keypoints])
        loss_v_val = criterion_v(coordinates_fake_val[:, 2 * total_keypoints:3 * total_keypoints],
                                 coordinates_real_val[:, 2 * total_keypoints:3 * total_keypoints])
        loss_val = loss_xy_val + w * loss_v_val

    # print and log
    print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val xy loss: ' + str(
        loss_xy_val.item()) + ' val v loss: ' + str(loss_v_val.item()) + ' val overall loss: ' + str(loss_val.item()))
    writer.add_scalar('loss_val/xy', loss_xy_val, e)
    writer.add_scalar('loss_val/v', loss_v_val, e)
    writer.add_scalar('loss_val/xy+v', loss_val, e)

    net.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
