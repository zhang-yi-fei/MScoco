from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

workers = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 1024
learning_rate_xy = 0.01
learning_rate_v = 0.001
epoch = 500

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
visibility_real_val = data_val.get('visibility').to(device)

net_xy = Regression_xy().to(device)
net_v = Regression_v().to(device)
net_xy.apply(weights_init)
net_v.apply(weights_init)

optimizer_xy = optim.Adam(net_xy.parameters(), lr=learning_rate_xy, betas=(beta_1, beta_2))
optimizer_v = optim.Adam(net_v.parameters(), lr=learning_rate_v, betas=(beta_1, beta_2))
criterion_xy = nn.MSELoss()
criterion_v = nn.BCEWithLogitsLoss()

# fixed training data (from validation set), noise and sentence vectors to see the progression
fixed_w = 5
fixed_train = dataset_val.get_random_coordinates_visibility_with_caption(fixed_w)
fixed_real_xy = fixed_train.get('coordinates').to(device)
fixed_real_xy_array = np.array(fixed_real_xy.tolist())
fixed_real_v = fixed_train.get('visibility').to(device)
fixed_real_v_array = np.array(fixed_real_v.tolist())
fixed_caption = fixed_train.get('caption')
fixed_text = torch.tensor([get_caption_vector(text_model, caption) for caption in fixed_caption], dtype=torch.float32,
                          device=device)

# save models before training
torch.save(net_xy.state_dict(), regression_xy_path + '_' + f'{0:05d}')
torch.save(net_v.state_dict(), regression_v_path + '_' + f'{0:05d}')

# plot and save generated samples (before training begins)
net_xy.eval()
net_v.eval()
with torch.no_grad():
    fixed_fake_xy = net_xy(fixed_text)
    fixed_fake_v = net_v(fixed_text)
net_xy.train()
net_v.train()
fixed_fake_xy = np.array(fixed_fake_xy.tolist())
fixed_fake_v = np.array(fixed_fake_v.tolist())
f = plt.figure(figsize=(19.2, 12))
for sample in range(fixed_w):
    plt.subplot(2, fixed_w, sample + 1)
    plot_pose(fixed_real_xy_array[sample][0:total_keypoints],
              fixed_real_xy_array[sample][total_keypoints:2 * total_keypoints], fixed_real_v_array[sample],
              skeleton=skeleton)
    plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
    plt.xlabel('(real)')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, fixed_w, fixed_w + sample + 1)
    x, y, v = result_to_coordinates_visibility(fixed_fake_xy[sample], fixed_fake_v[sample])
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
net_xy.train()
net_v.train()
writer = SummaryWriter(comment='_regression')

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: xy ' + str(optimizer_xy.param_groups[0].get('lr')) + ' v ' + str(
        optimizer_v.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        net_xy.zero_grad()
        net_v.zero_grad()

        # get coordinates, visibility, sentence vectors
        coordinates_real = batch.get('coordinates')
        visibility_real = batch.get('visibility')
        text_match = batch.get('vector')
        current_batch_size = len(coordinates_real)

        coordinates_real = coordinates_real.to(device)
        visibility_real = visibility_real.to(device)
        text_match = text_match.to(device)

        # regression
        coordinates_fake = net_xy(text_match)
        visibility_fake = net_v(text_match)

        # calculate losses adn update
        loss_xy = criterion_xy(coordinates_fake, coordinates_real)
        loss_v = criterion_v(visibility_fake, visibility_real)
        loss_xy.backward()
        loss_v.backward()
        optimizer_xy.step()
        optimizer_v.step()

        # log
        writer.add_scalar('loss/xy', loss_xy, batch_number * e + i)
        writer.add_scalar('loss/v', loss_v, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' xy loss: ' + str(loss_xy.item()) + ' v loss: ' + str(loss_v.item()))

    # save models
    torch.save(net_xy.state_dict(), regression_xy_path + '_' + f'{e + 1:05d}')
    torch.save(net_v.state_dict(), regression_v_path + '_' + f'{e + 1:05d}')

    # plot and save generated samples
    net_xy.eval()
    net_v.eval()
    with torch.no_grad():
        fixed_fake_xy = net_xy(fixed_text)
        fixed_fake_v = net_v(fixed_text)
    net_xy.train()
    net_v.train()
    fixed_fake_xy = np.array(fixed_fake_xy.tolist())
    fixed_fake_v = np.array(fixed_fake_v.tolist())
    f = plt.figure(figsize=(19.2, 12))
    for sample in range(fixed_w):
        plt.subplot(2, fixed_w, sample + 1)
        plot_pose(fixed_real_xy_array[sample][0:total_keypoints],
                  fixed_real_xy_array[sample][total_keypoints:2 * total_keypoints], fixed_real_v_array[sample],
                  skeleton=skeleton)
        plt.title(fixed_caption[sample][0:30] + '\n' + fixed_caption[sample][30:])
        plt.xlabel('(real)')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, fixed_w, fixed_w + sample + 1)
        x, y, v = result_to_coordinates_visibility(fixed_fake_xy[sample], fixed_fake_v[sample])
        plot_pose(x, y, v, skeleton=skeleton)
        plt.title(None)
        plt.xlabel('(fake)')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('figures/fixed_samples_' + f'{e + 1:05d}' + '.png')
    plt.close()

    # validate
    net_xy.eval()
    net_v.eval()

    # calculate losses
    with torch.no_grad():
        coordinates_fake_val = net_xy(text_match_val).detach()
        visibility_fake_val = net_v(text_match_val).detach()
        loss_xy_val = criterion_xy(coordinates_fake_val, coordinates_real_val)
        loss_v_val = criterion_v(visibility_fake_val, visibility_real_val)

    # print and log
    print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val xy loss: ' + str(
        loss_xy_val.item()) + ' val v loss: ' + str(loss_v_val.item()))
    writer.add_scalar('loss_val/xy', loss_xy_val, e)
    writer.add_scalar('loss_val/v', loss_v_val, e)

    net_xy.train()
    net_v.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
