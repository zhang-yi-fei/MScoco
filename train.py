from model import *
from pycocotools.coco import COCO
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

workers = 8
device = torch.device('cpu')

# training parameters
batch_size = 128
learning_rate_g = 0.0001
learning_rate_d = 0.0001
epoch = 200

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
coco_caption_val = COCO(caption_path_val)
coco_keypoint_val = COCO(keypoint_path_val)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0].get('skeleton')) - 1

# get the
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val)

# data loader, containing heatmap information

# data to validate
data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
heatmap_real_val = data_val.get('heatmap').to(device)

net_g = Generator().to(device)
net_d = Discriminator().to(device)


# train
start = datetime.now()
print(start)
print('training')
net_g.train()
net_d.train()
iteration = 1
writer = SummaryWriter(comment='_pose')
loss_g = torch.tensor(0)
loss_d = torch.tensor(0)


for e in range(epoch):

    net_g.load_state_dict(torch.load(generator_path + '_' + f'{e+1:05d}'), False)
    net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{e+1:05d}'), False)

    # validate
    net_g.eval()
    net_d.eval()

    # calculate d loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        score_right_val = net_d(heatmap_real_val).detach()
        heatmap_fake_val = net_g(noise_val).detach()
        score_fake_val = net_d(heatmap_fake_val).detach()
    epsilon_val = np.random.rand(dataset_val.__len__())
    heatmap_sample_val = torch.empty_like(heatmap_real_val)
    for j in range(dataset_val.__len__()):
        heatmap_sample_val[j] = epsilon_val[j] * heatmap_real_val[j] + (1 - epsilon_val[j]) * heatmap_fake_val[j]
    heatmap_sample_val.requires_grad = True
    score_sample_val = net_d(heatmap_sample_val)
    gradient_val, = grad(score_sample_val, heatmap_sample_val, torch.ones_like(score_sample_val), create_graph=True)
    gradient_norm_val = gradient_val.pow(2).sum((1, 2, 3)).sqrt()
    loss_d_val = (score_fake_val - score_right_val + lamb * ((gradient_norm_val - 1).pow(2))).mean()

    # calculate g loss
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        heatmap_fake_val = net_g(noise_val).detach()
        score_fake_val = net_d(heatmap_fake_val).detach()
    loss_g_val = -score_fake_val.mean()

    # print and log
    print(
        'epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val g loss: ' + str(loss_g_val.item()) + ' val d loss: ' + str(
            loss_d_val.item()))
    writer.add_scalar('loss_val/g', loss_g_val, e)
    writer.add_scalar('loss_val/d', loss_d_val, e)

    net_g.train()
    net_d.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
