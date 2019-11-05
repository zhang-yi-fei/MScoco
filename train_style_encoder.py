from model import *
from pycocotools.coco import COCO
import torch.optim as optim
import fasttext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

workers = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# training parameters
batch_size = 128
learning_rate = 0.001
epoch = 5000
gan_epoch = 2000

# ADAM solver
beta_1 = 0.9
beta_2 = 0.999

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)
coco_caption_val = COCO(caption_path_val)
coco_keypoint_val = COCO(keypoint_path_val)

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get the dataset (single person, with captions)
dataset = HeatmapDataset(coco_keypoint, coco_caption, single_person=True, text_model=text_model)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, single_person=True, text_model=text_model)

# data loader, containing heatmap information
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# data to validate
data_val = enumerate(torch.utils.data.DataLoader(dataset_val, batch_size=dataset_val.__len__())).__next__()[1]
text_val = data_val.get('vector').to(device)

# generator from previous training
net_g = Generator2()
net_g.load_state_dict(torch.load(generator_path + '_' + f'{gan_epoch:05d}'))
net_g.to(device)
net_g.eval()

# style encoder
net_s = Encoder(noise_size, False).to(device)
net_s.apply(weights_init)
optimizer_s = optim.Adam(net_s.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
criterion = nn.MSELoss()

# train
start = datetime.now()
print(start)
print('training')
net_s.train()
writer = SummaryWriter(comment='_style')

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: ' + str(optimizer_s.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # optimize style encoder
        net_s.zero_grad()

        # get sentence vectors and noises
        text = batch.get('vector')
        current_batch_size = len(text)
        noise = get_noise_tensor(current_batch_size)

        text = text.to(device)
        noise = noise.to(device)

        # generate heatmaps
        with torch.no_grad():
            heatmap_fake = net_g(noise, text).detach()

        # predict style
        style = net_s(heatmap_fake)

        # calculate losses and update
        loss = criterion(style, noise)
        loss.backward()
        optimizer_s.step()

        # log
        writer.add_scalar('loss/s', loss, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' loss: ' + str(loss.item()))

    # save model
    torch.save(net_s.state_dict(), style_encoder_path + '_' + f'{e + 1:05d}')

    # validate
    net_s.eval()
    noise_val = get_noise_tensor(dataset_val.__len__()).to(device)
    with torch.no_grad():
        heatmap_val = net_g(noise_val, text_val).detach()
        style_val = net_s(heatmap_val).detach()
        loss_val = criterion(style_val, noise_val)
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val loss: ' + str(loss_val.item()))
        writer.add_scalar('loss_val/s', loss_val, e)
    net_s.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
