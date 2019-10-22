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
epoch = 2000

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
heatmap_val = data_val.get('heatmap').to(device)

# style encoder
net_c = ContentEncoder().to(device)
net_c.apply(weights_init)
optimizer_c = optim.Adam(net_c.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
criterion = nn.MSELoss()

# train
start = datetime.now()
print(start)
print('training')
net_c.train()
writer = SummaryWriter(comment='_content')

# log
writer.add_graph(net_c, dataset.get_random_heatmap_with_caption(batch_size).get('heatmap').to(device))

# number of batches
batch_number = len(data_loader)

for e in range(epoch):
    print('learning rate: ' + str(optimizer_c.param_groups[0].get('lr')))

    for i, batch in enumerate(data_loader, 0):
        # optimize style encoder
        net_c.zero_grad()

        # get heatmaps and sentence vectors
        text = batch.get('vector')
        heatmap = batch.get('heatmap')
        current_batch_size = len(text)

        text = text.to(device)
        heatmap = heatmap.to(device)

        # predict content
        content = net_c(heatmap)

        # calculate losses and update
        loss = criterion(content, text)
        loss.backward()
        optimizer_c.step()

        # log
        writer.add_scalar('loss/c', loss, batch_number * e + i)

        # print progress
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
            batch_number) + ' loss: ' + str(loss.item()))

    # save model
    torch.save(net_c.state_dict(), content_encoder_path + '_' + f'{e + 1:05d}')

    # validate
    net_c.eval()
    with torch.no_grad():
        content_val = net_c(heatmap_val).detach()
        loss_val = criterion(content_val, text_val)
        print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' val loss: ' + str(loss_val.item()))
        writer.add_scalar('loss_val/c', loss_val, e)
    net_c.train()

print('\nfinished')
print(datetime.now())
print('(started ' + str(start) + ')')
writer.close()
