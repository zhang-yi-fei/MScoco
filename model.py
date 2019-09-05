import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import io
from math import sin, cos, pi

generator_path = 'models/generator'
discriminator_path = 'models/discriminator'
image_folder = '/media/data/yzhang2/coco/train/coco/images/'
caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
keypoint_path = '/media/data/yzhang2/coco/train/coco/annotations/person_keypoints_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
total_keypoints = 17

# ground truth size in heatmap
sigma = 1

# size of heatmap input to network
heatmap_size = int(32)

# heatmap augmentation parameters
flip = 0.5
rotate = 10
scale = 1
translate = 0

# size of text encoding
sentence_vector_size = 300

# size of compressed text encoding
compress_size = 0

# text encoding interpolation
beta = 0.5

# numbers of channels of the convolutions
convolution_channel_g = [1024, 512, 256]
convolution_channel_d = [256, 512, 1024]

noise_size = 100
g_input_size = noise_size + compress_size
d_final_size = convolution_channel_d[0]

# x-y grids
x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')

# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.5

# have more than this number of keypoints to be included
keypoint_threshold = 8


# return ground truth heatmap of a training image (fixed-sized square-shaped, augmented)
def get_heatmap(keypoint):
    # heatmap dimension is (number of keypoints)*(heatmap size)*(heatmap size)
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    heatmap = np.empty((total_keypoints, heatmap_size, heatmap_size), dtype='float32')

    # keypoints location (x, y) and visibility (v)
    x = np.array(keypoint.get('keypoints')[0::3])
    y = np.array(keypoint.get('keypoints')[1::3])
    v = np.array(keypoint.get('keypoints')[2::3])

    # calculate the scaling
    heatmap_half = heatmap_size / 2
    if h > w:
        x = heatmap_half - w / h * heatmap_half + (x - x0) / h * heatmap_size
        y = (y - y0) / h * heatmap_size
    else:
        x = (x - x0) / w * heatmap_size
        y = heatmap_half - h / w * heatmap_half + (y - y0) / w * heatmap_size

    # do heatmap augmentation
    x = x - heatmap_half
    y = y - heatmap_half

    # random flip
    if random.random() < flip:
        x = -x

    # random rotation
    a = random.uniform(-rotate, rotate) * pi / 180
    sin_a = sin(a)
    cos_a = cos(a)
    x, y = tuple(np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), np.array([x, y])))

    # random scaling
    a = random.uniform(scale, 1 / scale)
    x = x * a
    y = y * a

    # random translation
    x = x + random.uniform(-translate, translate) + heatmap_half
    y = y + random.uniform(-translate, translate) + heatmap_half

    for i in range(total_keypoints):
        # labeled keypoints' v > 0
        if v[i] > 0:
            # ground truth in heatmap is normal distribution shaped
            heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / sigma ** 2, dtype='float32')
        else:
            heatmap[i] = empty.copy()

    return heatmap


# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None):
    x_skeleton = np.empty((2, 0), dtype='int32')
    y_skeleton = np.empty((2, 0), dtype='int32')

    # locate the keypoints (the maximum of each channel)
    keypoint = np.empty((2, total_keypoints), dtype='int32')
    heatmap_max = np.amax(np.amax(heatmap, axis=1), axis=1)
    keypoint[0] = np.argmax(np.amax(heatmap, axis=1), axis=1)
    keypoint[1] = np.argmax(np.amax(heatmap, axis=2), axis=1)

    # option to plot skeleton
    if skeleton is not None:
        for line in skeleton:
            if (heatmap[(line,) + tuple(keypoint[::-1, line])] > heatmap_threshold).all():
                # keypoint is located in the maximum of the heatmap
                x_skeleton = np.hstack((x_skeleton, keypoint[0:1, line].transpose()))
                y_skeleton = np.hstack((y_skeleton, keypoint[1:2, line].transpose()))

    keypoint = keypoint[:, heatmap_max > heatmap_threshold]

    # get a heatmap in single image
    heatmap = np.amax(heatmap, axis=0)

    # plot the heatmap in black-white and the optional training image
    if image_path is not None:
        image = io.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap, 'gray', vmin=0.0, vmax=1.0)
        plt.plot(x_skeleton, y_skeleton, 'red', linewidth=2)
        plt.plot(keypoint[0], keypoint[1], 'og', markersize=4)
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('training image')
    else:
        plt.imshow(heatmap, 'gray', vmin=0.0, vmax=1.0)
        plt.plot(x_skeleton, y_skeleton, 'red', linewidth=2)
        plt.plot(keypoint[0], keypoint[1], 'og', markersize=4)
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)


# get a batch of noise vectors
def get_noise_tensor(number):
    noise_tensor = torch.randn((number, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor


# a dataset that constructs heatmaps and optional matching caption encodings tensors on the fly
class HeatmapDataset(torch.utils.data.Dataset):
    # a dataset contains keypoints and captions, can add sentence encoding
    def __init__(self, coco_keypoint, coco_caption, single_person=False, text_model=None):
        # get all containing 'person' image ids
        image_ids = coco_keypoint.getImgIds()

        self.with_vector = (text_model is not None)
        self.dataset = []

        for image_id in image_ids:
            keypoint_ids = coco_keypoint.getAnnIds(imgIds=image_id)
            if (single_person and len(keypoint_ids) == 1) or (not single_person):
                caption_ids = coco_caption.getAnnIds(imgIds=image_id)
                captions = coco_caption.loadAnns(ids=caption_ids)

                # each person in the image
                for keypoint in coco_keypoint.loadAnns(ids=keypoint_ids):
                    # with enough keypoints
                    if keypoint.get('num_keypoints') > keypoint_threshold:
                        data = {'keypoint': keypoint.copy(), 'caption': captions.copy(),
                                'image': coco_keypoint.loadImgs(image_id)[0]}

                        # add sentence encoding
                        if text_model is not None:
                            data['vector'] = [text_model.get_sentence_vector(caption.get('caption').replace('\n', ''))
                                              for caption in captions]
                        self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # change heatmap range from [0,1] to[-1,1]
        heatmap = torch.tensor(get_heatmap(data.get('keypoint')) * 2 - 1, dtype=torch.float32)
        if self.with_vector:
            # randomly select from all matching captions
            vector = torch.tensor(random.choice(data.get('vector')), dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            return {'heatmap': heatmap, 'vector': vector}
        return {'heatmap': heatmap}

    # get a batch of random caption sentence vectors from the whole dataset
    def get_random_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        for i in range(number):
            # randomly select from all captions
            vector = random.choice(random.choice(self.dataset).get('vector'))
            vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)

        return vector_tensor.unsqueeze(-1).unsqueeze(-1)

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        for i in range(number):
            # randomly select 2 captions from all captions
            vector = random.choice(random.choice(self.dataset).get('vector'))
            vector2 = random.choice(random.choice(self.dataset).get('vector'))

            # interpolate caption sentence vectors
            interpolated_vector = beta * vector + (1 - beta) * vector2
            vector_tensor[i] = torch.tensor(interpolated_vector, dtype=torch.float32)

        return vector_tensor.unsqueeze(-1).unsqueeze(-1)


# generator given noise and text encoding input
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # several layers of transposed convolution, batch normalization and ReLu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(g_input_size, convolution_channel_g[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(convolution_channel_g[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[0], convolution_channel_g[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[1], convolution_channel_g[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[2], total_keypoints, 4, 2, 1, bias=False),
            nn.Tanh()

        )

        # compress text encoding first
        # self.compress = nn.Identity(
        #     nn.Linear(sentence_vector_size, compress_size),
        #     nn.BatchNorm1d(compress_size),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.compress = nn.Identity()

    def forward(self, noise_vector):
        # concatenate noise vector and compressed sentence vector
        # input_vector = torch.cat((noise_vector, (
        #     (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1))), 1)
        # input_vector = noise_vector

        # return self.main(input_vector)
        return self.main(noise_vector)


# Discriminator given heatmap and sentence vector
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # several layers of convolution, batch normalization and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel_d[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[0], convolution_channel_d[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[1], convolution_channel_d[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[2], 1, 4, 1, 0, bias=False)

        )

        # compute final score of the Discriminator with concatenated sentence vector
        # self.second = nn.Sequential(
        #     nn.Conv2d(convolution_channel_d[3] + compress_size, d_final_size, 1, bias=False),
        #     nn.BatchNorm2d(d_final_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(d_final_size, 1, 4, bias=False),
        #     nn.Sigmoid()
        # )

        # compress text encoding first
        # self.compress = nn.Sequential(
        #     nn.Linear(sentence_vector_size, compress_size),
        #     nn.BatchNorm1d(compress_size),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.compress = nn.Identity()

    def forward(self, input_heatmap):
        # first transposed convolution, then sentence vector
        # tensor = torch.cat((self.main(input_heatmap), (
        #     (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1)).repeat(1, 1,
        #                                                                                                           4,
        #                                                                                                           4)),
        #                    1)
        # tensor = self.main(input_heatmap)

        return self.main(input_heatmap)


# custom weights initialization called on net_g and net_d
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.normal_(m.bias.data, 0.0, 0.02)


# a GAN model
class GAN(object):
    def __init__(self, generator_path, discriminator_path, device=torch.device('cpu')):
        # load generator and Discriminator models
        self.net_g = Generator()
        self.net_d = Discriminator()
        self.net_g.load_state_dict(torch.load(generator_path))
        self.net_d.load_state_dict(torch.load(discriminator_path))
        self.device = device
        self.net_g.to(self.device)
        self.net_d.to(self.device)
        self.net_g.eval()
        self.net_d.eval()

    # generate a heatmap from noise
    def generate(self, noise=None):
        if noise is None:
            noise = torch.randn(noise_size, dtype=torch.float32)
        noise_vector = noise.view(1, noise_size, 1, 1).to(self.device)

        # generate
        with torch.no_grad():
            heatmap = self.net_g(noise_vector)
        return np.array(heatmap.squeeze().tolist()) * 0.5 + 0.5

    # discriminate a heatmap
    def discriminate(self, heatmap):
        # heatmap to tensor
        heatmap = torch.tensor(heatmap * 2 - 1, dtype=torch.float32, device=self.device).view(1, total_keypoints,
                                                                                              heatmap_size,
                                                                                              heatmap_size)

        # discriminate
        with torch.no_grad():
            score = self.net_d(heatmap)
        return score.item()


# join a generator and a discriminator for display in Tensorboard
class JoinGAN(nn.Module):
    def __init__(self):
        super(JoinGAN, self).__init__()
        self.g = Generator()
        self.d = Discriminator()

    def forward(self, noise_vector):
        return self.d(self.g(noise_vector))
