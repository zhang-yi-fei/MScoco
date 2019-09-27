import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage import io
from math import sin, cos, pi

generator_path = 'models/generator'
discriminator_path = 'models/discriminator'
image_folder = '/media/data/yzhang2/coco/train/coco/images/'
caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
keypoint_path = '/media/data/yzhang2/coco/train/coco/annotations/person_keypoints_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
total_keypoints = 17
keypoint_colors = ['#057020', '#11bb3b', '#12ca3e', '#11bb3b', '#12ca3e', '#1058d1', '#2e73e5', '#cabe12', '#eae053',
                   '#cabe12', '#eae053', '#1058d1', '#2e73e5', '#9dc15c', '#b1cd7e', '#9dc15c', '#b1cd7e']
skeleton_colors = ['#b0070a', '#b0070a', '#f40b0f', '#f40b0f', '#ec7f18', '#ad590b', '#ef9643', '#ec7f18', '#952fe9',
                   '#b467f4', '#952fe9', '#b467f4', '#ee6da5', '#ee6da5', '#ee6da5', '#c8286e', '#e47ca9', '#c8286e',
                   '#e47ca9']

# ground truth size in heatmap
sigma = 2

# size of heatmap input to network
heatmap_size = int(64)

# heatmap augmentation parameters
flip = 0.5
rotate = 10
scale = 1
translate = 0
left_right_swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# size of text encoding
sentence_vector_size = 300

# size of compressed text encoding
compress_size = 128

# text encoding interpolation
beta = 0.5

# numbers of channels of the convolutions
convolution_channel_g = [1024, 512, 256, 128]
convolution_channel_d = [128, 256, 512, 1024]

noise_size = 128
g_input_size = noise_size + compress_size
d_final_size = convolution_channel_d[-1]

# x-y grids
x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')

# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.5

# have more than this number of keypoints to be included
keypoint_threshold = 7


# return ground truth heatmap of a training image (fixed-sized square-shaped, can be augmented)
def get_heatmap(keypoint, augment=True):
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

    if augment:
        # do heatmap augmentation
        x = x - heatmap_half
        y = y - heatmap_half

        # random flip
        if random.random() < flip:
            x = -x

            # when flipped, left and right should be swapped
            x = x[left_right_swap]
            y = y[left_right_swap]
            v = v[left_right_swap]

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
    # locate the keypoints (the maximum of each channel)
    heatmap_max = np.amax(np.amax(heatmap, axis=1), axis=1)
    x_keypoint = np.argmax(np.amax(heatmap, axis=1), axis=1)
    y_keypoint = np.argmax(np.amax(heatmap, axis=2), axis=1)
    keypoint_show = np.arange(total_keypoints)[heatmap_max > heatmap_threshold]

    # option to plot skeleton
    x_skeleton = []
    y_skeleton = []
    skeleton_show = []
    if skeleton is not None:
        x_skeleton = x_keypoint[skeleton]
        y_skeleton = y_keypoint[skeleton]
        skeleton_show = [i for i in range(len(skeleton)) if (heatmap_max[skeleton[i]] > heatmap_threshold).all()]

    # get a heatmap in single image with colors
    heatmap_color = np.empty((total_keypoints, heatmap_size, heatmap_size, 3), dtype='float32')
    for i in range(total_keypoints):
        heatmap_color[i] = np.tile(np.array(matplotlib.colors.to_rgb(keypoint_colors[i])),
                                   (heatmap_size, heatmap_size, 1))
        for j in range(3):
            heatmap_color[i, :, :, j] = heatmap_color[i, :, :, j] * heatmap[i]
    heatmap_color = np.amax(heatmap_color, axis=0)

    # plot the heatmap in black-white and the optional training image
    if image_path is not None:
        image = io.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap_color)
        [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=2) for i in skeleton_show]
        [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=4, markeredgecolor='k',
                  markeredgewidth=1) for i in keypoint_show]
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('training image')
    else:
        plt.imshow(heatmap_color)
        [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=2) for i in skeleton_show]
        [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=4, markeredgecolor='k',
                  markeredgewidth=1) for i in keypoint_show]
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
                            data['vector'] = [
                                text_model.get_sentence_vector(caption.get('caption').replace('\n', '')) * 100 / pi
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

        if self.with_vector:
            for i in range(number):
                # randomly select from all captions
                vector = random.choice(random.choice(self.dataset).get('vector'))
                vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)

        return vector_tensor.unsqueeze(-1).unsqueeze(-1)

    # get a batch of random caption from the whole dataset
    def get_random_heatmap_with_caption(self, number):
        caption = [[]] * number
        heatmap = torch.empty((number, total_keypoints, heatmap_size, heatmap_size), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            heatmap[i] = torch.tensor(get_heatmap(data.get('keypoint'), augment=False) * 2 - 1, dtype=torch.float32)
            caption[i] = random.choice(data.get('caption')).get('caption')

        return {'heatmap': heatmap, 'caption': caption}

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        if self.with_vector:
            for i in range(number):
                # randomly select 2 captions from all captions
                vector = random.choice(random.choice(self.dataset).get('vector'))
                vector2 = random.choice(random.choice(self.dataset).get('vector'))

                # interpolate caption sentence vectors
                interpolated_vector = beta * vector + (1 - beta) * vector2
                vector_tensor[i] = torch.tensor(interpolated_vector, dtype=torch.float32)

        return vector_tensor.unsqueeze(-1).unsqueeze(-1)


# generator given noise input
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # several layers of transposed convolution, batch normalization and ReLu
        self.first = nn.ConvTranspose2d(noise_size, convolution_channel_g[0], 4, 1, 0, bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(convolution_channel_g[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[0], convolution_channel_g[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[1], convolution_channel_g[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[2], convolution_channel_g[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_g[3]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel_g[3], total_keypoints, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, noise_vector):
        return self.main(self.first(noise_vector))


# discriminator given heatmap
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # several layers of convolution and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel_d[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[0], convolution_channel_d[1], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[1], convolution_channel_d[2], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[2], convolution_channel_d[3], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.second = nn.Conv2d(convolution_channel_d[-1], d_final_size, 1, bias=False)
        self.third = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False)

        )

    def forward(self, input_heatmap):
        return self.third(self.second(self.main(input_heatmap)))


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


# generator given noise and text encoding input
class Generator2(Generator):
    def __init__(self):
        super(Generator2, self).__init__()

        self.first2 = nn.ConvTranspose2d(g_input_size, convolution_channel_g[0], 4, 1, 0, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise_vector, sentence_vector):
        # concatenate noise vector and compressed sentence vector
        input_vector = torch.cat((noise_vector, (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1))), 1)

        return self.main(self.first2(input_vector))


# discriminator given heatmap and sentence vector
class Discriminator2(Discriminator):
    def __init__(self):
        super(Discriminator2, self).__init__()

        # convolution with concatenated sentence vector
        self.second2 = nn.Conv2d(convolution_channel_d[-1] + compress_size, d_final_size, 1, bias=False)

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input_heatmap, sentence_vector):
        # first convolution, then concatenate sentence vector
        tensor = torch.cat((self.main(input_heatmap), (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1)).repeat(1, 1,
                                                                                                                  4,
                                                                                                                  4)),
                           1)
        return self.third(self.second2(tensor))


# join a generator and a discriminator for display in Tensorboard
class JoinGAN2(nn.Module):
    def __init__(self):
        super(JoinGAN2, self).__init__()
        self.g = Generator2()
        self.d = Discriminator2()

    def forward(self, noise_vector, sentence_vector):
        return self.d(self.g(noise_vector, sentence_vector), sentence_vector)


# second GAN model
class GAN2(object):
    def __init__(self, generator_path, discriminator_path, text_model, device=torch.device('cpu')):
        # load generator and Discriminator models
        self.net_g = Generator2()
        self.net_d = Discriminator2()
        self.net_g.load_state_dict(torch.load(generator_path))
        self.net_d.load_state_dict(torch.load(discriminator_path))
        self.device = device
        self.net_g.to(self.device)
        self.net_d.to(self.device)
        self.net_g.eval()
        self.net_d.eval()
        self.text_model = text_model

    # generate a heatmap from noise
    def generate(self, caption, noise=None):
        if noise is None:
            noise = torch.randn(noise_size, dtype=torch.float32)
        noise_vector = noise.view(1, noise_size, 1, 1).to(self.device)
        sentence_vector = self.text_model.get_sentence_vector(caption.replace('\n', '')) * 100 / pi
        sentence_vector = torch.tensor(sentence_vector, dtype=torch.float32).view(1, sentence_vector_size, 1, 1).to(
            self.device)

        # generate
        with torch.no_grad():
            heatmap = self.net_g(noise_vector, sentence_vector)
        return np.array(heatmap.squeeze().tolist()) * 0.5 + 0.5

    # discriminate a heatmap
    def discriminate(self, heatmap, caption):
        # heatmap to tensor
        heatmap = torch.tensor(heatmap * 2 - 1, dtype=torch.float32, device=self.device).view(1, total_keypoints,
                                                                                              heatmap_size,
                                                                                              heatmap_size)
        sentence_vector = self.text_model.get_sentence_vector(caption.replace('\n', '')) * 100 / pi
        sentence_vector = torch.tensor(sentence_vector, dtype=torch.float32).view(1, sentence_vector_size, 1, 1).to(
            self.device)

        # discriminate
        with torch.no_grad():
            score = self.net_d(heatmap, sentence_vector)
        return score.item()
