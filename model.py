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
image_folder_val = '/media/data/yzhang2/coco/val/coco/images/'
caption_path_val = '/media/data/yzhang2/coco/val/coco/annotations/captions_val2017.json'
keypoint_path_val = '/media/data/yzhang2/coco/val/coco/annotations/person_keypoints_val2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
total_keypoints = 17
keypoint_colors = ['#057020', '#11bb3b', '#12ca3e', '#11bb3b', '#12ca3e', '#1058d1', '#2e73e5', '#cabe12', '#eae053',
                   '#cabe12', '#eae053', '#1058d1', '#2e73e5', '#9dc15c', '#b1cd7e', '#9dc15c', '#b1cd7e']
keypoint_colors_reverse = [hex(0xffffff - int(c.replace('#', '0x'), 16)).replace('0x', '#') for c in keypoint_colors]
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

# weight of text encoding
encoding_weight = 30

# size of compressed text encoding
compress_size = 128

# text encoding interpolation
beta = 0.5

# numbers of channels of the convolutions
convolution_channel_g = [256, 128, 64, 32]
convolution_channel_d = [32, 64, 128, 256]

# hidden layers
hidden = [250, 200]

# coordinate scaling
factor = 20

# visible threshold
v_threshold = 0.8

noise_size = 128
g_input_size = noise_size + compress_size
d_final_size = convolution_channel_d[-1]

# x-y grids
x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')

# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.2

# have more than this number of keypoints to be included
keypoint_threshold = 7


# do heatmap augmentation
def augment_heatmap(x, y, v, heatmap_half, f, a, s, tx, ty):
    x = x - heatmap_half
    y = y - heatmap_half

    # flip
    if f:
        x = -x

        # when flipped, left and right should be swapped
        x = x[left_right_swap]
        y = y[left_right_swap]
        v = v[left_right_swap]

    # rotation
    sin_a = sin(a)
    cos_a = cos(a)
    x, y = tuple(np.dot(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), np.array([x, y])))

    # scaling
    x = x * s
    y = y * s

    # translation
    x = x + tx + heatmap_half
    y = y + ty + heatmap_half

    return x, y, v


# return parameters for an augmentation
def get_augment_parameters(flip, scale, rotate, translate):
    # random flip, rotation, scaling, translation
    f = random.random() < flip
    a = random.uniform(-rotate, rotate) * pi / 180
    s = random.uniform(scale, 1 / scale)
    tx = random.uniform(-translate, translate)
    ty = random.uniform(-translate, translate)
    return f, a, s, tx, ty


# return coordinates of keypoints in the heatmap and visibility
def get_coordinates(x0, y0, w, h, keypoint):
    # keypoints location (x, y) and visibility (v, 0 invisible, 1 visible)
    x = np.array(keypoint.get('keypoints')[0::3])
    y = np.array(keypoint.get('keypoints')[1::3])
    v = np.array(keypoint.get('keypoints')[2::3]).clip(0, 1)

    # calculate the scaling
    heatmap_half = heatmap_size / 2
    if h > w:
        x = heatmap_half - w / h * heatmap_half + (x - x0) / h * heatmap_size
        y = (y - y0) / h * heatmap_size
    else:
        x = (x - x0) / w * heatmap_size
        y = heatmap_half - h / w * heatmap_half + (y - y0) / w * heatmap_size

    # set invisible keypoint coordinates as (0,0)
    x[v < 1] = 0
    y[v < 1] = 0

    return x, y, v


# return coordinates of keypoints in the heatmap and visibility with augmentation
def get_augmented_coordinates(keypoint):
    # coordinates and visibility before augmentation
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    x, y, v = get_coordinates(x0, y0, w, h, keypoint)

    # random flip, rotation, scaling, translation
    f, a, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)
    x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, a, s, tx, ty)

    # set invisible keypoint coordinates as (0,0)
    x[v < 1] = 0
    y[v < 1] = 0

    # concatenate the coordinates and visibility
    return np.concatenate([x, y, v])


# seperate coordinates and visibility from an array
def result_to_coordinates(result):
    x = result[0:total_keypoints]
    y = result[total_keypoints:2 * total_keypoints]
    v = (np.sign(result[2 * total_keypoints:3 * total_keypoints] - v_threshold) + 1) / 2
    return x, y, v


# return ground truth heatmap of a training sample (fixed-sized square-shaped, can be augmented)
def get_heatmap(keypoint, augment=True):
    # heatmap dimension is (number of keypoints)*(heatmap size)*(heatmap size)
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    heatmap = np.empty((total_keypoints, heatmap_size, heatmap_size), dtype='float32')

    # keypoints location (x, y) and visibility (v)
    x, y, v = get_coordinates(x0, y0, w, h, keypoint)

    # do heatmap augmentation
    if augment:
        # random flip, rotation, scaling, translation
        f, a, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)
        x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, a, s, tx, ty)

    for i in range(total_keypoints):
        # labeled keypoints' v > 0
        if v[i] > 0:
            # ground truth in heatmap is normal distribution shaped
            heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / (2 * sigma ** 2), dtype='float32')
        else:
            heatmap[i] = empty.copy()

    return heatmap


# return ground truth heatmap of a whole training image (fixed-sized square-shaped, can be augmented)
def get_full_image_heatmap(image, keypoints, augment=True):
    # heatmap dimension is (number of keypoints)*(heatmap size)*(heatmap size)
    h = image.get('height')
    w = image.get('width')
    heatmap = np.empty((len(keypoints), total_keypoints, heatmap_size, heatmap_size), dtype='float32')

    if augment:
        # random flip, rotation, scaling, translation
        f, a, s, tx, ty = get_augment_parameters(flip, scale, rotate, translate)

    # create individual heatmaps
    for j, keypoint in enumerate(keypoints, 0):
        # keypoints location (x, y) and visibility (v)
        x, y, v = get_coordinates(0, 0, w, h, keypoint)

        # do heatmap augmentation
        if augment:
            x, y, v = augment_heatmap(x, y, v, heatmap_size / 2, f, a, s, tx, ty)

        for i in range(total_keypoints):
            # labeled keypoints' v > 0
            if v[i] > 0:
                # ground truth in heatmap is normal distribution shaped
                heatmap[j][i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / (2 * sigma ** 2),
                                       dtype='float32')
            else:
                heatmap[j][i] = empty.copy()

    # sum individual heatmaps
    return heatmap.sum(axis=0).clip(0, 1)


# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None, only_skeleton=False):
    # locate the keypoints (the maximum of each channel)
    heatmap_max = np.amax(np.amax(heatmap, axis=1), axis=1)
    index_max = np.array([np.unravel_index(np.argmax(h), h.shape) for h in heatmap])
    x_keypoint = index_max[:, 1]
    y_keypoint = index_max[:, 0]
    keypoint_show = np.arange(total_keypoints)[heatmap_max > heatmap_threshold]

    # option to plot skeleton
    x_skeleton = []
    y_skeleton = []
    skeleton_show = []
    if skeleton is not None:
        skeleton = skeleton[0:17]
        x_skeleton = x_keypoint[skeleton]
        y_skeleton = y_keypoint[skeleton]
        skeleton_show = [i for i in range(len(skeleton)) if (heatmap_max[skeleton[i]] > heatmap_threshold).all()]

        # only plot keypoints and skeleton
        if only_skeleton:
            plt.imshow(np.ones((64, 64, 3), 'float32'))
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
            plt.title('pose')
            plt.xlabel(caption)
            return

    # get a heatmap in single image with colors
    heatmap_color = np.empty((total_keypoints, heatmap_size, heatmap_size, 3), dtype='float32')
    for i in range(total_keypoints):
        heatmap_color[i] = np.tile(np.array(matplotlib.colors.to_rgb(keypoint_colors_reverse[i])),
                                   (heatmap_size, heatmap_size, 1))
        for j in range(3):
            heatmap_color[i, :, :, j] = heatmap_color[i, :, :, j] * heatmap[i]
    heatmap_color = 1 - np.amax(heatmap_color, axis=0)

    # plot the heatmap in black-white and the optional training image
    if image_path is not None:
        image = io.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap_color)
        if skeleton is not None:
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('training image')
    else:
        plt.imshow(heatmap_color)
        if skeleton is not None:
            [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
            [plt.plot(x_keypoint[i], y_keypoint[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k',
                      markeredgewidth=1) for i in keypoint_show]
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)


# plot a pose
def plot_pose(x, y, v, skeleton, caption=None):
    # visible keypoints
    keypoint_show = np.arange(total_keypoints)[v > 0]

    # visible skeleton
    skeleton = skeleton[0:17]
    x_skeleton = x[skeleton]
    y_skeleton = y[skeleton]
    skeleton_show = [i for i in range(len(skeleton)) if (v[skeleton[i]] > 0).all()]

    # plot keypoints and skeleton
    plt.imshow(np.ones((64, 64, 3), 'float32'))
    [plt.plot(x_skeleton[i], y_skeleton[i], c=skeleton_colors[i], linewidth=5) for i in skeleton_show]
    [plt.plot(x[i], y[i], 'o', c=keypoint_colors[i], markersize=10, markeredgecolor='k', markeredgewidth=1) for i in
     keypoint_show]
    plt.title('pose')
    plt.xlabel(caption)


# return the caption encoding
def get_caption_vector(text_model, caption):
    return text_model.get_sentence_vector(caption.replace('\n', '').lower()) * encoding_weight


# get a batch of noise vectors
def get_noise_tensor(number):
    noise_tensor = torch.randn((number, noise_size, 1, 1), dtype=torch.float32)
    return noise_tensor


# a dataset that constructs heatmaps and optional matching caption encodings tensors on the fly
class HeatmapDataset(torch.utils.data.Dataset):
    # a dataset contains keypoints and captions, can add sentence encoding
    def __init__(self, coco_keypoint, coco_caption, single_person=False, text_model=None, full_image=False,
                 for_regression=False):

        # get all containing 'person' image ids
        image_ids = coco_keypoint.getImgIds()

        self.with_vector = (text_model is not None)
        self.for_regression = for_regression
        if for_regression:
            full_image = False
        self.full_image = full_image
        self.dataset = []

        for image_id in image_ids:
            keypoint_ids = coco_keypoint.getAnnIds(imgIds=image_id)
            if len(keypoint_ids) > 0 and ((single_person and len(keypoint_ids) == 1) or (not single_person)):
                caption_ids = coco_caption.getAnnIds(imgIds=image_id)
                captions = coco_caption.loadAnns(ids=caption_ids)
                keypoints = coco_keypoint.loadAnns(ids=keypoint_ids)

                if full_image:
                    data = {'keypoints': [], 'caption': captions.copy(),
                            'image': coco_keypoint.loadImgs(image_id)[0]}
                    for keypoint in keypoints:
                        if keypoint.get('num_keypoints') > keypoint_threshold:
                            data['keypoints'].append(keypoint.copy())
                    if len(data['keypoints']) == 0:
                        continue

                    # add sentence encoding
                    if text_model is not None:
                        data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in
                                          captions]
                    self.dataset.append(data)
                else:
                    # each person in the image
                    for keypoint in keypoints:
                        # with enough keypoints
                        if keypoint.get('num_keypoints') > keypoint_threshold:
                            data = {'keypoint': keypoint.copy(), 'caption': captions.copy(),
                                    'image': coco_keypoint.loadImgs(image_id)[0]}

                            # add sentence encoding
                            if text_model is not None:
                                data['vector'] = [get_caption_vector(text_model, caption.get('caption')) for caption in
                                                  captions]
                            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    # return either individual heatmap of heatmap of a whole image
    def get_heatmap(self, data, augment=True):
        if self.full_image:
            return get_full_image_heatmap(data.get('image'), data.get('keypoints'), augment)
        else:
            return get_heatmap(data.get('keypoint'), augment)

    def __getitem__(self, index):
        data = self.dataset[index]
        item = dict()

        if self.for_regression:
            item['coordinates'] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
        else:
            # change heatmap range from [0,1] to[-1,1]
            item['heatmap'] = torch.tensor(self.get_heatmap(data) * 2 - 1, dtype=torch.float32)

        if self.with_vector:
            # randomly select from all matching captions
            item['vector'] = torch.tensor(random.choice(data.get('vector')), dtype=torch.float32)
            if not self.for_regression:
                item['vector'].unsqueeze_(-1).unsqueeze_(-1)
        return item

    # get a batch of random caption sentence vectors from the whole dataset
    def get_random_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size), dtype=torch.float32)

        if self.with_vector:
            for i in range(number):
                # randomly select from all captions
                vector = random.choice(random.choice(self.dataset).get('vector'))
                vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)

        if self.for_regression:
            return vector_tensor
        else:
            return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)

    # get a batch of random heatmaps and captions from the whole dataset
    def get_random_heatmap_with_caption(self, number):
        caption = []
        heatmap = torch.empty((number, total_keypoints, heatmap_size, heatmap_size), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            heatmap[i] = torch.tensor(self.get_heatmap(data, augment=False) * 2 - 1, dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'heatmap': heatmap, 'caption': caption}

    # get a batch of random coordinates and captions from the whole dataset
    def get_random_coordinates_with_caption(self, number):
        caption = []
        coordinates = torch.empty((number, total_keypoints * 3), dtype=torch.float32)

        for i in range(number):
            # randomly select from all images
            data = random.choice(self.dataset)
            coordinates[i] = torch.tensor(get_augmented_coordinates(data.get('keypoint')), dtype=torch.float32)
            caption.append(random.choice(data.get('caption')).get('caption'))

        return {'coordinates': coordinates, 'caption': caption}

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

        if self.for_regression:
            return vector_tensor
        else:
            return vector_tensor.unsqueeze_(-1).unsqueeze_(-1)


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
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator, self).__init__()

        # several layers of convolution and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel_d[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[0]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[0], convolution_channel_d[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[1]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[1], convolution_channel_d[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[2]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel_d[2], convolution_channel_d[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel_d[3]) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.second = nn.Conv2d(convolution_channel_d[-1], d_final_size, 1, bias=False)
        self.third = nn.Sequential(
            nn.BatchNorm2d(d_final_size) if bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() if sigmoid else nn.Identity()

        )

    def forward(self, input_heatmap):
        return self.third(self.second(self.main(input_heatmap)))


# generator given noise and sentence input (regression)
class Generator_R(nn.Module):
    def __init__(self):
        super(Generator_R, self).__init__()

        # several layers of FC and ReLu
        self.main = nn.Sequential(
            nn.Linear(noise_size + sentence_vector_size, hidden[0], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[1], total_keypoints * 3, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, noise_vector, sentence_vector):
        output = self.main(torch.cat((noise_vector, sentence_vector), 1))
        return torch.cat(
            (output[:, 0:total_keypoints * 2] * factor,
             self.sigmoid(output[:, total_keypoints * 2:total_keypoints * 3])),
            1)


# discriminator given coordinates and sentence input (regression)
class Discriminator_R(nn.Module):
    def __init__(self):
        super(Discriminator_R, self).__init__()

        # several layers of FC and ReLu
        self.main = nn.Sequential(
            nn.Linear(total_keypoints * 3 + sentence_vector_size, hidden[0], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.ReLU(True),
            nn.Linear(hidden[1], 1, bias=True)
        )

    def forward(self, input_coordinates, sentence_vector):
        input = torch.cat((input_coordinates[:, 0:total_keypoints * 2] / factor,
                           input_coordinates[:, total_keypoints * 2:total_keypoints * 3], sentence_vector), 1)
        return self.main(input)


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
    def __init__(self, bn=False, sigmoid=False):
        super(Discriminator2, self).__init__(bn, sigmoid)

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


# get max index of a heatmap
def heatmap_to_max_index(heatmap):
    max_index = np.array([np.unravel_index(np.argmax(h), h.shape) for h in heatmap])

    # set the index of heatmap below threshold to the middle
    for i in range(len(heatmap)):
        if heatmap[i][tuple(max_index[i])] < heatmap_threshold:
            max_index[i][:] = heatmap_size / 2
    return max_index


# get max index of a pose
def coordinates_to_max_index(x, y, v):
    max_index = np.array([x, y]).transpose()

    # set the invisible keypoints to the middle
    max_index[v < 1] = [heatmap_size / 2, heatmap_size / 2]
    return max_index


# distance between two heatmaps: the sum of the distances between maximum points of all 17 keypoint heatmaps
def heatmap_distance(heatmap_max_index, heatmap_max_index2):
    return sum(np.sqrt(np.sum((heatmap_max_index - heatmap_max_index2) ** 2, axis=1)))


# find the nearest neighbor distance of a heatmap in a list of heatmaps
def nearest_neighbor(heatmap_max_index, heatmap_max_index_list):
    distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[0])

    # find nearest neighbor
    for heatmap_max_index2 in heatmap_max_index_list[1:]:
        new_distance = heatmap_distance(heatmap_max_index, heatmap_max_index2)
        if new_distance < distance:
            distance = new_distance
    return distance


# find the nearest neighbor of a heatmap in a list of heatmaps
def nearest_neighbor_index(heatmap_max_index, heatmap_max_index_list):
    distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[0])
    index = 0

    # find nearest neighbor
    for i in range(len(heatmap_max_index_list)):
        new_distance = heatmap_distance(heatmap_max_index, heatmap_max_index_list[i])
        if new_distance < distance:
            distance = new_distance
            index = i
    return index


# calculate the mean distance of a heatmap to a list of heatmaps
def mean_distance(heatmap_max_index, heatmap_max_index_list):
    distance = []

    # calculate distances
    for heatmap_max_index2 in heatmap_max_index_list:
        distance.append(heatmap_distance(heatmap_max_index, heatmap_max_index2))

    return np.mean(distance)


# calculate the mean distance of a vector to a list of vectors
def mean_vector_distance(vector, vector_list):
    distance = []

    # calculate distances
    for vector2 in vector_list:
        distance.append(np.sqrt(np.sum((vector - vector2) ** 2)))

    return np.mean(distance)


# find the nearest neighbor of a vector in a list of vectors
def vector_nearest_neighbor_index(vector, vector_list):
    distance = np.sqrt(np.sum((vector - vector_list[0]) ** 2))
    index = 0

    # find nearest neighbor
    for i in range(len(vector_list)):
        new_distance = np.sqrt(np.sum((vector - vector_list[i]) ** 2))
        if new_distance < distance:
            distance = new_distance
            index = i
    return index


# calculate the one-nearest-neighbor accuracy
def one_nearest_neighbor(heatmap_max_index_list, heatmap_max_index_list2):
    size = len(heatmap_max_index_list)

    # number of correct classifications
    count = 0
    for i in range(size):
        # a heatmap from the first list
        if nearest_neighbor(heatmap_max_index_list[i],
                            heatmap_max_index_list[0:i] + heatmap_max_index_list[i + 1:]) < nearest_neighbor(
            heatmap_max_index_list[i], heatmap_max_index_list2):
            count = count + 1

        # a heatmap from the second list
        if nearest_neighbor(heatmap_max_index_list2[i],
                            heatmap_max_index_list2[0:i] + heatmap_max_index_list2[i + 1:]) < nearest_neighbor(
            heatmap_max_index_list2[i], heatmap_max_index_list):
            count = count + 1

    # accuracy
    return count / size / 2
