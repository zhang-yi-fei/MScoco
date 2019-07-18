import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage

generator_path = 'generator'
discriminator_path = 'discriminator'
image_folder = '/media/data/yzhang2/coco/train/coco/images/'
caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
keypoint_path = '/media/data/yzhang2/coco/train/coco/annotations/person_keypoints_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
total_keypoints = 17

# ground truth size in heatmap
sigma = 2

# size of heatmap input to network
heatmap_size = 64

# resized heatmap size
resize_high = 90
resize_low = 70

# size of text encoding
sentence_vector_size = 300

# size of compressed text encoding
compress_size = 0

# text encoding interpolation
beta = 0.5

# numbers of channels of the convolutions
convolution_channel = [128, 256, 512, 1024]

noise_size = 100
g_input_size = noise_size + compress_size
d_final_size = convolution_channel[-1]

# big 2d normal distribution
center = 2000
size = 2 * center + 1
grid = np.repeat(np.array([range(size)]), size, axis=0)
normal = np.exp(-((grid - center) ** 2 + (grid.transpose() - center) ** 2) / sigma ** 2, dtype='float32')
zero = np.zeros([size, size], dtype='float32')

# x-y grids
x_grid = np.repeat(np.array([range(heatmap_size)]), heatmap_size, axis=0)
y_grid = np.repeat(np.array([range(heatmap_size)]).transpose(), heatmap_size, axis=1)
empty = np.zeros([heatmap_size, heatmap_size], dtype='float32')
testmap = np.array([np.exp(-((x_grid - 3 * i) ** 2 + (y_grid - 3 * i) ** 2) / sigma ** 2, dtype='float32') for i in
                    range(total_keypoints)])

# to decide whether a keypoint is in the heatmap
heatmap_threshold = 0.5

# have more than this number of keypoint to be included
keypoint_threshold = 8


# return ground truth heat map of a training image
def get_heatmap(data):
    # get image size and keypoint locations
    image = data.get('image')
    keypoint = data.get('keypoint')

    # heatmap size is (number of keypoints)*(image height)*(image width)
    h = image.get('height')
    w = image.get('width')
    c = total_keypoints
    heatmap = np.empty((c, h, w))

    # keypoints location (x, y) and visibility (v)
    x = keypoint.get('keypoints')[0::3]
    y = keypoint.get('keypoints')[1::3]
    v = keypoint.get('keypoints')[2::3]

    for i in range(c):
        # labeled keypoints' v > 0
        if v[i] > 0:
            # # ground truth in heatmap is normal distribution shaped
            heatmap[i] = normal[(center - y[i]):(center - y[i] + h), (center - x[i]):(center - x[i] + w)].copy()
        else:
            heatmap[i] = zero[0:h, 0:w].copy()

    return heatmap


# return ground truth heat map of a training image (fixed-sized square-shaped)
def get_heatmap2(keypoint):
    # heatmap size is (number of keypoints)*(bounding box height)*(bounding box width)
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    c = total_keypoints
    heatmap = np.empty((c, heatmap_size, heatmap_size))

    # keypoints location (x, y) and visibility (v)
    x = np.array(keypoint.get('keypoints')[0::3])
    y = np.array(keypoint.get('keypoints')[1::3])
    v = np.array(keypoint.get('keypoints')[2::3])

    # calculate the scaling
    if h > w:
        x = heatmap_size / 2 - w / h * heatmap_size / 2 + (x - x0) / h * heatmap_size
        y = (y - y0) / h * heatmap_size
    else:
        x = (x - x0) / w * heatmap_size
        y = heatmap_size / 2 - h / w * heatmap_size / 2 + (y - y0) / w * heatmap_size

    for i in range(c):
        # labeled keypoints' v > 0
        if v[i] > 0:
            # ground truth in heatmap is normal distribution shaped
            heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / sigma ** 2, dtype='float32')
        else:
            heatmap[i] = empty.copy()
    # heatmap = testmap.copy()

    return heatmap


# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None):
    x_skeleton = np.empty((2, 0))
    y_skeleton = np.empty((2, 0))

    # option to plot skeleton
    if skeleton is not None:
        for line in skeleton:
            if np.any(heatmap[line[0]] > heatmap_threshold) and np.any(heatmap[line[1]] > heatmap_threshold):
                x_skeleton = np.hstack((x_skeleton, np.array([[0], [0]])))
                y_skeleton = np.hstack((y_skeleton, np.array([[0], [0]])))

                # keypoint is located in the maximum of the heatmap
                y_skeleton[0, -1], x_skeleton[0, -1] = np.unravel_index(np.argmax(heatmap[line[0]], axis=None),
                                                                        heatmap[line[0]].shape)
                y_skeleton[1, -1], x_skeleton[1, -1] = np.unravel_index(np.argmax(heatmap[line[1]], axis=None),
                                                                        heatmap[line[1]].shape)

    # locate the keypoints (the maximum of each channel)
    keypoint = np.empty((2, total_keypoints))
    heatmap_max = np.amax(np.amax(heatmap, axis=1), axis=1)
    keypoint[0] = np.argmax(np.amax(heatmap, axis=1), axis=1)
    keypoint[1] = np.argmax(np.amax(heatmap, axis=2), axis=1)
    keypoint = keypoint[:, heatmap_max > heatmap_threshold]

    # get a heatmap in single image
    heatmap = np.amax(heatmap, axis=0)
    plt.figure()

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

    plt.show()


# return a randomly resized, cropped and flipped heatmap
def augmented_heatmap(heatmap):
    # resize
    resize = random.randint(resize_low, resize_high)
    zoom = resize / min(heatmap[0].shape)
    heatmap = ndimage.zoom(heatmap, [1, zoom, zoom], order=1)

    # crop
    h, w = heatmap[0].shape
    i = random.randint(0, h - heatmap_size)
    j = random.randint(0, w - heatmap_size)
    heatmap = heatmap[:, i:(i + heatmap_size), j:(j + heatmap_size)]

    # flip
    if random.random() < 0.5:
        heatmap = np.flip(heatmap, 2).copy()

    # put the images into one 3D tensor
    return heatmap


# get a batch of noise vectors
def get_noise_tensor(number):
    noise_tensor = torch.empty((number, noise_size, 1, 1))
    for i in range(number):
        noise_tensor[i] = torch.randn(noise_size, 1, 1)
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

            # limit to single-person images
            if single_person:
                # only one person in the image
                if len(keypoint_ids) == 1:
                    keypoint = coco_keypoint.loadAnns(ids=keypoint_ids)[0]

                    # with enough keypoints
                    if keypoint.get('num_keypoints') > keypoint_threshold:
                        caption_ids = coco_caption.getAnnIds(imgIds=image_id)
                        captions = coco_caption.loadAnns(ids=caption_ids)
                        data = {'keypoint': keypoint.copy(), 'caption': captions.copy(),
                                'image': coco_keypoint.loadImgs(image_id)[0]}

                        # add sentence encoding
                        if text_model is not None:
                            data['vector'] = [text_model.get_sentence_vector(caption.get('caption').replace('\n', ''))
                                              for caption in captions]
                        self.dataset.append(data)

            # no person limit
            else:
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
        heatmap = torch.tensor(get_heatmap2(data.get('keypoint')) * 2 - 1, dtype=torch.float32)
        if self.with_vector:
            # randomly select from all matching captions
            vector = torch.tensor(random.choice(data.get('vector')), dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            return {'heatmap': heatmap, 'vector': vector}
        return {'heatmap': heatmap}

    # get a batch of random caption sentence vectors from the whole dataset
    def get_random_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size))

        for i in range(number):
            # randomly select from all captions
            vector = random.choice(random.choice(self.dataset).get('vector'))
            vector_tensor[i] = torch.tensor(vector, dtype=torch.float32)

        return vector_tensor.unsqueeze(-1).unsqueeze(-1)

    # get a batch of random interpolated caption sentence vectors from the whole dataset
    def get_interpolated_caption_tensor(self, number):
        vector_tensor = torch.empty((number, sentence_vector_size))

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
            nn.ConvTranspose2d(g_input_size, convolution_channel[-1], 4, 1, 0, bias=False),
            nn.BatchNorm2d(convolution_channel[-1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel[-1], convolution_channel[-2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[-2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel[-2], convolution_channel[-3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[-3]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel[-3], convolution_channel[-4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[-4]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel[-4], total_keypoints, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(total_keypoints),
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


# discriminator given heatmap and sentence vector
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # several layers of convolution, batch normalization and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel[0], 4, 2, 1, bias=False),
            # nn.BatchNorm2d(convolution_channel[0]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel[0], convolution_channel[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel[1], convolution_channel[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[2]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(convolution_channel[2], convolution_channel[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[3]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_final_size, 1, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()

        )

        # compute final score of the discriminator with concatenated sentence vector
        # self.second = nn.Sequential(
        #     nn.Conv2d(convolution_channel[3] + compress_size, d_final_size, 1, bias=False),
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
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('Linear') != -1:
    #     nn.init.kaiming_normal_(m.weight.data)
    #     nn.init.constant_(m.bias.data, 0.0)


# using trained generator, give a caption, plot a generated heatmap
def plot_caption(caption, text_model, generator_path, device, skeleton=None):
    # load generator model
    net_g = Generator()
    net_g.load_state_dict(torch.load(generator_path))
    net_g.to(device)
    net_g.eval()

    # sentence and noise vector as input to the generator
    sentence_vector = torch.tensor(text_model.get_sentence_vector(caption), dtype=torch.float32,
                                   device=device).view(1, sentence_vector_size, 1, 1)
    noise_vector = torch.randn(1, noise_size, 1, 1, device=device)

    # generate
    with torch.no_grad():
        heatmap = net_g(noise_vector)

    # plot heatmap
    plot_heatmap(np.array(heatmap.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)


# using trained generator, plot a generated heatmap
def plot_fake(generator_path, device, skeleton=None):
    # load generator model
    net_g = Generator()
    net_g.load_state_dict(torch.load(generator_path))
    net_g.to(device)
    net_g.eval()

    # noise vector as input to the generator
    noise_vector = torch.randn(1, noise_size, 1, 1, device=device)

    # generate
    with torch.no_grad():
        heatmap = net_g(noise_vector)

    # plot heatmap
    plot_heatmap(np.array(heatmap.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton)


# give generator a caption, generate a heatmap, then give it and another heatmap to discriminate
def discriminate(heatmap, caption, text_model, generator_path, discriminator_path, device, skeleton=None):
    # load generator and discriminator models
    net_g = Generator()
    net_d = Discriminator()
    net_g.load_state_dict(torch.load(generator_path))
    net_d.load_state_dict(torch.load(discriminator_path))
    net_g.to(device)
    net_d.to(device)
    net_g.eval()
    net_d.eval()

    # sentence and noise vector as input to the generator
    sentence_vector = torch.tensor(text_model.get_sentence_vector(caption), dtype=torch.float32,
                                   device=device).view(1, sentence_vector_size, 1, 1)
    noise_vector = torch.randn(1, noise_size, 1, 1, device=device)

    # heatmap to tensor
    heatmap_real = torch.tensor(heatmap * 2 - 1, dtype=torch.float32, device=device).view(1, total_keypoints,
                                                                                          heatmap_size, heatmap_size)

    # generate and discriminate
    with torch.no_grad():
        heatmap_generated = net_g(noise_vector)
        score_real = net_d(heatmap_real)
        score_generated = net_d(heatmap_generated)

    # plot heatmaps
    plot_heatmap(np.array(heatmap_real.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)
    plot_heatmap(np.array(heatmap_generated.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)

    # print scores
    print('score of real heatmap: ' + str(score_real.item()))
    print('score of generated heatmap: ' + str(score_generated.item()))


# generate a heatmap, then give it and another heatmap to discriminate
def discriminate_fake(heatmap, generator_path, discriminator_path, device, skeleton=None):
    # load generator and discriminator models
    net_g = Generator()
    net_d = Discriminator()
    net_g.load_state_dict(torch.load(generator_path))
    net_d.load_state_dict(torch.load(discriminator_path))
    net_g.to(device)
    net_d.to(device)
    net_g.eval()
    net_d.eval()

    # noise vector as input to the generator
    noise_vector = torch.randn(1, noise_size, 1, 1, device=device)

    # heatmap to tensor
    heatmap_real = torch.tensor(heatmap * 2 - 1, dtype=torch.float32, device=device).view(1, total_keypoints,
                                                                                          heatmap_size, heatmap_size)

    # generate and discriminate
    with torch.no_grad():
        heatmap_generated = net_g(noise_vector)
        score_real = net_d(heatmap_real)
        score_generated = net_d(heatmap_generated)

    # plot heatmaps
    plot_heatmap(np.array(heatmap_real.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton)
    plot_heatmap(np.array(heatmap_generated.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton)

    # print scores
    print('score of real heatmap: ' + str(score_real.item()))
    print('score of generated heatmap: ' + str(score_generated.item()))
