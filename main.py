from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms.functional as TF
import fastText
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

generator_path = './generator.pt'
discriminator_path = './discriminator.pt'
image_folder = '/media/data/yzhang2/coco/train/coco/images/'
caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
keypoint_path = '/media/data/yzhang2/coco/train/coco/annotations/person_keypoints_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
batch_size = 64
total_keypoints = 17

# ground truth size in heatmap
sigma = 4

# size of heatmap input to network
image_size = 64

# resized heatmap size
resize_high = 135
resize_low = 75

# size of text encoding
sentence_vector_size = 300

# size of compressed text encoding
compress_size = 64

# text encoding interpolation
beta = 0.5

# number of channels of the largest convolution
convolution_channel = [32, 64, 128, 256]

noise_size = 64
g_input_size = noise_size + compress_size
d_final_size = 64
learning_rate = 0.001
epoch = 1

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# return ids of all training set's images containing one person
def get_one_person_image_ids(coco_keypoint):
    # get all 'person' image ids
    image_ids = coco_keypoint.getImgIds()

    image_ids_output = []

    for image_id in image_ids:

        # keypoint annotations of the image
        annotation_ids = coco_keypoint.getAnnIds(imgIds=image_id)

        # only one person in the image, with enough keypoints
        if len(annotation_ids) == 1 and coco_keypoint.loadAnns(annotation_ids)[0].get('iscrowd') == 0 and \
                coco_keypoint.loadAnns(annotation_ids)[0].get('num_keypoints') > 5:
            image_ids_output.append(image_id)

    return image_ids_output


# return random caption of given images
def get_one_random_caption(coco_caption, image_ids):
    # all caption ids
    caption_ids = coco_caption.getAnnIds(imgIds=image_ids)

    # choose a random caption
    caption = coco_caption.loadAnns(random.choice(caption_ids))[0]['caption'].replace('\n', '')

    return caption


# return ground truth heat map of a training image
def get_heatmap(coco_keypoint, image_id):
    # get image size and keypoint locations
    image = coco_keypoint.loadImgs(image_id)[0]
    keypoints = coco_keypoint.loadAnns(coco_keypoint.getAnnIds(imgIds=image_id))[0]

    # heatmap size is (number of keypoints)*(image height)*(image width)
    h = image['height']
    w = image['width']
    c = total_keypoints
    heatmap = np.zeros([c, h, w])
    x_grid = np.array([range(w)]).repeat(h, axis=0)
    y_grid = np.array([range(h)]).transpose().repeat(w, axis=1)

    # keypoints location (x, y) and visibility (v)
    x = keypoints['keypoints'][0::3]
    y = keypoints['keypoints'][1::3]
    v = keypoints['keypoints'][2::3]

    for i in range(c):

        # labeled keypoints' v > 0
        if v[i] > 0:
            # ground truth in heatmap is normal distribution shaped
            heatmap[i] = np.exp(-((x_grid - x[i]) ** 2 + (y_grid - y[i]) ** 2) / sigma ** 2)

    return heatmap


# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None):
    x_skeleton = np.empty((2, 0))
    y_skeleton = np.empty((2, 0))

    # option to plot skeleton
    if skeleton is not None:
        for line in skeleton:
            if np.all(np.any(heatmap[line, :, :], axis=(1, 2))):
                x_skeleton = np.hstack((x_skeleton, np.array([[0], [0]])))
                y_skeleton = np.hstack((y_skeleton, np.array([[0], [0]])))

                # keypoint is located in the maximum of the heatmap
                y_skeleton[0, -1], x_skeleton[0, -1] = np.unravel_index(np.argmax(heatmap[line[0], :, :], axis=None),
                                                                        heatmap.shape[1:3])
                y_skeleton[1, -1], x_skeleton[1, -1] = np.unravel_index(np.argmax(heatmap[line[1], :, :], axis=None),
                                                                        heatmap.shape[1:3])

    # get a heatmap in single image
    heatmap = np.max(heatmap, axis=0)
    plt.figure()

    # plot the heatmap in black-white and the optional training image
    if image_path is not None:
        image = io.imread(image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap, 'gray')
        plt.plot(x_skeleton, y_skeleton, 'white', linewidth=0.5)
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('training image')
    else:
        plt.imshow(heatmap, 'gray')
        plt.plot(x_skeleton, y_skeleton, 'white', linewidth=0.5)
        plt.title('stacked heatmaps' + (' and skeleton' if skeleton is not None else ''))
        plt.xlabel(caption)

    plt.show()


# return a randomly resized, cropped, flipped heatmap
def normalized_random_view_tensor(image):
    # set the heatmap as a sequence of PIL one-band images
    bands = [Image.fromarray(image[k, :, :]) for k in range(total_keypoints)]

    # resize
    resize = random.randint(resize_low, resize_high)
    bands = [TF.resize(bands[k], resize) for k in range(total_keypoints)]

    # crop
    w, h = bands[0].size
    i = random.randint(0, h - image_size)
    j = random.randint(0, w - image_size)
    bands = [TF.crop(bands[k], i, j, image_size, image_size) for k in range(total_keypoints)]

    # flip
    if random.random() < 0.5:
        bands = [TF.hflip(bands[k]) for k in range(total_keypoints)]

    bands = [TF.to_tensor(bands[k]) for k in range(total_keypoints)]

    # put the images into one 3D tensor, set range from [0,1] to [-1,-1]
    return (torch.cat(tuple(bands), 0) - 0.5) * 2


# from a batch of image ids, get a batch of heatmaps and matching caption sentence vectors
def get_random_view_and_caption_tensor(coco_keypoint, coco_caption, text_model, image_ids):
    image_tensor = torch.empty((0, total_keypoints, image_size, image_size))
    sentence_vector_tensor = torch.empty((0, sentence_vector_size, 1, 1))

    for image_id in image_ids:
        image = get_heatmap(coco_keypoint, image_id.item())

        # heatmaps
        image_tensor = torch.cat(
            (image_tensor, normalized_random_view_tensor(image).view(1, total_keypoints, image_size, image_size)), 0)

        # randomly select from all matching captions of the image
        caption = random.choice(coco_caption.loadAnns(coco_caption.getAnnIds(image_id.item())))['caption'].replace('\n',
                                                                                                                   '')

        # caption sentence vector
        sentence_vector_tensor \
            = torch.cat((sentence_vector_tensor,
                         torch.tensor(text_model.get_sentence_vector(caption)).view(1, sentence_vector_size, 1, 1)), 0)

    return image_tensor, sentence_vector_tensor


# get a batch of random interpolated caption sentence vectors from all captions of the batch of image ids
def get_interpolated_caption_tensor(coco_caption, text_model, image_ids):
    sentence_vector_tensor = torch.empty((0, sentence_vector_size, 1, 1))

    # all caption ids of the image batch
    caption_ids = coco_caption.getAnnIds(image_ids.tolist())

    for k in range(len(image_ids)):
        # randomly select 2 captions
        annotations = coco_caption.loadAnns(random.sample(caption_ids, 2))

        # interpolate caption sentence vectors
        sentence_vectors = [text_model.get_sentence_vector(x['caption'].replace('\n', '')) for x in annotations]
        interpolated_sentence_vectors = [beta * a + (1 - beta) * b for a, b in list(zip(*sentence_vectors))]

        # interpolated sentence vector
        sentence_vector_tensor = torch.cat(
            (sentence_vector_tensor, torch.tensor(interpolated_sentence_vectors).view(1, sentence_vector_size, 1, 1)),
            0)

    return sentence_vector_tensor


# get a batch of random caption sentence vectors from all captions
def get_random_caption_tensor(coco_caption, text_model, image_ids):
    sentence_vector_tensor = torch.empty((0, sentence_vector_size, 1, 1))

    # all caption ids
    caption_ids = coco_caption.getAnnIds()

    for k in range(len(image_ids)):
        # randomly select from all captions
        caption = coco_caption.loadAnns(random.choice(caption_ids))[0]['caption'].replace('\n', '')

        # caption sentence vector
        sentence_vector_tensor \
            = torch.cat((sentence_vector_tensor,
                         torch.tensor(text_model.get_sentence_vector(caption)).view(1, sentence_vector_size, 1, 1)), 0)

    return sentence_vector_tensor


# get a batch of noise vectors
def get_noise_tensor(image_ids):
    noise_tensor = torch.empty((0, noise_size, 1, 1))

    for k in range(len(image_ids)):
        noise_tensor = torch.cat((noise_tensor, torch.randn(1, noise_size, 1, 1)), 0)

    return noise_tensor


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
            nn.Tanh()

        )

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise_vector, sentence_vector):
        # concatenate noise vector and compressed sentence vector
        input_vector = torch.cat((noise_vector, (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1))), 1)

        return self.main(input_vector)


# discriminator given heatmap and sentence vector
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # several layers of convolution, batch normalization and leaky ReLu
        self.main = nn.Sequential(
            nn.Conv2d(total_keypoints, convolution_channel[0], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[0]),
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

        )

        # compute final score of the discriminator with concatenated sentence vector
        self.second = nn.Sequential(
            nn.Conv2d(convolution_channel[3] + compress_size, d_final_size, 1, bias=False),
            nn.ReLU(d_final_size),
            nn.Conv2d(d_final_size, 1, 4, bias=False),
            nn.Sigmoid()
        )

        # compress text encoding first
        self.compress = nn.Sequential(
            nn.Linear(sentence_vector_size, compress_size),
            nn.ReLU(True)
        )

    def forward(self, input_heamap, sentence_vector):
        # first transposed convolution, then sentence vector
        tensor = torch.cat((self.main(input_heamap), (
            (self.compress(sentence_vector.view(-1, sentence_vector_size))).view(-1, compress_size, 1, 1)).repeat(1, 1,
                                                                                                                  4,
                                                                                                                  4)),
                           1)

        return self.second(tensor)


# using trained generator, give a caption, plot a generated heatmap
def plot_caption(caption, text_model, generator_path, device, skeleton=None):
    # load generator model
    net = Generator()
    net.load_state_dict(torch.load(generator_path))
    net.to(device)
    net.eval()

    # sentence and noise vector as input to the generator
    sentence_vector = torch.tensor(text_model.get_sentence_vector(caption), dtype=torch.float32,
                                   device=device).view(1, sentence_vector_size, 1, 1)
    noise_vector = torch.randn(1, noise_size, 1, 1, device=device)

    # generate
    with torch.no_grad():
        image = net(noise_vector, sentence_vector)

    # plot heatmap
    plot_heatmap(np.array(image.squeeze().tolist()), skeleton=skeleton, caption=caption)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
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

    net_G = Generator()
    net_D = Discriminator()
    net_G.to(device)
    net_D.to(device)
    net_G.apply(weights_init)
    net_D.apply(weights_init)
    optimizer_G = optim.Adam(net_G.parameters(), lr=learning_rate, betas=(first_momentum, second_momentum))
    optimizer_D = optim.Adam(net_D.parameters(), lr=learning_rate, betas=(first_momentum, second_momentum))

    # train
    print(datetime.now())
    print('training')
    net_G.train()
    net_D.train()

    for e in range(epoch):

        # number of batches
        batch = len(image_loader)
        for i, data in enumerate(image_loader, 0):
            # get heatmaps, sentence vectors and noises
            image_real, text_match = get_random_view_and_caption_tensor(coco_keypoint, coco_caption, text_model, data)
            text_mismatch = get_random_caption_tensor(coco_caption, text_model, data)
            text_interpolated = get_interpolated_caption_tensor(coco_caption, text_model, data)
            noise = get_noise_tensor(data)
            noise2 = get_noise_tensor(data)

            image_real = image_real.to(device)
            text_match = text_match.to(device)
            text_mismatch = text_mismatch.to(device)
            text_interpolated = text_interpolated.to(device)
            noise = noise.to(device)
            noise2 = noise2.to(device)

            # optimize
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # generate heatmaps
            image_fake = net_G(noise, text_match)
            image_interpolated = net_G(noise2, text_interpolated)

            # discriminate heatmpap-text pairs
            score_right = net_D(image_real, text_match)
            score_wrong = net_D(image_real, text_mismatch)
            score_fake = net_D(image_fake.detach(), text_match)
            score_fake_to_back = net_D(image_fake, text_match)
            score_interpolated = net_D(image_interpolated, text_interpolated)

            # calculate losses
            loss_D = -torch.mean(torch.log(score_right) + (torch.log(1 - score_wrong) + torch.log(1 - score_fake)) / 2)
            loss_G = -torch.mean(torch.log(score_fake_to_back) + torch.log(score_interpolated))

            # update
            loss_G.backward()
            optimizer_G.step()
            loss_D.backward()
            optimizer_D.step()

            # print progress
            print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ', batch ' + str(i + 1) + ' of ' + str(
                batch) + '. loss_G = ' + str(loss_G.item()) + ', loss_D = ' + str(loss_D.item()))

    print('\nfinished')
    print(datetime.now())

    # save models
    torch.save(net_G.state_dict(), generator_path)
    torch.save(net_D.state_dict(), discriminator_path)
