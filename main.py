from pycocotools.coco import COCO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import fastText
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage

generator_path = './generator.pt'
discriminator_path = './discriminator.pt'
image_folder = '/media/data/yzhang2/coco/train/coco/images/'
caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
keypoint_path = '/media/data/yzhang2/coco/train/coco/annotations/person_keypoints_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'
batch_size = 128
total_keypoints = 17

# ground truth size in heatmap
sigma = 5

# size of heatmap input to network
heatmap_size = 128

# resized heatmap size
resize_high = 270
resize_low = 150

# size of text encoding
sentence_vector_size = 300

# size of compressed text encoding
compress_size = 256

# text encoding interpolation
beta = 0.5

# number of channels of the largest convolution
convolution_channel = [32, 64, 128, 256, 512]

noise_size = 128
g_input_size = noise_size + compress_size
d_final_size = 256

# update discriminator how many iterations before updating generator
k = 1

learning_rate_g = 0.0002
learning_rate_d = 0.0002
epoch = 1

# ADAM solver
first_momentum = 0.5
second_momentum = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# big 2d normal distribution
center = 2000
size = 2 * center + 1
grid = np.array([range(size)]).repeat(size, axis=0)
normal = np.exp(-((grid - center) ** 2 + (grid.transpose() - center) ** 2) / sigma ** 2, dtype='float32')
zero = np.zeros([size, size], dtype='float32')


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


# return random id and caption of given images
def get_one_random_image_id_with_caption(coco_caption, image_ids):
    # a random image id
    image_id = random.choice(image_ids)
    # all caption ids of this image
    caption_ids = coco_caption.getAnnIds(imgIds=image_id)

    # choose a random caption
    caption = random.choice(coco_caption.loadAnns(caption_ids))['caption'].replace('\n', '')

    return image_id, caption


# return ground truth heat map of a training image
def get_heatmap(coco_keypoint, image_id):
    # get image size and keypoint locations
    image = coco_keypoint.loadImgs(image_id)[0]
    keypoints = coco_keypoint.loadAnns(coco_keypoint.getAnnIds(imgIds=image_id))[0]

    # heatmap size is (number of keypoints)*(image height)*(image width)
    h = image['height']
    w = image['width']
    c = total_keypoints
    heatmap = []

    # keypoints location (x, y) and visibility (v)
    x = keypoints['keypoints'][0::3]
    y = keypoints['keypoints'][1::3]
    v = keypoints['keypoints'][2::3]

    for i in range(c):

        # labeled keypoints' v > 0
        if v[i] > 0:
            # # ground truth in heatmap is normal distribution shaped
            heatmap.append(normal[(center - y[i]):(center - y[i] + h), (center - x[i]):(center - x[i] + w)])
        else:
            heatmap.append(zero[0:h, 0:w])

    return np.array(heatmap)


# plot a heatmap
def plot_heatmap(heatmap, skeleton=None, image_path=None, caption=None):
    x_skeleton = np.empty((2, 0))
    y_skeleton = np.empty((2, 0))

    # option to plot skeleton
    if skeleton is not None:
        for line in skeleton:
            if np.any(heatmap[line[0]]) and np.any(heatmap[line[1]]):
                x_skeleton = np.hstack((x_skeleton, np.array([[0], [0]])))
                y_skeleton = np.hstack((y_skeleton, np.array([[0], [0]])))

                # keypoint is located in the maximum of the heatmap
                y_skeleton[0, -1], x_skeleton[0, -1] = np.unravel_index(np.argmax(heatmap[line[0]], axis=None),
                                                                        heatmap[line[0]].shape)
                y_skeleton[1, -1], x_skeleton[1, -1] = np.unravel_index(np.argmax(heatmap[line[1]], axis=None),
                                                                        heatmap[line[1]].shape)

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
def normalized_random_view_tensor(heatmap):
    # resize
    resize = random.randint(resize_low, resize_high)
    zoom = resize / min(heatmap[0].shape)
    heatmap = [ndimage.zoom(heatmap[k], zoom, order=1) for k in range(total_keypoints)]

    # crop, set range from [0,1] to [-1,-1]
    h, w = heatmap[0].shape
    i = random.randint(0, h - heatmap_size)
    j = random.randint(0, w - heatmap_size)
    heatmap = [(heatmap[k][i:(i + heatmap_size), j:(j + heatmap_size)] - 0.5) * 2 for k in range(total_keypoints)]

    # flip
    if random.random() < 0.5:
        heatmap = [np.fliplr(heatmap[k]) for k in range(total_keypoints)]

    # put the images into one 3D tensor
    return torch.tensor(heatmap, dtype=torch.float32)


# from a batch of image ids, get a batch of heatmaps and matching caption sentence vectors
def get_random_view_and_caption_tensor(coco_keypoint, coco_caption, text_model, image_ids):
    heatmap_tensor = torch.empty((0, total_keypoints, heatmap_size, heatmap_size))
    sentence_vector_tensor = torch.empty((0, sentence_vector_size, 1, 1))

    for image_id in image_ids:
        heatmap = get_heatmap(coco_keypoint, image_id.item())

        # heatmaps
        heatmap_tensor = torch.cat((heatmap_tensor,
                                    normalized_random_view_tensor(heatmap).view(1, total_keypoints, heatmap_size,
                                                                                heatmap_size)), 0)

        # randomly select from all matching captions of the image
        caption = random.choice(coco_caption.loadAnns(coco_caption.getAnnIds(image_id.item())))['caption'].replace('\n',
                                                                                                                   '')

        # caption sentence vector
        sentence_vector_tensor = torch.cat((sentence_vector_tensor,
                                            torch.tensor(text_model.get_sentence_vector(caption)).view(1,
                                                                                                       sentence_vector_size,
                                                                                                       1, 1)), 0)

    return heatmap_tensor, sentence_vector_tensor


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
        sentence_vector_tensor = torch.cat((sentence_vector_tensor,
                                            torch.tensor(text_model.get_sentence_vector(caption)).view(1,
                                                                                                       sentence_vector_size,
                                                                                                       1, 1)), 0)

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

            nn.ConvTranspose2d(convolution_channel[-4], convolution_channel[-5], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[-5]),
            nn.ReLU(True),

            nn.ConvTranspose2d(convolution_channel[-5], total_keypoints, 4, 2, 1, bias=False),
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

            nn.Conv2d(convolution_channel[3], convolution_channel[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(convolution_channel[4]),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # compute final score of the discriminator with concatenated sentence vector
        self.second = nn.Sequential(
            nn.Conv2d(convolution_channel[4] + compress_size, d_final_size, 1, bias=False),
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
        image = net_g(noise_vector, sentence_vector)

    # plot heatmap
    plot_heatmap(np.array(image.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)


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
    heatmap_real = torch.tensor(heatmap, dtype=torch.float32, device=device).view(1, total_keypoints, heatmap_size,
                                                                                heatmap_size)

    # generate and discriminate
    with torch.no_grad():
        heatmap_generated = net_g(noise_vector, sentence_vector)
        score_real = net_d(heatmap_real, sentence_vector)
        score_generated = net_d(heatmap_generated, sentence_vector)

    # plot heatmaps
    plot_heatmap(np.array(heatmap_real.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)
    plot_heatmap(np.array(heatmap_generated.squeeze().tolist()) * 0.5 + 0.5, skeleton=skeleton, caption=caption)

    # print scores
    print('score of real heatmap: ' + str(score_real.item()))
    print('score of generated heatmap: ' + str(score_generated.item()))


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

    net_g = Generator()
    net_d = Discriminator()
    net_g.to(device)
    net_d.to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)
    optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate_g, betas=(first_momentum, second_momentum))
    optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate_d, betas=(first_momentum, second_momentum))

    # train
    print(datetime.now())
    print('training')
    net_g.train()
    net_d.train()

    for e in range(epoch):

        # number of batches
        batch = len(image_loader)
        for i, data in enumerate(image_loader, 0):
            # get heatmaps, sentence vectors and noises
            heatmap_real, text_match = get_random_view_and_caption_tensor(coco_keypoint, coco_caption, text_model, data)
            heatmap_real = heatmap_real.to(device)
            text_match = text_match.to(device)
            for j in range(k):
                text_mismatch = get_random_caption_tensor(coco_caption, text_model, data)
                noise = get_noise_tensor(data)

                text_mismatch = text_mismatch.to(device)
                noise = noise.to(device)

                # optimize
                net_d.zero_grad()

                # generate heatmaps
                heatmap_fake = net_g(noise, text_match)

                # discriminate heatmpap-text pairs
                score_right = net_d(heatmap_real, text_match)
                score_wrong = net_d(heatmap_real, text_mismatch)
                score_fake = net_d(heatmap_fake.detach(), text_match)

                # calculate losses
                loss_d = -torch.mean(
                    torch.log(score_right) + (torch.log(1 - score_wrong) + torch.log(1 - score_fake)) * 0.5)

                # update
                loss_d.backward()
                optimizer_d.step()

                # print progress
                print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
                    batch) + ' score_right: ' + str(score_right.mean().item()) + ' score_wrong: ' + str(
                    score_wrong.mean().item()) + ' score_fake: ' + str(score_fake.mean().item()))

            # get heatmaps, sentence vectors and noises
            text_mismatch = get_random_caption_tensor(coco_caption, text_model, data)
            text_interpolated = get_interpolated_caption_tensor(coco_caption, text_model, data)
            noise = get_noise_tensor(data)
            noise2 = get_noise_tensor(data)

            text_mismatch = text_mismatch.to(device)
            text_interpolated = text_interpolated.to(device)
            noise = noise.to(device)
            noise2 = noise2.to(device)

            # optimize
            net_g.zero_grad()

            # generate heatmaps
            heatmap_fake = net_g(noise, text_match)
            heatmap_interpolated = net_g(noise2, text_interpolated)

            # discriminate heatmpap-text pairs
            score_fake = net_d(heatmap_fake, text_match)
            score_interpolated = net_d(heatmap_interpolated, text_interpolated)

            # calculate losses
            loss_g = -torch.mean(torch.log(score_fake) + torch.log(score_interpolated))

            # update
            loss_g.backward()
            optimizer_g.step()

            # print progress
            print('epoch ' + str(e + 1) + ' of ' + str(epoch) + ' batch ' + str(i + 1) + ' of ' + str(
                batch) + ' score_right: ' + str(score_right.mean().item()) + ' score_wrong: ' + str(
                score_wrong.mean().item()) + ' score_fake: ' + str(
                score_fake.mean().item()) + ' score_interpolated: ' + str(score_interpolated.mean().item()))

    print('\nfinished')
    print(datetime.now())

    # save models
    torch.save(net_g.state_dict(), generator_path)
    torch.save(net_d.state_dict(), discriminator_path)
