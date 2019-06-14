import cv2 as cv
import numpy as np
import math
from shutil import copy2
from pycocotools.coco import COCO

# data path
dataDir = '/media/data/yzhang2/coco/train/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
annFileKeypoints = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations and keypoints annotations
coco = COCO(annFile)
cocoKeypoints = COCO(annFileKeypoints)

# get 'person' category id
personCatId = coco.getCatIds(catNms=['person'])

# get all images ids
imgIds = coco.getImgIds()

for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    imgSrc = '{}/images/{}'.format(dataDir, img['file_name'])
    annIds = coco.getAnnIds(imgIds=imgId, catIds=personCatId)

    # no person in the image
    if len(annIds) == 0:
        copy2(imgSrc, '{}/no_person/{}'.format(dataDir, img['file_name']))

    # only one person in the image
    elif len(annIds) == 1 and coco.loadAnns(annIds)[0].get('iscrowd') == 0:
        annId = annIds[0]
        ann = cocoKeypoints.loadAnns(annId)[0]

        # with enough keypoints
        if ann['num_keypoints'] > 5:
            copy2(imgSrc, '{}/one_person/{}'.format(dataDir, img['file_name']))

            # visualize
            stickwidth = 2
            colorSkeleton = [0, 255, 0]
            colorKeypoint = [0, 0, 255]

            canvas = cv.imread(imgSrc)
            cur_canvas = canvas.copy()
            sks = np.array(cocoKeypoints.loadCats(personCatId)[0]['skeleton']) - 1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]

            # visualize skeleton
            for sk in sks:
                if np.all(v[sk] > 0):
                    Y = x[sk]
                    X = y[sk]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    cv.fillConvexPoly(cur_canvas, polygon, colorSkeleton)

            # visualize keypoints
            for keypoint in zip(x, y, v):
                if keypoint[2] == 1:
                    cv.circle(cur_canvas, (keypoint[0], keypoint[1]), stickwidth * 2, colorKeypoint, 2)
                elif keypoint[2] == 2:
                    cv.circle(cur_canvas, (keypoint[0], keypoint[1]), stickwidth * 2, colorKeypoint, -1)

            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            cv.imwrite('{}/one_person_visualized/{}'.format(dataDir, img['file_name']), canvas)

        # without enough keypoints
        else:
            copy2(imgSrc, '{}/one_person_few_keypoints/{}'.format(dataDir, img['file_name']))

    # more than one persons in the image
    else:
        copy2(imgSrc, '{}/more_persons/{}'.format(dataDir, img['file_name']))
