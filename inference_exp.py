
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import json
import imageio
import coco
import matplotlib
import matplotlib.pyplot as plt
import moviepy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import moviepy.editor as mpy
from tqdm import tqdm

from glob import glob
from pathlib import Path
import cv2
from skimage.measure import find_contours
#from youtube_dl import YoutubeDL
#import imageio
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    #print(image.shape,'image')
    #print(mask.shape,'mask')
    #print(len(color),'color')
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def filter_objects(frame_stat, c_names, s_names):
    #res = {'rois':np.array([]), 'class_ids':np.array([]), 'scores':np.array([]), 'masks':np.array([])}
    rois=[]
    class_ids=[]
    scores=[]
    masks=[]
    for i, cid in enumerate(frame_stat['class_ids']):
        if c_names[cid] in s_names:
            rois.append(frame_stat['rois'][i])
            class_ids.append(frame_stat['class_ids'][i])
            scores.append(frame_stat['scores'][i])
            masks.append(frame_stat['masks'][:,:,i])
    #cur_shape_masks = np.array(masks).shape
    #masks = np.array(masks).reshape(cur_shape_masks[1],cur_shape_masks[2],cur_shape_masks[0])
    return {'rois':np.array(rois), 'class_ids':np.array(class_ids), 'scores':np.array(scores), 'masks':np.array(masks)}

def get_display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    #if not N:
       #print("\n*** No instances to display *** \n")
    #else:
        #assert boxes.shape[0] == masks.shape[0] == class_ids.shape[0],'Custom Try_except'
    '''
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
    '''
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    
    fig = Figure(figsize=((width/100),(height/100)), dpi=100.0, frameon=False, tight_layout=False)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    
    # Generate random colors
    #colors = random_colors(N)
    colors=[]
    for i in range(N):
        object_color = []
        object_color=list(map(lambda i:round(i,3),list(np.random.choice(range(256), size=3)/256)))
        colors.append(object_color)
    #ax.set_ylim(height + 10, -10)
    #ax.set_xlim(-10, width + 10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    #ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    #print(masked_image,masked_image.shape,'masked_image+shape')
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:,:,i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    
    ax.imshow(masked_image.astype(np.uint8))
    canvas.draw()       # draw the canvas, cache the renderer

    ret_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    #ncols, nrows = fig.canvas.get_width_height()
    
    
    ret_image = ret_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return ret_image

def masked_places_video_frames(video, root_dir, pmodel, frame_skip = 1):
        vcolors={'fake': (255,0,0),
            'original':(36,255,12),
            'vien': (0,255,0)}
        print('Started')
        capture = cv2.VideoCapture(video)
        
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        heigth = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = (capture.get(cv2.CAP_PROP_FOURCC))
        duration = frames_num/fps
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        id = os.path.splitext(os.path.basename(video))[0]
        
        os.makedirs(os.path.join(root_dir, 'output'), exist_ok=True)
        
        frames=[]
        frames_batch=[]
        video_stat=[]
        for i in tqdm(range(0, frames_num, frame_skip)):
            #clear_output()
            print("%s / %s" % (i, frames_num))
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success,frame = capture.read()
            
            if not success:
                print('Error in reading frame %s!!!' %i)
                continue
            
            frame_bboxes = frame.copy()
            print('Frame skip before!!!', i, frame_skip)
            if (i % frame_skip == 0):
                print('Frame skip!!!', i, frame_skip)
                frames_batch.append(frame_bboxes)
                #probs, bboxes =.detect(Image.fromarray(frame_bboxes), pmodel, ptransform),
                # Run detection,
            if len(frames_batch) == config.BATCH_SIZE:
                print('Batch processed!!!!')
                results = pmodel.detect(frames_batch, verbose=1)
                #print(results)
                for j, res in enumerate(results):
		    #res_selected = res 
                    #res_selected = filter_objects(res, class_names, selected_names)
                    #print(res,'res')
                    #print(res['rois'].shape,'rois')
                    #print(res['masks'].shape,'masks')
                    #print(res['class_ids'].shape,'class_ids')
                    #print(frames_batch[j].shape,'frames_batch')
                    #frames_batch[j] = get_display_instances(image = frames_batch[j], boxes = res[0]['rois'], masks = res[0]['masks'], class_ids = res[0]['class_ids'], class_names = class_names)
                    #frames_batch[j] = visualize.display_instances(image = frames_batch[j], boxes = res_selected[0]['rois'], masks = res_selected[0]['masks'], class_ids = res_selected[0]['class_ids'], class_names = class_names)
                    frames_batch[j] = get_display_instances(image = frames_batch[j], boxes = res['rois'], masks = res['masks'], class_ids = res['class_ids'], class_names = class_names)
                    #print('Correct')
                    #frames_batch[j] = cv2.putText(frames_batch[j], 'Processed by VIEN VPP service', (0, heigth-10), cv2.FONT_HERSHEY_PLAIN, 0.5, vcolors['vien'], 1)
                    frames_batch[j] = cv2.cvtColor(frames_batch[j], cv2.COLOR_BGR2RGB)
                    frames.append(frames_batch[j])
		    #print(frames_batch[j])
                    PolygonVerts = []
                    for mask in res['masks']:
                        # Mask Polygon
                        # Pad to ensure proper polygons for masks that touch image edges.
                        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                        padded_mask[1:-1, 1:-1] = mask
                        contours = find_contours(padded_mask, 0.5)
                        for verts in contours:
                            # Subtract the padding and flip (y, x) to (x, y)
                            verts = np.fliplr(verts) - 1
                       	    PolygonVerts.append(np.array(verts, np.int32))
                    if len(res['class_ids'])>0:
                        video_stat.append({'Timestamp':(i-8+j), 
                                           'Name': [class_names[c] for c in res['class_ids']], 
                                           'Confidence': res['scores'],
                                           'BoundingBoxes': res['rois'],
                                           'PolygonVerts': PolygonVerts
                                          })
                frames_batch=[]
        capture.release()
        #print(frames)
        if len(frames) >0:
            imageio.mimsave(os.path.join(root_dir, 'output', '{}.mp4'.format(id)), frames, fps=fps)
        else:
            print('Frames is empty (probably frames number less than BATCH_SIZE)!!!')
        return [os.path.join(root_dir, 'output', '{}.mp4'.format(id)), video_stat]



# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_EXP_PATH = os.path.join(MODEL_DIR, "mask_rcnn_experiment_bag_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

class CustomConfig(coco.CocoConfig):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """

    GPU_COUNT = 1
    BATCH_SIZE = 16 #16
    IMAGES_PER_GPU = 16 #16
    NUM_CLASSES = 3  # Background + 4
    # Number of training steps per epoch
    #STEPS_PER_EPOCH = 20
    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

config = CustomConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", config=config,model_dir=MODEL_DIR)
model.load_weights(COCO_EXP_PATH, by_name=True)


#video_dir = 'office 480p cut_orginal.avi'
#root_dir = ''

#t1 = 50
#t2 = 60
#ffmpeg_extract_subclip(video_dir, t1, t2, targetname="test.mp4")
#display(mpy.ipython_display("test.mp4", height=400 , autoplay=1, loop=1, maxduration=100))
class_names = ['BG','backpack', 'handbag']
selected_names = ['backpack', 'handbag']


##############################################################################################3


import numpy as np
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def display_instances_exp(image, boxes, masks, class_ids, class_names,
                  scores=None, title="",
                  figsize=(16, 16), ax=None,
                  show_mask=True, show_bbox=True,
                  colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        #auto_show = True

    # Generate random colors
    #colors = colors or random_colors(N)
    colors=[]
    for i in range(N):
    	object_color = []
    	object_color=list(map(lambda i:round(i,3),list(np.random.choice(range(256), size=3)/256)))
    	object_color.append(0.8)
    	colors.append(object_color)
    #for i in range(N):
   	#colors.append(list(map(lambda i:round(i,3),list(np.random.choice(range(256), size=4)/256))))
    # colors = list(map(lambda i:round(i,1),list(np.random.choice(range(256), size=3)/256)))
    #colors = [[0.9,0,0,0.8], [0,0.9,0,0.8], [0,0,0.9,0.8]]
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            print(':(')
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
            color='r', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = plt.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
    plt.savefig('output_fig.png')
    #ax.figsave('./yours.png', masked_image.astype(np.uint8))
    img1 = Image.fromarray(masked_image.astype(np.uint8), 'RGB')
    img1.save('my.png')

video_dir = "./test_vid_bag.mp4"

from skimage import io
images = [io.imread('./test_2.jpg')] #io.imread('https://img3.goodfon.ru/original/1920x1200/0/cc/vino-beloe-bokal-butylka-stol.jpg')]
#print(images[0])
#ax = get_ax(1)
#res = model.detect(images, verbose=1)
#print(res)
#display_instances_exp(images[0], res[0]['rois'], res[0]['masks'], res[0]['class_ids'], class_names, res[0]['scores'],ax=ax)
#plt.savefig('output_fig.png')
#print(img)
#io.imsave(os.path.join('', res.jpg), img[0])
#display_instances_exp(images[0], res[0]['rois'], res[0]['masks'], res[0]['class_ids'], class_names, res[0]['scores'],ax=ax)



output = masked_places_video_frames(video_dir, '.', model, 2)
dumped = json.dumps(output[1], cls=NumpyEncoder)

with open('data_480p_bags.json', 'w') as fp:
    #json.dump(dumped, fp)
    print(dumped, file=fp)



