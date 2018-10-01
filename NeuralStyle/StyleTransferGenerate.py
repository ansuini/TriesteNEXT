import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.misc
import datetime
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# ROOT folder
ROOT = '/home/ansuini/repos/TriesteNEXT/NeuralStyle'

# general modifiers
MULTISCALE = True
FRAME = False
PAINTER = False

# Make sure we all see the same
np.random.seed(456);
torch.manual_seed(456);

# Import libary
import style
from style.utils import gallery, animate_progress

# parse arguments
parser = argparse.ArgumentParser(description='Style Transfer Generate')
parser.add_argument('--style_img_name', default='', type=str, metavar='N', help='name of style image')
parser.add_argument('--content_img_name', default='', type=str, metavar='N', help='name of content image')
parser.add_argument('--style_weight', default=1.0, type=float, metavar='N', help='style weight')

args                = parser.parse_args()
style_img_name      = args.style_img_name
content_img_name    = args.content_img_name
alpha               = args.style_weight


bb = style.Backbone()
st = style.IteratedStyleTransfer(bb)

# Transfer Style

p = style.image.open(os.path.join(ROOT, 'photos_originals', content_img_name + '.jpg')).scale_long_to(512)
a = style.image.open(os.path.join(ROOT, 'styles', style_img_name + '.jpg')).scale_long_to(512)

# Get generator


if MULTISCALE:
    g = st.generate_multiscale(
        content=style.Content(p),
        style=style.GramStyle(a), 
        seed=p,
        niter=300,
        alpha=alpha)
else:
    g = st.generate(
        content=style.Content(p, lambda_loss=0),
        style=style.GramStyle(a, [4,8,12,14]),
        seed=p,
        alpha=alpha)


# Get next (final) result
x = next(g)
x = np.asarray(x)
width = x.shape[0]
heigth = x.shape[1]

# put frame 
if FRAME:
    offset = 40 # choose an even number
    cornice = np.zeros( ( width + 2*offset, heigth + 2*offset, 3) )
    cornice[offset:width+offset, offset:heigth+offset,:] = x
    cornice = cornice*255
else:
    cornice = x*255

# save as jpg with progressive number
index = 0
dirname = content_img_name
if not os.path.exists(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname)):
    os.mkdir(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname))
                                   
while os.path.exists(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname,  style_img_name + '_' + str(index) + '.jpg') ):
    index += 1

result_filename = os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname, style_img_name + '_' + str(index) +'.jpg') 
scipy.misc.toimage(cornice, cmin=0.0, cmax=255.0).save(result_filename)

if PAINTER:
    # add title of the painting ('Style : name of the painting')
    img = Image.open(result_filename)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype(os.path.join(ROOT,'Arimo-Italic.ttf'), 16 )
    # draw.text((x, y),"Sample Text",(r,g,b))
    y = cornice.shape[0]-(2/3)*offset
    x = cornice.shape[1]/2-offset
    
    parts = style_img_name.split('-')
    painter = parts[1]
    painter = painter[0].upper() + painter[1:]
    
    draw.text((x, y),painter,(255,255,255),font=font)
    img.save(result_filename)
    
    