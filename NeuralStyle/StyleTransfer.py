'''
This code is a modified version of the original code by
Alexis Jacq available in the official PyTorch documentation at :

https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import scipy.misc
from PIL import Image
import datetime

ROOT = '/home/ansuini/repos/TriesteNEXT/NeuralStyle'

# parameters settings
parser = argparse.ArgumentParser(description='Style Transfer')
parser.add_argument('--num_steps', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--style_weight', default=1e6, type=float, metavar='N', help='weight of the style loss term in the total loss')
parser.add_argument('--content_weight', default=1, type=int, metavar='N', help='weight of the content loss term in the total loss')
parser.add_argument('--style_img_name', default='', type=str, metavar='N', help='name of style image')
parser.add_argument('--content_img_name', default='', type=str, metavar='N', help='name of content image')

args                = parser.parse_args()
num_steps           = args.num_steps
style_weight        = args.style_weight
content_weight      = args.content_weight
style_img_name      = args.style_img_name
content_img_name    = args.content_img_name

# desired depth layers to compute style/content losses
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# device and image size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize,Image.LANCZOS),  # scale imported image
    transforms.ToTensor()])     # transform it into a torch tensor

def crop(image): 
    '''
    Cropping function : the image has to be squared in order to be
    forwarded to the network
    '''
    w,h = image.size
    print('Original image size : Width {} Height {}'.format(  w, h ) )
    if w == h:
        cropped = image
    if w > h:
        #print('w > h')
        offset = int( ( w - h )/2 )
        #print('Offset : {}'.format( offset ) )
        cropped = image.crop((offset, 0, w - offset, h))
        #print('Size cropped image : {}'.format( cropped.size ) )
    elif w < h:
        #print('w < h')
        offset = int( ( h - w )/2 )
        #print('Offset : {}'.format( offset ) )
        cropped = image.crop((0,  offset, w, h - offset))

    print('Cropped image : {}'.format( cropped.size ) )              
    return cropped

                             
def image_loader(image_name):
    '''
    Utility function to load, crop and remove singleton dimension
    '''                                                                                  
    image = Image.open(image_name)
    #print(image.size)
    image = crop(image)
    #print('Size image after cropping : {}'.format(image.size()) )
    image = loader(image).unsqueeze(0)
    print('Size image after resizing : {}'.format(image.size()) )
    # final adjustment
    if image.size()[2] != image.size()[3]:
        minimum = np.min([ image.size()[2], image.size()[3] ] )
        image = image[:,:,0:minimum,0:minimum]
    return image.to(device, torch.float)


# load images
style_img   = image_loader(os.path.join(ROOT, 'styles', style_img_name + '.jpg') )
content_img = image_loader(os.path.join(ROOT, 'photos_originals', content_img_name + '.jpg') )


# vgg instantiation and normalization
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

            
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        #elif isinstance(layer, nn.MaxPool2d):
        #    name = 'pool_{}'.format(i)
            
            
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            
            layer = nn.AvgPool2d(kernel_size=layer.kernel_size, 
                                   stride=layer.stride, padding = layer.padding)
         
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


                             
input_img = content_img.clone()
# if you want to use a white noise instead uncomment the below line:
#input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
#plt.figure()
#imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=10000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model ...\n')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing ...\n')
    history = []
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            history.append(loss.item())
            
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, history



#-------------------------------------- main script --------------------------------------------------

os.system('clear')
from time import time
print('******************************************************')
print('NEURAL STYLE TRANSFER IS STARTING, PLEASE WAIT ...')
print('')
tic = time()
output, history = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,  num_steps=num_steps, 
                            style_weight=style_weight, content_weight=content_weight)

print('Elapsed time is : {}'.format(time() - tic))
print('')
print('SUCCESS !')
print('******************************************************')

                             
# put frame and save as jpg
offset = 40 # choose an even number
image_size = list(output.size())
image_size[2] += 2*offset
image_size[3] += 2*offset
cornice = torch.zeros(image_size)
cornice[:,:, offset:imsize+offset, offset:imsize+offset] = output
out = cornice.detach().cpu().numpy()
out = out[0]
out *= 255



index = 0
dirname = content_img_name
if not os.path.exists(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname)):
    os.mkdir(os.path.join(ROOT, 'results',datetime.date.today().strftime("%d"), dirname))
                                   
while os.path.exists(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname,  style_img_name + '_' + str(index) + '.jpg') ):
    index += 1
scipy.misc.toimage(out, cmin=0.0, cmax=255.0).save(os.path.join(ROOT, 'results', datetime.date.today().strftime("%d"), dirname, style_img_name + '_' + str(index) +'.jpg') )

# save current result in results folder (this is always overwritten)
#scipy.misc.toimage(out, cmin=0.0, cmax=255.0).save(os.path.join(ROOT, 'results', 'current.jpg') )
