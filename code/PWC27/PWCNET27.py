import xml.dom.minidom
import script_pwc as pwc
import math
import numpy as np
import sys
import cv2
import torch
from math import ceil
from scipy.ndimage import imread
import models
import os
import io
import xml.dom.minidom
import errno
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont

def get_box_path(boxpath):
    box = os.listdir(boxpath)
    box.sort(key=lambda x: int(x.replace(' ', '').split('.')[0]))

    boxs = []
    for count in range(len(box)):
        im_name = box[count]
        im_path = os.path.join(boxpath, im_name)
        boxs.append(im_path)

    return boxs

def _color_wheel():
    # Original inspiration: http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # RGB

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    # YG
    colorwheel[col: YG + col, 0] = 255 - \
        np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col: YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col: GC + col, 1] = 255
    colorwheel[col: GC + col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC

    # CB
    colorwheel[col: CB + col, 1] = 255 - \
        np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col: CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col: BM + col, 2] = 255
    colorwheel[col: BM + col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM

    # MR
    colorwheel[col: MR + col, 2] = 255 - \
        np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col: MR + col, 0] = 255

    return colorwheel

def _compute_color(u, v):
    colorwheel = _color_wheel()
    idxNans = np.where(np.logical_or(
        np.isnan(u),
        np.isnan(v)
    ))
    u[idxNans] = 0
    v[idxNans] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1)
    k0 = fk.astype(np.uint8)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]

    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1-f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        col[~idx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(np.uint8)  # RGB
        # img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8) # BGR

    return img.astype(np.uint8)

def convert_from_file(path,u,v, mode='RGB'):
    return convert_from_flow(u,v, mode)

def convert_from_flow(u,v, mode='RGB'):
    if mode == 'RGB':
        img = _compute_color(u, v)
        return img
    if mode == 'UV':
        uv = (np.dstack([u, v]) * 127.999 + 128).astype('uint8')
        return uv

    img = _compute_color(u, v)
    return img

def convert_files(files, u,v,outdir=None):

    if outdir != None and not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
            print("> Created directory: " + outdir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    print(files)
    t = tqdm(files,ascii=True,desc='flow')

    for f in t:
        image = convert_from_file(f,u,v)

        if outdir == None:
            path = f + '.png'
            t.set_description(path)
            _save_png(image, path)
        else:
            path = os.path.join((outdir[:-4]) + '.png')
            # t.set_description(path)
            _save_png(image, path)
    return path

def _save_png(arr, path):
    # TO-DO: No dependency
    Image.fromarray(arr).save(path)

def boxsize(box_path):
    dom = xml.dom.minidom.parse(box_path)
    root = dom.documentElement
    xmi = dom.getElementsByTagName('xmin')
    xma = dom.getElementsByTagName('xmax')
    ymi = dom.getElementsByTagName('ymin')
    yma = dom.getElementsByTagName('ymax')  # DOM Element: xmin at 0x7fb22c451470
    xmin, xmax, ymin, ymax, size = [], [], [], [], []
    for i in range(len(xmi)):
        xmin.append(int(xmi[i].firstChild.data))
        ymin.append(int(ymi[i].firstChild.data))
        ymax.append(int(yma[i].firstChild.data))
        xmax.append(int(xma[i].firstChild.data))
        size.append((int(yma[i].firstChild.data) - int(ymi[i].firstChild.data)) * (
                    int(xma[i].firstChild.data) - int(xmi[i].firstChild.data)))
    return xmin, xmax, ymin, ymax, size


def get_opticaluv(xml_path, xmllen,picfile_path):
   
    xmin, xmax, ymin, ymax, bsize = boxsize(xml_path)
   
    img_path1 = os.path.join(picfile_path, xml_path[xmllen:-4] + '.jpg')

    img_path2 = os.path.join(picfile_path, str(int(xml_path[j][xmllen:-4])+3) + '.jpg')
    print('img1',img_path1)
    print('img2', img_path2)
    # '''optical'''
    flow_fn = os.path.join('/home/show/flo', xml_path[xmllen:-4]  + '.flo')
    print('flo', flow_fn)

    TAG_FLOAT = 202021.25
    flags = {
        'debug': False
    }

    if len(sys.argv) > 1:
        img_path1 = sys.argv[1]
    if len(sys.argv) > 2:
        img_path2 = sys.argv[2]
    if len(sys.argv) > 3:
        flow_fn = sys.argv[3]

    pwc_model_fn = './pwc_net.pth.tar';

    im_all = [imread(img) for img in [img_path1, img_path2]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)
    for c in range(len(im_all)):
        im_all[c] = cv2.resize(im_all[c], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i] / 255.0

        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        im_all[_i] = torch.from_numpy(im_all[_i])
        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
        im_all[_i] = im_all[_i].float()

    im_all = torch.autograd.Variable(torch.cat(im_all, 1).cuda(), volatile=True)

    net = models.pwc_dc_net(pwc_model_fn)
    net = net.cuda()
    net.eval()

    flo = net(im_all)
    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)  #
    u_ = cv2.resize(flo[:, :, 0], (W, H))
    v_ = cv2.resize(flo[:, :, 1], (W, H))
    u_ *= W / float(W_)
    v_ *= H / float(H_)
    flo = np.dstack((u_, v_))
    flo_path = pwc.writeFlowFile(flow_fn, flo)

    if not isinstance(flo_path, io.BufferedReader):
        if not isinstance(flo_path, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=flo_path))
        if not os.path.isfile(flo_path):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=flo_path))
        if not flo_path.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=flo_path.split('.')[-1]))

        flo = open(flo_path, 'rb')
    else:
        flo = flo_path

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]

    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    nbands = 2
    tmp = np.frombuffer(flo.read(nbands * width * height * 4),
                        np.float32, count=nbands * width * height)
    flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    flo.close()

    UNKNOWN_FLOW_THRESH = 1e9

    height, width, nBands = flow.shape

    if not nBands == 2:
        raise AssertionError("Image must have two bands. [{h},{w},{nb}] shape given instead".format(
            h=height, w=width, nb=nBands))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # Fix unknown flow
    idxUnknown = np.where(np.logical_or(
        abs(u) > UNKNOWN_FLOW_THRESH,
        abs(v) > UNKNOWN_FLOW_THRESH
    ))
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max([-999, np.max(u)])
    maxv = max([-999, np.max(v)])
    minu = max([999, np.min(u)])
    minv = max([999, np.min(v)])

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([-1, np.max(rad)])

    if flags['debug']:
        print(
            "Max Flow : {maxrad:.4f}. Flow Range [u, v] -> [{minu:.3f}:{maxu:.3f}, {minv:.3f}:{maxv:.3f}] ".format(
                minu=minu, minv=minv, maxu=maxu, maxv=maxv, maxrad=maxrad
            ))

    eps = np.finfo(np.float32).eps
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    # print('u', u.shape)
    ##draw flow pic
    flos = flow_fn
    outdir = flow_fn
    print("> Rendering images [.png] from the flows [.flo]")
    # show anchor box in optical flow image
    imgflo_path = convert_files(flos,u,v, outdir)  #save optical flow pic .flo
    #
    return u,v, xmin, xmax, ymin, ymax, bsize

