import numpy as np
import os
import math
import cv2
from PIL import Image,ImageDraw,ImageFont
from PWC27 import PWCNET27 as pwc

def get_box_path(boxpath):
    box = os.listdir(boxpath)
    box.sort(key=lambda x: int(x.replace(' ', '').split('.')[0]))

    boxs = []
    for count in range(len(box)):
        im_name = box[count]
        im_path = os.path.join(boxpath, im_name)
        boxs.append(im_path)

    return boxs

def save_data(anchor,npypath,img_path):
    anc_path=os.path.join(npypath,img_path +'.npy')  
    print('anc',anc_path)
    if os.path.exists(anc_path)==0:
        np.save(anc_path,anchor)

    else:
        raise AssertionError(
            "File exists, [{f}] given".format(f=anc_path))
    return anc_path

def ResizeBox(xmin, xmax, ymin, ymax, optical_img, wid, heigh, fileout,img_path):
    image1 = Image.open(img_path)
    h, w = image1.height, image1.width

    bxmin = int((1 - math.sqrt(2)) / 2 * xmax + (1 + math.sqrt(2)) / 2 * xmin)
    bxmax = int((1 - math.sqrt(2)) / 2 * xmin + (1 + math.sqrt(2)) / 2 * xmax)
    bymin = int((1 - math.sqrt(2)) / 2 * ymax + (1 + math.sqrt(2)) / 2 * ymin)
    bymax = int((1 - math.sqrt(2)) / 2 * ymin + (1 + math.sqrt(2)) / 2 * ymax)

    if bxmin < 0:
        bxmin = xmin
    if bxmax > w:
        bxmax = xmax
    if bymin < 0:
        bymin = ymin
    if bymax > h:
        bymax = ymax

    selected_part = optical_img[bymin:bymax, bxmin:bxmax]
    out = cv2.resize(selected_part, (int(wid), int(heigh)))
    cv2.imwrite(fileout, out)
    print(bxmin,bxmax,bymin,type(bymax))
    return int(bxmin),int(bxmax),int(bymin),int(bymax),out

if __name__=='__main__':
    xmlfile_path='/home/data/xml'

    picfile_path='/home/data/pic/'

    xml_path = get_box_path(xmlfile_path)
    xmllen = len(xmlfile_path)
    imglen = len(picfile_path)


    for j in range(len(xml_path) - 1):
        img_path1 = os.path.join(picfile_path, xml_path[j][xmllen:-4] + '.jpg')
        u, v, xmin, xmax, ymin, ymax, bsize = pwc.get_opticaluv(xml_path[j], xmllen, picfile_path)
        width = 100
        height = 100
        for i in range(len(xmin)):

            fileoutu = os.path.join('/home/show/resizebox',
                                    xml_path[j][xmllen:-4] + '_u' + str(i) + '.png')
            print('resizeimgu', fileoutu)
            fileoutv = os.path.join('/home/show/resizebox',
                                    xml_path[j][xmllen:-4] + '_v' + str(i) + '.png')
            print('resizeimgv', fileoutv)
            bxmin, bxmax, bymin, bymax, out_u = ResizeBox(xmin[i], xmax[i], ymin[i], ymax[i], u, width, height, fileoutu,
                                                          img_path1)
            bxmin, bxmax, bymin, bymax, out_v = ResizeBox(xmin[i], xmax[i], ymin[i], ymax[i], v, width, height, fileoutv,
                                                          img_path1)

            anchor = np.zeros([100, 100])

            for x in range(100):
                for y in range(100):
                    anchor[y][x] = math.sqrt((out_u[y][x] * out_u[y][x]) + (out_v[y][x] * out_v[y][x]))
                  
            npy_path = '/home/show/elm/npy2'
            tmp_path = os.path.join(xml_path[j][xmllen:-4] + '_' + str(i))
            npyfile = save_data(anchor=anchor, npypath=npy_path, img_path=tmp_path)
            print(f'npy file has been save in {npyfile}\n')

