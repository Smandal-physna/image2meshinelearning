import os
import PIL
import numpy
import cv2
from PIL import Image
import random
en = -1
mm = ['sofa','bookcase','chair','table']
rang = [70,71,72,73,74,75]
required-size = 224
outp = 'pix-all-3d-224'
inp-base = 'pix3d'
split_train_validation = 8 #out of 10: 80-20% train validation split
for furn in mm:
    for r,d,f in os.walk(inp-base+'/'+furn):
        for files in f:
            en+=1
            # print(r,files)
            if 'synthetic_img' in r and '.png' in files:
                print(r+'/'+files)
                # print(r.split('\\')[1])

                image = numpy.asarray(Image.open(r+'/'+files))
                old_size = image.shape
                # ratio = float(desired_size)/max(old_size)
                # new_size = tuple([int(x*ratio) for x in old_size])
                image = cv2.resize(image, (desired_size, desired_size))
                image3 = numpy.zeros((image.shape[1],image.shape[1],3))
                for x in range(image.shape[0]):
                    for y in range(image.shape[1]):
                      # print(image1[x][y],image2[x][y])
                        # if image[x][y][3]==1:
                        if image[x][y][0]==image[x][y][1]==image[x][y][2] and image[x][y][0] in rang:
                            image3[x][y][:]=[255,255,255]
                            # print('1:',image[x][y][3])
                        else:
                            image3[x][y][:]=image[x][y][0:3]
                            # print('2:',image[x][y][3])

                # print(image.shape)
                
                bs = r.split('/')
                
                if random.randint(0,10)<=8:
                    out_f = outp+'/train/'+furn+'/'+r.split('/')[2]+'/'+str(en)+'.png'

                else:
                    out_f = outp+'/validation/'+furn+'/'+r.split('/')[2]+'/'+str(en)+'.png'


                if not os.path.exists(outp+'validation/'+furn+'/'+r.split('/')[2]):
                    os.makedirs(outp+'/validation/'+furn+'/'+r.split('/')[2])

                if not os.path.exists(outp+'/train/'+furn+'/'+r.split('/')[2]):
                    os.makedirs(outp+'/train/'+furn+'/'+r.split('/')[2])

                
                cv2.imwrite(out_f,image3)
                print('done',bs[1],out_f,outp+'/validation/'+r.split('/')[2])
                # exit(0)
                # exit(0)
                # exit(0)

                # if 'bed' in r:

