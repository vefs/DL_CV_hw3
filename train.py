from models import *
from utils.func import *
from utils.datasets import *
from utils.parse_config import *
from utils.construct_datasets import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

#################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, help="training mode")
    parser.add_argument("--detect", default=False, help="detection mode")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    opt = parser.parse_args()
    opt.batch_size = 4 #6
    opt.n_cpu = 16
    print("\n [INFO] parsing running opt: ", opt)

    def prep_dataset_file():
        print("\n\n prep_dataset_file -----------------\n")
        covert_to_label(train_folder, 'digitStruct.mat', label_folder)
        gen_path_file(train_folder, label_folder, dataset_folder)

    # Get data configuration

    dataset_folder = './data/digit_mini'
    #dataset_folder = './data/svhn'
    train_folder = dataset_folder+'/images/train'
    label_folder = dataset_folder+'/labels/train'

    #dtrain_path = dataset_folder+'/'+'train_mini.f'
    dtrain_path = dataset_folder+'/'+'train.f'
    #eval_folder = './data/svhn/images/test_train/'
    eval_folder = './data/svhn/images/test/'
    class_names = ['10', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    print("\n [INFO] train_folder: ", train_folder)
    print(" [INFO] label_folder: ", label_folder)
    print(" [INFO] train_path: ", dtrain_path)

    print("\n [INFO] eval_folder: ", eval_folder)
    print(" [INFO] class_names: ", class_names)

    if not os.path.isfile(dtrain_path):
        print(" [INFO] Convert dataset file: ", dtrain_path)
        prep_dataset_file()


    # Initiate model
    if (torch.cuda.is_available()):
    	model = Darknet('./config/custom.cfg').cuda()
    else:
    	model = Darknet('./config/custom.cfg')
    #model = Darknet('./config/yolov3-tiny.cfg')
    #model.apply(weights_init_normal)


    # Initiate Dataloader
    my_dataset = MyDataset(dtrain_path, augment=True, multiscale=opt.multiscale_training)
    my_dataloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=my_dataset.collate_fn,
    )

    my_testloader = DataLoader(
        ImageFolder(eval_folder, img_size=opt.img_size),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # if epoch == 1:
    #     learning_rate = 0.0005
    # if epoch == 2:
    #     learning_rate = 0.00075
    # if epoch == 3:
    #     learning_rate = 0.001
    # if epoch == 30:
    #    learning_rate=0.0001
    # if epoch == 40:
    #    learning_rate=0.00001


    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #################################################
    def train_dataset(load_mdl=False, save_mdl=True):
        print(" [INFO] START training")
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        if(load_mdl):
            print(" [INFO] Load ./checkpoints/yolo.pth")
            #model.load_state_dict(torch.load('./checkpoints/yolo.pth'))
            checkpoint = torch.load('./checkpoints/yolo.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.train()
        loss_values = []
        iters = 1
        for epoch in range(iters):
            #if epoch %5 ==0:
            print(" [TRAIN] New Epoch ", epoch, " ", datetime.datetime.now().time())
            running_loss = 0.0
            for batch_i, (_, imgs, targets) in enumerate(my_dataloader):
                imgs = imgs.cuda()
                targets = targets.cuda()

                optimizer.zero_grad()
                loss, outputs = model(imgs, targets)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if epoch %5 ==0:
                print("      running_loss=", running_loss)
                loss_values.append(running_loss)

        if(save_mdl):
            #torch.save(model.state_dict(), './checkpoints/yolo_new.pth')
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                       },'./checkpoints/yolo_new.pth')
        print(" [INFO] END training")


    def detect_img(load_mdl=False):
        print("\n [INFO] START detection")

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        if(load_mdl):
            print(" [EVAL] Load ./checkpoints/yolo_save")
            #model.load_state_dict(torch.load('./checkpoints/yolo_save'))
            checkpoint = torch.load('./checkpoints/yolo_1226.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        for b_ii, (img_paths, input_imgs) in enumerate(my_testloader):
            path = img_paths
            #print (" b_ii=", b_ii," img path: ", path)
            input_imgs = Variable(input_imgs.type(Tensor))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, 0.8, 0.4)
            # Save image and detections
            imgs.extend(img_paths)
            img_detections.extend(detections)

        out_json = []
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            print ("\n img_i=", (img_i+1)," img path: ", path)

            img = np.array(Image.open(path))
            if detections is not None :
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
                #print ("detections: ", detections)

                det, rank = detections[:,0].sort(0, descending=False)
                bbox_g = []
                label_g = []
                score_g = []
                for ii in range(detections.shape[0]):
                    ind = rank[ii]
                    #print ("  ii: ", rank[ii]," detections", detections[ind])
                    (x1, y1, x2, y2, conf, cls_conf, cls_pred) = detections[ind]
                    t_label = class_names[int(cls_pred)]
                    p_score = format(conf, '.5f')
                    b_box   = [int(y1), int(x1), int(y2), int(x2)]
                    #print ("  label:", t_label, "p=", p_score, " [",int(x1), int(y1), int(x2), int(y2), "]")
                    #print ("  label:", t_label, "p=", p_score, " b_box=", b_box)

                    label_g.append(t_label)
                    score_g.append(p_score)
                    bbox_g.append(b_box)
                dict_g = {'bbox':bbox_g, 'score':score_g, 'label':label_g}
                out_json.append(dict_g)

        #print(out_json)
        with open('./output/out.json', 'w') as outfile:
            json.dump(out_json, outfile)

    #################################################
    # MAIN FUNC
    #check_dataset()
    if(opt.train):
        train_dataset(load_mdl=False, save_mdl=False)
    if(opt.detect):
        detect_img(load_mdl=True)

    #################################################

