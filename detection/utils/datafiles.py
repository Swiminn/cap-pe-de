import os
import cv2
import tqdm
import json
import glob
import pathlib
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='root data path')
    parser.add_argument('--train_txt_name', type=str, default='train.txt')
    parser.add_argument('--val_txt_name', type=str, default='val.txt')
    parser.add_argument('--exp', type=str, default='1')
    return parser.parse_args()


def K_Fashion_get_label_1(label_dict, filename=None):
    cls_map = {'상의': 0, '하의': 1, '아우터': 2, '원피스': 3}
    img_w, img_h = label_dict['이미지 정보']['이미지 너비'], label_dict['이미지 정보']['이미지 높이']
    labels = []
    for cls, bboxes in label_dict['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
        if len(bboxes[0]) > 0:
            for bbox in bboxes:
                l, t, w, h = bbox['X좌표'], bbox['Y좌표'], bbox['가로'], bbox['세로']
                x, y = l + w / 2, t + h / 2
                # r, b = l + w, t + h
                c = cls_map[cls]
                labels += ['{} {} {} {} {}\n'.format(
                    c, min(max(x/img_w, 0),1), min(max(y/img_h, 0),1), min(max(w/img_w, 0),1), min(max(h/img_h, 0),1)
                )]

    # CHECK: image and bounding boxes
    # try:
    #     img = cv2.imread(filename.replace('.json', '.jpg')) / 255.
    # except:
    #     img = cv2.imread(filename.replace('.json', '.JPG')) / 255.
    # for cls, bboxes in label_dict['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
    #     if len(bboxes[0]) > 0:
    #         for bbox in bboxes:
    #             l, t, w, h = bbox['X좌표'], bbox['Y좌표'], bbox['가로'], bbox['세로']
    #             x, y = l + w / 2, t + h / 2
    #             c = cls_map[cls]
    #             cv2.rectangle(img, (int(l), int(t)), (int(l + w), int(t + h)), (1 - c/4, c/4, c/4), 3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return labels


def K_Fashion_get_label_2(label_dict, filename=None):
    category_cls_map = {'상의': 0, '하의': 1, '아우터': 2, '원피스': 3}
    color_cls_map = {'블랙': 0, '그레이': 0, '실버': 0, '화이트': 1, '레드': 2, '와인': 2, '핑크': 3, '퍼플': 3, '라벤더': 3,
                     '베이지': 4, '브라운': 4, '오렌지': 5, '옐로우': 5, '골드': 5, '그린': 6, '카키': 6,
                     '블루': 7, '네이비': 7, '스카이블루': 7, '민트': 7, '네온': 8}
    img_w, img_h = label_dict['이미지 정보']['이미지 너비'], label_dict['이미지 정보']['이미지 높이']
    labels = []
    for cls, bboxes in label_dict['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
        if len(bboxes[0]) > 0 and '색상' in label_dict['데이터셋 정보']['데이터셋 상세설명']['라벨링'][cls][0]:
            for bbox in bboxes:
                l, t, w, h = bbox['X좌표'], bbox['Y좌표'], bbox['가로'], bbox['세로']
                x, y = l + w / 2, t + h / 2
                # r, b = l + w, t + h
                c = category_cls_map[cls]
                color_name = label_dict['데이터셋 정보']['데이터셋 상세설명']['라벨링'][cls][0]['색상']
                color = color_cls_map[color_name]
                labels += ['{} {} {} {} {} {}\n'.format(
                    c, min(max(x/img_w, 0),1), min(max(y/img_h, 0),1), min(max(w/img_w, 0),1), min(max(h/img_h, 0),1), color
                )]

    # CHECK: image and bounding boxes
    # try:
    #     img = cv2.imread(filename.replace('.json', '.jpg')) / 255.
    # except:
    #     img = cv2.imread(filename.replace('.json', '.JPG')) / 255.
    # for cls, bboxes in label_dict['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'].items():
    #     if len(bboxes[0]) > 0:
    #         for bbox in bboxes:
    #             l, t, w, h = bbox['X좌표'], bbox['Y좌표'], bbox['가로'], bbox['세로']
    #             x, y = l + w / 2, t + h / 2
    #             c = cls_map[cls]
    #             cv2.rectangle(img, (int(l), int(t)), (int(l + w), int(t + h)), (1 - c/4, c/4, c/4), 3)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return labels


def K_Fashion_datafiles(data_path, mode, exp):
    mode_name = 'Training' if mode == 'train' else 'Validation'

    # make labels for json files
    jsonfile_list = glob.glob(str(data_path / 'images' / mode_name) + '/*/*.json')
    pbar = tqdm.tqdm(total=len(jsonfile_list), leave=True, desc='processing {} label'.format(mode), dynamic_ncols=True)
    accumulated_iter = 0
    data_list = []
    for jsonfile in jsonfile_list:
        # read json file
        with open(jsonfile, 'r') as f1:
            label_dict = json.load(f1)
        labels = globals()['K_Fashion_get_label_' + exp](label_dict, jsonfile)
        if len(labels) == 0:
            continue

        # save yolo-style label
        save_dir, save_name = jsonfile.replace('images', 'labels').rsplit('/', 1)
        os.makedirs(save_dir, exist_ok=True)
        f2 = open(save_dir + '/' + save_name.replace('.json', '.txt'), 'w')
        for l in labels:
            f2.write(l)
        f2.close()
        data_list.append(jsonfile.replace('.json', '.jpg'))

        accumulated_iter += 1
        pbar.update()
        pbar.set_postfix(dict(total_it=accumulated_iter))

    # # output data list
    # data_list = glob.glob(str(data_path / 'images' / mode_name) + '/*/*.jpg')
    return data_list


def create_datafiles(opt):
    data_path = pathlib.Path(opt.data_path)

    train_file = open(data_path / opt.train_txt_name, "w")
    # for datafile in glob.glob(str(data_path / 'train' / 'images' / '*.jpg')):
    for datafile in K_Fashion_datafiles(data_path, 'train', opt.exp):
        train_file.write(datafile + '\n')
    train_file.close()

    val_file = open(data_path / opt.val_txt_name, "w")
    # for datafile in glob.glob(str(data_path / 'valid' / 'images' / '*.jpg')):
    for datafile in K_Fashion_datafiles(data_path, 'val', opt.exp):
        val_file.write(datafile + '\n')
    val_file.close()


if __name__ == '__main__':
    opt = parse_opt()
    create_datafiles(opt)
