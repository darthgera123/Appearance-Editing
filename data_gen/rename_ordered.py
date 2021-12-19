import os, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='')

    args = parser.parse_args()

    img_list = os.listdir(args.dir)
    img_list.sort()

    for i, item in enumerate(img_list):
        img_name = str(i).zfill(5)
        extension = item.split('.')[-1]

        os.system('mv %s/%s %s/%s.%s' % (args.dir, item, args.dir, img_name, extension))