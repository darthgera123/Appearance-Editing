import os, argparse, cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=540)
    parser.add_argument('--skip', type=int, default=12)

    args = parser.parse_args()

    img_list = sorted( os.listdir(args.input_dir) )

    j = 0
    for i, item in enumerate(img_list):
        if i % args.skip == 0:
            extension = item.split('.')[-1]

            img = cv2.imread('%s/%s' % (args.input_dir, item))
            img = cv2.resize(img, (args.img_width, args.img_height))

            # os.system('rm %s/%s' % (args.input_dir, item))
            cv2.imwrite('%s/%s.%s' % (args.output_dir, str(j).zfill(5), extension), img)
            j += 1