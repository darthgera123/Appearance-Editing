import os, argparse, cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_width', type=int, default=960)
    parser.add_argument('--img_height', type=int, default=540)
    parser.add_argument('--skip', type=int, default=3)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_file)

    i = 0
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if i%args.skip == 0:
            frame = cv2.resize(frame, (args.img_width, args.img_height))
            cv2.imwrite('%s/%s.jpg' % (args.output_dir, str(j).zfill(5)), frame)
            j += 1

        i += 1