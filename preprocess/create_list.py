import os, argparse
#assume that new_frames dir is in the same dir as images"
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--frames_dir', type=str, default='')
    parser.add_argument('--relative_path', type=str, default='../')
    parser.add_argument('--output_file', type=str, default='image-list.txt', help='')

    args = parser.parse_args()

    images = os.listdir(args.dataset_dir+'/'+args.frames_dir)
    file_name = args.output_file

    with open(file_name,'w') as f:
        for image in images:
            path = os.path.join(args.relative_path+'/'+args.frames_dir,image)
            f.write(path+'\n')

