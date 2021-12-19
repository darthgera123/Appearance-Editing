import cv2
import pathlib


def extract_and_sample_frames(args: dict) -> None:
    train_file = pathlib.Path(args.train_video).resolve(strict=True)
    test_file = pathlib.Path(args.test_video).resolve(strict=True)

    video_file = {
        'train': train_file,
        'test': test_file
    }
    skip_frames = {'train': args.train_skip, 'test': args.test_skip}

    dataset_dir = train_file.parent

    dirs = []
    for _dir in ['video_frames', 'video_frames_test', 'colmap_capture',
                 'video_frames_test_mask', 'video_frames_mask', 'colmap_output',
                 'colmap_output/dense', 'colmap_output/dense/0', 'colmap_output/colmap_output_test',
                 'colmap_output/colmap_output_test/dense', 'colmap_output/colmap_output_test/dense/0', 
                 'dr_log', 'dr_tensorboard', 'optimized_textures']:
        dirs.append(dataset_dir / _dir)
        dirs[-1].mkdir(parents=True, exist_ok=True)

    frame_num = 0
    for split_type in ['train', 'test']:
        reader = cv2.VideoCapture(str(video_file[split_type]))

        success = True
        all_frames = 0
        while success:
            all_frames += 1
            success, frame = reader.read()

            if not success:
                continue

            if args.center_crop == 'yes':  # center crop
                frame = frame[:, frame.shape[0] // 2: -frame.shape[0] // 2]
            else:  # frame resize
                frame = cv2.resize(frame, (args.width, args.height))

            if all_frames % skip_frames[split_type] == 0:
                img_path = dirs[0 if split_type == 'train' else 1] / f'{frame_num}.png'

                cv2.imwrite(str(img_path), frame)

                if split_type == 'train':
                    if frame_num % args.colmap_skip == 0:
                        cv2.imwrite(str(dirs[2] / f'{frame_num}.png'), frame)

                frame_num += 1
