from moviepy.editor import ImageSequenceClip
import argparse, os
import numpy as np
import matplotlib.image as cimg


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    imgPaths = sorted([os.path.join(root, name) for root, dirs, files in os.walk(args.image_folder) 
                                for name in files 
                                if name.split('.')[-1] in ['jpg', 'png', 'jpeg']])
    imgs = [cimg.imread(imgPath) for imgPath in imgPaths]
    clip = ImageSequenceClip(imgs, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
