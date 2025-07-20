import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    '--source',
    type=str,
    help='image to rotate about axis',
)
parser.add_argument(
    '--dest',
    type=str,
    help='where to write the output',
)
parser.add_argument(
    '--rotation',
    type=float,
    help='degrees to rotate by',
)
args = parser.parse_args()

img = cv2.imread(args.source)
h, w, _ = img.shape
rot_x = int(w * args.rotation / 360.0)
out = img.copy()
out[:, :rot_x] = img[:, -rot_x:]
out[:, rot_x:] = img[:, :-rot_x]
cv2.imwrite(args.dest, out)
