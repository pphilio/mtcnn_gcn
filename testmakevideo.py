import re
import os
root_dir = os.path.join('face_frames/')    # FIXME
test_dir = os.path.join(root_dir, 'video_face')

draw_dir = os.path.join(test_dir, 'draws')  # FIXME

imgs = [os.path.join(draw_dir, i) for i in os.listdir(draw_dir) if re.search(".jpg$", i)]
print(imgs)