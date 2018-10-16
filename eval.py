#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from mve import VelocityEstimator


def draw_mask(img):
  cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

  x_top_offset = 180
  x_btm_offset = 35

  poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
  cv2.fillPoly(img, poly_pts, (255, 255, 255))

  return img


if __name__ == "__main__":

  if len(sys.argv) != 3:
    print("{} video.mp4 <scale or ground_truth.txt>".format(sys.argv[0]))
    print("second arg is either ground truth file or scale")
    print("example: {} video.mp4 6000".format(sys.argv[0]))
    exit(-1)

  cap = cv2.VideoCapture(sys.argv[1])

  frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  FRAMES = frame_cnt
  if os.getenv("FRAMES") is not None:
    FRAMES = int(os.getenv("FRAMES"))

  try:
    gt = np.loadtxt(sys.argv[2], delimiter='\n')[:FRAMES]

    scale = None
  except:
    gt = None

    try:
      scale = float(sys.argv[2])
    except:
      scale = None

  ve = VelocityEstimator(FRAMES, gt=gt, scale=scale, kp_offset=(130, 35))

  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  mask = np.zeros(shape=(H, W), dtype=np.uint8)
  mask.fill(255)
  draw_mask(mask)

  kps_prev = None

  try:
    while True:
      _, frame = cap.read()

      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ve.process_frame(frame_gray[130:350, 35:605], mask[130:350, 35:605])

      # annotate & show frame
      if os.getenv("SHOW_DISPLAY") is not None:
        mask_inv = frame.copy()
        draw_mask(mask_inv)
        mask_inv = cv2.bitwise_not(mask_inv)
        frame = cv2.addWeighted(frame, 1, mask_inv, 0.3, 0)

        # draw keypoints from current and last frame
        kps = ve.get_kps()
        cv2.drawKeypoints(frame, kps, frame, color=(0, 0, 255))
        cv2.drawKeypoints(frame, kps_prev, frame, color=(0, 255, 0))

        font = cv2.FONT_HERSHEY_DUPLEX
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        cv2.putText(frame, "frame {}".format(frame_id), (10, 35), font, 1.2, (0, 0, 255))

        cv2.imshow("video", frame)
        cv2.waitKey(50)

        kps_prev = kps

      print("frame", cap.get(cv2.CAP_PROP_POS_FRAMES))

      if cap.get(cv2.CAP_PROP_POS_FRAMES) >= FRAMES:
        break
  except KeyboardInterrupt:
    pass

  cap.release()
  cv2.destroyAllWindows()

  preds = ve.get_preds()

  with open("preds.txt", "w") as f:
    for p in preds:
      f.write(str(p) + "\n")

  plt.xlabel("frame")
  plt.ylabel("speed (m/s)")
  if gt is not None:
    mse = mean_squared_error(preds, gt)
    print("mse", mse)
    print("scale", ve.get_scale())
    plt.title("mse={}".format(mse))
    plt.plot(gt)
  plt.plot(preds)

  plt.savefig("plt.jpg")

  if os.getenv("SHOW_PLOT") is not None:
    plt.show()

