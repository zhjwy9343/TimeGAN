import argparse
import os
from pyheatmap.heatmap import HeatMap
import numpy as np


def count_ccn(ccn, ccn_cnt):
  if ccn == 0:
    return ccn_cnt

  if ccn > 10:
    print(ccn)
    input()

  if ccn in ccn_cnt.keys():
    ccn_cnt[ccn] = ccn_cnt[ccn] + 1
  else:
    ccn_cnt[ccn] = 1

  return ccn_cnt


def head_map(folder, file, file_type):
  click = []
  swipe = []
  for line in open(file):
    if "Array shape" in line or "New slice" in line:
      continue
    [wait_time, x_beg, y_beg, x_end, y_end, is_click] = line.strip().split()
    is_click = float(is_click)
    if is_click > 0.5:
      is_click = 1
    else:
      is_click = 0

    if is_click == 0 and int(float(x_beg)) < 720 and int(float(y_beg)) < 1280:
      swipe.append([int(float(x_beg)), int(float(y_beg))])
      swipe.append([int(float(x_end)), int(float(y_end))])

    if is_click == 1 and int(float(x_beg)) < 720 and int(float(y_beg)) < 1280:
      click.append([int(float(x_beg)), int(float(y_beg))])

  print(file_type, "swipe num:", len(swipe), "click num:", len(click))

  swipe_heat = HeatMap(swipe)
  swipe_heat.clickmap(save_as=os.path.join(folder, file_type + "_swipe_click_map.png"))  # swipe map
  swipe_heat.heatmap(save_as=os.path.join(folder, file_type + "_swipe_heat_map.png"))  # swipe heatmap

  click_heat = HeatMap(click)
  click_heat.clickmap(save_as=os.path.join(folder, file_type + "_click_click_map.png"))  # click map
  click_heat.heatmap(save_as=os.path.join(folder, file_type + "_click_heat_map.png"))  # click heatmap


def click_cnt_stat(file):
  last_status = 0
  ccn = 0 # continuous_click_num
  ccn_cnt = {}
  click_cnt = 0
  for line in open(file):
    if "Array shape" in line or "New slice" in line:
      print("========================")
      ccn_cnt = count_ccn(ccn, ccn_cnt)
      ccn = 0
      continue
    # [wait_time, x_beg, y_beg, x_end, y_end, is_click, 0] = line.strip().split()
    wait_time, x_beg, y_beg, x_end, y_end, is_click, _ = line.strip().split()
    print(wait_time, x_beg, y_beg, x_end, y_end, is_click)
    is_click = float(is_click)
    if is_click > 0.5:
      is_click = 1
    else:
      is_click = 0

    if is_click == 1:
      click_cnt = click_cnt + 1

    if is_click == 0:
      ccn_cnt = count_ccn(ccn, ccn_cnt)
      ccn = 0

    if is_click == 1 and last_status == 1:
      ccn = ccn + 1

    if is_click == 1 and last_status == 0:
      ccn = 1

    last_status = is_click

  ccn_cnt = sorted(ccn_cnt.items(), key=lambda x: x[0])
  print(ccn_cnt)
  print(click_cnt)


def click_session_cnt(file):
  click = False
  session_cnt = 0
  click_session_cnt = 0

  for line in open(file):
    if "Array shape" in line or "New slice" in line:
      if click:
        click_session_cnt = click_session_cnt + 1
      session_cnt = session_cnt + 1
      click = False
      continue
    [wait_time, x_beg, y_beg, x_end, y_end, is_click] = line.strip().split()

    is_click = float(is_click)
    if is_click > 0.5:
      is_click = 1
    else:
      is_click = 0

    if is_click == 1:
      click =True

  print(session_cnt)
  print(click_session_cnt)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("")
  parser.add_argument("--data_folder", type=str, required=True, help="the input data folder with the two .txt files")
  args = parser.parse_args()
  print(args)

  ori_data = 'ori_data_1280_720.txt'
  gen_data = 'gen_data_1280_720.txt'

  head_map(args.data_folder, os.path.join(args.data_folder, ori_data), 'ori')
  head_map(args.data_folder, os.path.join(args.data_folder, gen_data), 'gen')
