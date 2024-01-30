#
# Created on Tue Nov 21 2023
#
# Copyright (c) 2023 DecParallel (十纬)
#
# 从图片中提取净值数据，目前跑通了20231110的5yi图片，10yi图片2022识别识别不出来需要再调整。
# 代码需要整体再做一个整理，更加通用，方便识别私募排排网截取的图片。
#

import cv2
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
from compare_result import compare_res
import sys

def get_coordinate(image_path, x_keywords=['2022', '2023'], y_keywords=['0%', '+15%'], debug = False, image_source='也谈'):
  image = cv2.imread(image_path)

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Use pytesseract to detect the text and its bounding box coordinates
  custom_config = r'--oem 3 --psm 11'
  d = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)
  if debug:
    print(d)
  # Keywords to find the positions for y-axis and x-axis

  # Initialize the dictionary to store coordinates
  axis_coordinates = {'y': {}, 'x': {}}

  # Iterate through the detected text information
  for i in range(len(d['text'])):
      if d['text'][i].strip() in y_keywords:
          # Store the y-coordinates of the keywords for y-axis
          axis_coordinates['y'][d['text'][i].strip()] = (d['left'][i] + 0.5 * d['width'][i], d['top'][i] - 0.5 * d['height'][i])
      if d['text'][i].strip() in x_keywords:
          # Store the x-coordinates of the keywords for x-axis
          if image_source == '也谈':
            axis_coordinates['x'][d['text'][i].strip()] = (d['left'][i] + 0.5 * d['width'][i], d['top'][i] - 0.5 * d['height'][i])
          elif image_source == '私募排排网':
            axis_coordinates['x'][d['text'][i].strip()] = (d['left'][i], d['left'][i] + d['width'][i], d['top'][i] - d['height'][i], d['top'][i])

  # Print the detected coordinates for verification
  if debug:
    print(axis_coordinates)
  return axis_coordinates

def filter_by_gray(image_path):
  # Load the image
  image = cv2.imread(image_path)

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Invert the image
  gray = 255 - gray

  # Threshold the image to isolate the curve
  _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

  # Get the size of the image for normalization
  height, width = binary.shape

  print(height, width)

  # Find the indices of the curve points
  indices = np.where(binary == 255)

  # Since the origin (0,0) in image processing is top-left corner, we need to invert the y-axis
  # indices = (indices[1], height - indices[0])
  indices = (indices[1], indices[0])
  return indices

def filter_out_most_common_y(df, EPS = 0.001):
  # filter out the most common y value at precision eps.
  shift = EPS * 0.66
  def round_eps(x, eps):
    x = x + shift
    return round(x / eps) * eps
  df['y_normalized_round'] = df['y_normalized'].apply(lambda x: round_eps(x, eps=EPS))
  df_count = df.groupby('y_normalized_round')['x'].count().sort_values(ascending=False)
  # take the most common y value.
  y_most_common = df_count.index[0]
  print("y_most_common:", y_most_common)
  df = df[df['y_normalized_round'] != y_most_common]
  return df, y_most_common

def filter_outlier(df, sample_count = 80, outlier_std_count = 3, x_0_date = dt.datetime(2022, 1, 1)):
  # take the data most close to x_0_date
  df_x = df[df['x_normalized'] >= x_0_date]
  df_x = df_x.sort_values(by='x_normalized')
  df_keep = df_x.iloc[0:sample_count]
  # print("df_keep:\n", df_keep)
  # Filter out outliers
  def filter_func(df, start_pos):
    step_length = sample_count // 8
    for start_index in range(0, len(df) - step_length, step_length):
      df_sub = df[start_index:start_index + sample_count]
      mean = df_sub['y_normalized'].mean()
      std = df_sub['y_normalized'].std()
      threshold = max(outlier_std_count * std, 0.02)
      df_to_filter = df_sub[df_sub['y_normalized'] > mean + threshold]
      if len(df_to_filter) > 0:
        df = df.drop(df_to_filter.index)
      df_to_filter = df_sub[df_sub['y_normalized'] < mean - threshold]
      if len(df_to_filter) > 0:
        df = df.drop(df_to_filter.index)
    return df
  
  df = filter_func(df, start_pos=0)
  df = pd.concat([df_keep, df])
  df = df.sort_values(by='x_normalized', ascending=True)
  return df

def get_plot_data(image_path, x_keywords=['2022', '2023'], y_keywords=['0%', '+15%'], tag = '5yi'):
  print("=====================================")
  print("image_path:", image_path)
  # convert y_keywords from % to float
  y_keywords_value = [float(y.strip('%')) / 100 for y in y_keywords]
  print("y_keywords_value:", y_keywords_value)

  if len(x_keywords[0]) == 4:
    year = int(x_keywords[0])
    x_0_date = dt.datetime(year, 1, 1)
    year = int(x_keywords[1])
    x_1_date = dt.datetime(year, 1, 1)
  else:
    x_start = "".join(x_keywords[0].split('-'))
    year = int(x_start[0:4])
    month = int(x_start[4:6])
    day = int(x_start[6:8])
    x_0_date = dt.datetime(year, month, day)

    x_end = "".join(x_keywords[1].split('-'))
    year = int(x_end[0:4])
    month = int(x_end[4:6])
    day = int(x_end[6:8])
    x_1_date = dt.datetime(year, month, day)

  try:  
    if 'yi' in image_path:
      coordinate = get_coordinate(image_path, x_keywords, y_keywords, debug = False, image_source='也谈')
      x_0 = coordinate['x'][x_keywords[0]][0]
      x_1 = coordinate['x'][x_keywords[1]][0]
    elif "zhongxin" in image_path:
      coordinate = get_coordinate(image_path, x_keywords, y_keywords, debug = False, image_source='私募排排网')
      x_0 = coordinate['x'][x_keywords[0]][0]
      x_1 = coordinate['x'][x_keywords[1]][1]

    y_0 = coordinate['y'][y_keywords[0]][1]
    y_1 = coordinate['y'][y_keywords[1]][1]
  except:
    print("can not find coordinate for x or y.")
    get_coordinate(image_path, x_keywords, y_keywords, debug = True)
    return
  
  # 这里可以尝试改成filter_by_color，但是效果不好，可能需要调整红色范围.
  indices = filter_by_gray(image_path)

  # Create a DataFrame for easier handling of the data
  df = pd.DataFrame({'x': indices[0], 'y': indices[1]})

  # Sort by x value and drop duplicate x values to have a single y for each x
  df = df.sort_values(by='x').drop_duplicates(subset='x')

  print("x_0", x_0, "x_1", x_1, "y_0", y_0, "y_1", y_1)
  print("x_min", df['x'].min(), "x_max", df['x'].max(), "y_min", df['y'].min(), "y_max", df['y'].max())
  # remove x < 100 or x > 500
  # if 'yi' in image_path:
  #   df = df[df['y'] > 100]
  #   df = df[df['y'] < 750]

  # Normalize the data points with x_2022 and y_0 as the origin
  def normalize_y(y):
    return (y_keywords_value[1] - y_keywords_value[0]) * (y - y_0) / (y_1 - y_0) + y_keywords_value[0]
  
  df['y_normalized'] = df['y'].apply(normalize_y)
  days = (x_1_date - x_0_date).days

  print("x_0_date:", x_0_date)
  def normalize_x(x):
    x = x - x_0
    x = x / (x_1 - x_0)
    x = x * days
    return x_0_date + dt.timedelta(days=int(x))
  
  df['x_normalized'] = df['x'].apply(normalize_x)
  # filter out the most common y value at precision eps.
  if 'yi' in image_path:
    if tag == '10yi':
      df = filter_outlier(df, sample_count= 120, outlier_std_count = 3, x_0_date = x_0_date)
    df, y_most_common = filter_out_most_common_y(df, EPS = 0.01)
    if tag == '5yi':
      df = df[df['y_normalized'] > -0.035]
    df = df[df['x_normalized'] >= x_0_date]
    df['y_normalized'] = df['y_normalized'] - df['y_normalized'].iloc[0]
    print(df['x_normalized'].min(), df['x_normalized'].max(), df['y_normalized'].min(), df['y_normalized'].max())

  elif "中信" in image_path:
    df, y_most_common = filter_out_most_common_y(df, EPS = 0.01)
    df['y_normalized'] = df['y_normalized'] - y_most_common
    # filter out x_normalized < x_0_date
    df = df[df['x_normalized'] >= x_0_date]

  # filter out y_normalized < -0.3
  df = df[df['y_normalized'] >= -0.3]
  df.to_csv('test_%s.csv' % tag, index=False, header=False)

  # Let's also visualize the curve we have extracted
  # clear the plot
  plt.clf()
  plt.plot(df['x_normalized'], df['y_normalized'])
  plt.title('Extracted Curve Points')
  plt.xlabel('Normalized X')
  plt.ylabel('Normalized Y')
  # plot grid
  plt.grid(True)
  # tilt x label
  plt.xticks(rotation=90)
  plt.show()
  # save image.
  image_name = 'test_%s.png' % tag 
  print("save image to:", image_name)
  plt.savefig(image_name)
  return df

def test5yi():
  image_path = '20231123/5yi.png'
  y_keywords = ['+3%', '+15%']
  x_keywords = ['2022', '2023']
  get_plot_data(image_path, x_keywords, y_keywords, tag = '5yi')

def test10yi():
  image_path = '20231110/10yi.png'
  y_keywords = ['+3%', '+15%']
  x_keywords = ['2022', '2023']
  get_plot_data(image_path, x_keywords, y_keywords, tag = '10yi')

def test_zhongxin():
  image_path = '20231110/中信.png'
  y_keywords = ['20.00%', '40.00%']
  x_keywords = ['2021-11-10', '2023-11-10']
  get_plot_data(image_path, x_keywords, y_keywords, tag = '中信')

def test():
  test5yi()
  test10yi()
  test_zhongxin()
  get_data_of_dir('20231123')
  compare_res()

def get_data_of_dir(dir_name):
  import os
  dir_path = os.path.dirname(os.path.realpath(__file__))
  dir_path = os.path.join(dir_path, dir_name)
  for file in os.listdir(dir_path):
    if file.endswith('.png'):
      print(file)
      if '5yi' in file:
        y_keywords = ['+5%', '+15%']
        x_keywords = ['2022', '2023']
      elif '10yi' in file:
        y_keywords = ['+3%', '+15%']
        x_keywords = ['2022', '2023']
      else:
        y_keywords = ['20.00%', '40.00%']
        x_keywords = zhongxin_x_keywords
      df = get_plot_data(os.path.join(dir_path, file), x_keywords, y_keywords, tag = file.split('.')[0])
      output_filename = file.split('.')[0] + '.csv'
      output_filename = os.path.join(dir_path, output_filename)

      print("saving to:", output_filename)
      df[['x_normalized', 'y_normalized']].to_csv(output_filename, index=False, header=False)
      print("saved to:", output_filename)

if __name__ == "__main__":
  # print pytesseract version, and python version.
  print("pytesseract version:", pytesseract.get_tesseract_version())
  print("python version:", sys.version)
  if len(sys.argv) == 2:
    if sys.argv[1] == 'test':
      zhongxin_x_keywords = ['2021-11-17', '2023-11-17'] # 私募排排网中信图片上的日期
      test()
      exit(0)
    else:
      try:
        dir_name = sys.argv[1]
        zhongxin_x_keywords = ['2021-11-17', '2023-11-17'] # 私募排排网中信图片上的日期
        get_data_of_dir(dir_name)
      except Exception as e:
        print(e)
        print("Usage: python extract_data.py [dir_name]")
        print("Example: python extract_data.py 20231123")
  elif len(sys.argv) == 4:
    try:
      dir_name = sys.argv[1]
      zhongxin_x_keywords = [sys.argv[2], sys.argv[3]]
      get_data_of_dir(dir_name)
    except Exception as e:
      print(e)
      print("Usage: python extract_data.py [dir_name] [zhongxin_x_start] [zhongxin_x_end]")
      print("Example: python extract_data.py 20231123 2021-11-17 2023-11-17")
  else:
    print("Usage: python extract_data.py [dir_name]")
    print("Example: python extract_data.py 20231123")
    print("Usage: python extract_data.py [dir_name] [zhongxin_x_start] [zhongxin_x_end]")
    print("Example: python extract_data.py 20231123 2021-11-17 2023-11-17")
