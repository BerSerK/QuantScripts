import math

def get_direction(x1, y1, x2, y2):
  """
  计算两个经纬度之间的方向

  Args:
    x1: 起始点的经度
    y1: 起始点的纬度
    x2: 终点的经度
    y2: 终点的纬度

  Returns:
    方向角，单位为度, 与正北方的夹角. 0度表示正北方向，90度表示正东方向，180度表示正南方向，270度表示正西方向
  """

  # 将经纬度转换为弧度
  x1 = math.radians(x1)
  y1 = math.radians(y1)
  x2 = math.radians(x2)
  y2 = math.radians(y2)

  # 计算两点之间的经度差
  d_lon = x2 - x1

  # 计算两点之间的纬度差
  d_lat = y2 - y1

  # 计算方位角
  azimuth = math.atan2(d_lon, d_lat)

  # 将方位角转换为度
  azimuth = math.degrees(azimuth)

  # 将方位角归一化为 0 到 360 度之间
  azimuth = (azimuth + 360) % 360

  return azimuth


# 测试
x1 = 117.407394
y1 = 31.907465
x2 = 121.472669
y2 = 31.230416

direction = get_direction(x1, y1, x2, y2)

print(f"x2, y2 在 x1, y1 的 {direction} 方向")