import numpy as np

def bresenham(x0, y0, x1, y1):
   points = []
   dx = abs(x1 - x0)
   dy = abs(y1 - y0)
   x, y = x0, y0
   sx = -1 if x0 > x1 else 1
   sy = -1 if y0 > y1 else 1
   if dx > dy:
      err = dx / 2.0
      while x != x1:
         points.append((x, y))
         err -= dy
         if err < 0:
            y += sy
            err += dx
         x += sx
   else:
      err = dy / 2.0
      while y != y1:
         points.append((x, y))
         err -= dx
         if err < 0:
            x += sx
            err += dy
         y += sy
   points.append((x, y))

   return points


def strel_line(length, degrees):
   if length >= 1:
      theta = degrees * np.pi / 180
      x = round((length - 1) / 2 * np.cos(theta))
      y = -round((length - 1) / 2 * np.sin(theta))
      points = bresenham(-x, -y, x, y)
      points_x = [point[0] for point in points]
      points_y = [point[1] for point in points]
      n_rows = int(2 * max([abs(point_y) for point_y in points_y]) + 1)
      n_columns = int(2 * max([abs(point_x) for point_x in points_x]) + 1)
      strel = np.zeros((n_rows, n_columns))
      rows = ([point_y + max([abs(point_y) for point_y in points_y]) for point_y in points_y])
      columns = ([point_x + max([abs(point_x) for point_x in points_x]) for point_x in points_x])
      idx = []
      for x in zip(rows, columns):
         idx.append(np.ravel_multi_index((int(x[0]), int(x[1])), (n_rows, n_columns)))
      strel.reshape(-1)[idx] = 1

   return strel