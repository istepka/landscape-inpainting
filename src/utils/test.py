import PIL.Image as Image
import numpy as np

im = r"data\processed\0a0nfjnzvr2w60s5qp7b88v4l.jpg"

im = Image.open(im)
print(np.array(im).shape)
im.show()