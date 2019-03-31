from PIL import Image


image = Image.open('images/hoa10.jpg')
img = image.resize((128, 128), Image.ANTIALIAS)
img.save('images/hoa10_re.jpg')
