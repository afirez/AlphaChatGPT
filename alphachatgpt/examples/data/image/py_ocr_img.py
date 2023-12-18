# import cv2

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Image.MAX_IMAGE_PIXELS = 1000000000

import pytesseract


pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = 'D:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# img_path = "./tmp/20231117-210259.jpg"
img_path = "./tmp/20231215-111719.jpg"

def ocr_4_image(img_path):
    # 打开图像文件，仅获取图像信息而不加载图像数据
    with Image.open(img_path, 'r') as img:
        width, height = img.size


    # 打印图像尺寸
    print(f"图像宽度: {width} 像素")
    print(f"图像高度: {height} 像素")

    img = Image.open(img_path, 'r') 

    # txt = pytesseract.image_to_string(img, lang='chi_sim')
    # with open(f"./tmp/text_.txt", mode="w") as f:
    #     f.write(txt)

    i = 0
    with open(f"{img_path}.txt", mode="a") as f:
        while True:

            print(f"图  {i}")

            left = 0
            top = i * 3 * width 
            right = width
            bottom = (i+1) * 3 * width

            if top >= height: 
                break

            if bottom >= height: 
                bottom = height

            print(f"position {left} {top} {right} {bottom}")
            img2 = img.crop((left, top, right, bottom))

            txt = pytesseract.image_to_string(img2, lang='chi_sim')
            f.write(txt)  

            i = i + 1

            if bottom >= height: 
                break
        
ocr_4_image(img_path)

# i = 0
# while True:
    
#     print(f"图  {i}")

#     left = 0
#     top = i * 3 * width 
#     right = width
#     bottom = (i+1) * 3 * width

#     if top >= height: 
#         break

#     if bottom >= height: 
#         bottom = height

#     print(f"position {left} {top} {right} {bottom}")
#     img2 = img.crop((left, top, right, bottom))
#     # img_unit = Image.fromarray((left, top, width, width))
#     # img2 = img_unit

#     txt = pytesseract.image_to_string(img2, lang='chi_sim')
#     with open(f"./tmp/text_{i}.txt", mode="w") as f:
#         f.write(txt)

#     i = i + 1

#     if bottom >= height: 
#         break