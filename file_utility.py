from PIL import Image

def openImageList(list, resize: bool = True):
    stack = []
    for path in list:
        img = Image.open(path)
        if resize == True:
            img = img.resize((1280, 720))
        stack.append(img)
    return stack

def saveStacktoFile(stack, quality: int = 75):
    '''Saves the arrays in the stack returned by the get exposure stack function to files'''

    print("Saving..")
    i = 0
    path = []
    for array in stack:
        i += 1
        image = Image.fromarray(array)
        image.save(f"image_nr{i}.jpg", quality=quality)
        path.append(f"image_nr{i}.jpg")
    return path

def saveResultToFile(hdr_image , output_path: str = '/', quality: int = 75):
    image = Image.fromarray(hdr_image)
    image.save(f"{output_path}_hdr.jpg", quality=quality)