from PIL import Image
import numpy as np

def imresize_bicubic(img: Image.Image, scale: float, down: bool = True) -> Image.Image:
    if down:
        new_w, new_h = int(img.width / scale), int(img.height / scale)
    else:
        new_w, new_h = int(img.width * scale), int(img.height * scale)
    return img.resize((max(1, new_w), max(1, new_h)), Image.BICUBIC)

def rgb2y(img: Image.Image) -> np.ndarray:
    y, _, _ = img.convert('YCbCr').split()
    return np.asarray(y).astype('float32')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img1 = Image.open("example.jpg")
    img2 = imresize_bicubic(imresize_bicubic(img1, scale=16, down=True), scale=16, down=False)
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img1_np)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Resized Image")
    plt.imshow(img2_np)
    plt.axis("off")
    plt.tight_layout()
    plt.show()