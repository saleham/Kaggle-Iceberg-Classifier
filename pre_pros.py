def pre_pros(img):
    for id in range(img.shape[0]) :
        for channel in range(img.shape[3]):
            band = img[id,:,:,channel]
            #help with speckling
            band = lee_filter(band, 4)
            # Rescale
            band = (band - band.mean()) / (band.max() - band.min())
            img[id, :, :, channel] = band
    return img

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2

    overall_variance = variance(img)

    img_weights = img_variance ** 2 / (img_variance ** 2 + overall_variance ** 2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output