from PIL import Image as PILImage
import numpy as np

class LBP:
    def __init__(self) -> None:
        pass

    def get_RGB(self, path: str):
        image = np.array(PILImage.open(path).convert('RGB'))
        return image

    def get_combinations(self):
        return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

    def get_padding(self, array, padding_value=0):
        padded_array = np.pad(array, pad_width=1, mode='constant', constant_values=padding_value)
        return padded_array

    def get_LBP(self, image, channel):
        array = image[:, :, channel]  

        padded_arr = self.get_padding(array)
        new_arr = np.zeros((padded_arr.shape[0] - 2, padded_arr.shape[1] - 2), dtype=np.uint8)
        weights = (2 ** np.arange(8))[::-1]

        for i in range(1, padded_arr.shape[0] - 1):
            for j in range(1, padded_arr.shape[1] - 1):
                binary_pattern = (padded_arr[i-1:i+2, j-1:j+2] > padded_arr[i, j]).astype(int)
                neighbors = binary_pattern[tuple(zip(*self.get_combinations()))]
                new_arr[i-1, j-1] = np.dot(neighbors, weights)
        return new_arr

    def compute_histogram(self, lbp_image):
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(257), range=(0, 256))
        return hist

    def LBP_histogram(self, image):
        histograms = []
        for channel in range(3):  
            lbp_image = self.get_LBP(image, channel)

            for i in range(0, 256, 64): 
                for j in range(0, 256, 64):  
                    square = lbp_image[i:i+64, j:j+64]  
                    hist = self.compute_histogram(square)  
                    histograms.append(hist)

        final_histogram = np.concatenate(histograms)
        return final_histogram

    def display_LBP_RGB(self, image):
        import matplotlib.pyplot as plt
        lbp_images = [self.get_LBP(image, channel) for channel in range(3)]
        combined_lbp = np.stack(lbp_images, axis=-1).astype(np.uint8)
        plt.figure(figsize=(6, 6))
        plt.imshow(combined_lbp)
        plt.axis('off')
        plt.title("LBP for RGB Channels")
        plt.show()
