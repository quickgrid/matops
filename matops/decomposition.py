"""Currently only used for compression by low rank matrix, adding noise to input array by introducing
reconstruction error.

Initial poor implementation on cpu without async functionalities, numba jit and cupy gpu processing.
"""

import os

import PIL
from PIL import Image
import numpy as np
import numpy.typing as npt
import skimage
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class MatOpsBase:
    def __init__(self):
        pass

    @classmethod
    def apply_noise(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def save_animation(
            cls,
            file_name: str,
            save_path: str,
            original_img: npt.NDArray,
            feat_low: int = 1,
            feat_high: int = 64,
            extra_ops: str = None,
    ) -> None:
        frame_list = []
        for k in range(feat_low, feat_high):
            reconstructed_img = cls.apply_noise(original_img, num_components=k, extra_ops=extra_ops)
            reconstructed_img = np.clip(reconstructed_img, 0, 1)
            reconstructed_img = (reconstructed_img * 255.0).astype(np.uint8)
            img = Image.fromarray(reconstructed_img).convert('RGB')
            frame_list.append(img)

        frame_list[0].save(
            os.path.join(save_path, f'{file_name}.gif'),
            save_all=True,
            append_images=frame_list[1:],
            optimize=True,
            duration=80,
            loop=0
        )

    @classmethod
    def show_reconstruction(
            cls,
            original_img: npt.NDArray,
            feat_low: int = 1,
            feat_high: int = 64,
            reconstruction_delay: float = 0.01,
            extra_ops: str = None,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(100, 100))
        axes[0].set_label("Original image")
        axes[1].set_label("Compressed image")
        im1 = axes[0].imshow(original_img, cmap='gray')
        im2 = axes[1].imshow(original_img, cmap='gray')

        for k in range(feat_low, feat_high):
            reconstructed_img = cls.apply_noise(original_img, num_components=k, extra_ops=extra_ops)
            im1.set_data(original_img)
            im2.set_data(reconstructed_img)
            fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.draw()
            # fig.canvas.draw()
            plt.pause(reconstruction_delay)


class MTruncatedSVD(MatOpsBase):
    def __init__(self):
        super(MatOpsBase, self).__init__()

    @staticmethod
    def compress(img_gray: npt.NDArray, num_components: int = 16) -> [npt.NDArray, npt.NDArray]:
        pca = TruncatedSVD(n_components=num_components, n_iter=1, random_state=42)
        ims = csr_matrix(img_gray)
        m1 = pca.fit_transform(ims)
        m2 = pca.components_
        return m1, m2

    @staticmethod
    def decompress(m1: npt.NDArray, m2: npt.NDArray) -> npt.NDArray:
        return np.dot(m1, m2)

    @classmethod
    def apply_noise(
            cls, img_gray: npt.NDArray, num_components: int = 16, extra_ops: str = None
    ) -> [npt.NDArray, npt.NDArray]:
        m1, m2 = cls.compress(img_gray, num_components=num_components)
        image_apply_noised = cls.decompress(m1, m2)
        return image_apply_noised


class MQRFactorization(MatOpsBase):
    def __init__(self):
        super(MatOpsBase, self).__init__()

    @staticmethod
    def get_img_mat(img_gray: PIL.Image.Image) -> npt.NDArray:
        imgmat = np.array(list(img_gray.getdata(band=0)), float)
        imgmat.shape = (img_gray.size[1], img_gray.size[0])
        imgmat = np.matrix(imgmat)
        return imgmat

    @classmethod
    def compress(
            cls,
            img_gray: npt.NDArray | PIL.Image.Image,
            num_components: int = 16,
            is_pil_image=False,
            extra_ops: str = None,
    ) -> [npt.NDArray, npt.NDArray]:
        if is_pil_image:
            img_gray = cls.get_img_mat(img_gray)

        q, r = np.linalg.qr(img_gray)
        q, r = np.matrix(q[:, :num_components]), np.matrix(r[:num_components, :])

        if extra_ops is not None and num_components > 2:
            # matops_class = globals()["MTruncatedSVD"]
            matops_class = globals()[extra_ops]
            q_m1, q_m2 = matops_class.compress(q, num_components=num_components)
            q = np.dot(q_m1, q_m2)

        return q, r

    @staticmethod
    def decompress(q: npt.NDArray, r: npt.NDArray) -> npt.NDArray:
        return np.dot(q, r)

    @classmethod
    def apply_noise(
            cls, img_gray: npt.NDArray, num_components: int = 16, extra_ops: str = None
    ) -> [npt.NDArray, npt.NDArray]:
        q, r = cls.compress(img_gray, num_components=num_components, extra_ops=extra_ops)
        image_apply_noised = cls.decompress(q, r)
        return image_apply_noised


if __name__ == '__main__':
    skimg = skimage.data.cat()
    gray_img = skimage.color.rgb2gray(skimg)

    MTruncatedSVD.show_reconstruction(gray_img)
    # MQRFactorization.show_reconstruction(gray_img, extra_ops="MTruncatedSVD")
    # MQRFactorization.show_reconstruction(gray_img)

    # MQRFactorization.save_animation(
    #     original_img=gray_img, file_name="MQRFactorization", save_path=os.getcwd(), feat_high=48
    # )
    #
    # MTruncatedSVD.save_animation(
    #     original_img=gray_img, file_name="MTruncatedSVD", save_path=os.getcwd(), feat_high=32
    # )
