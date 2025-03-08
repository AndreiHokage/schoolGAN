from typing import List

import numpy as np
import torch
from captum.attr import DeepLiftShap
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from numpy import ndarray
from tensorboardX.utils import figure_to_image

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm


class DeepLiftShapExplanation(ExplanationAlgorithm):

    def __init__(self, low_prediction_threshold: float):
        super().__init__(low_prediction_threshold)

    def _computeAttributions(self, discriminator: DiscriminatorTeacher, data_forward_expl: torch.Tensor, indices_forward_expl: List, training_data: torch.Tensor) -> None:
        for i in range(len(indices_forward_expl)):
            explainer = DeepLiftShap(discriminator)
            attributionMask = explainer.attribute(data_forward_expl[i, :].detach().unsqueeze(0), training_data, target=0)
            self._gradient_matrix_mask[indices_forward_expl[i], :] = attributionMask

            if indices_forward_expl[i] == self._worst_counterfeit_entity_idx:
                # convert image tensor (C, H, W) to ndarray (H, W, C)
                image_post_explain = data_forward_expl[i, :].permute(1, 2, 0).detach().cpu().numpy()
                # convert attribution mask tensor (1, C, H, W) to ndarray (H, W, C)
                mask_post_explain = attributionMask.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                self.visualiseWorstEntityThroughXAI(image_post_explain, mask_post_explain)

    """
    image: ndarray with shape (H, W, C)
    mask: ndarray with shape (H, W, C)
    """
    def visualiseWorstEntityThroughXAI(self, image: ndarray, mask: ndarray) -> None:
        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

        plt_fig, plt_axis = viz.visualize_image_attr_multiple(mask, image,
                                                              ["original_image", "masked_image", "heat_map"],
                                                              ["all", "positive", "positive"],
                                                              cmap=default_cmap,
                                                              titles=["XAI_INNER1", "XAI_INNER2", "XAI_INNER2"],
                                                              show_colorbar=True,
                                                              use_pyplot=False)
        for ax in plt_axis:
            images_from_axis = ax.get_images()
            image_from_plot = images_from_axis[0].get_array()
            if len(image_from_plot.shape) == 2:
                image_from_plot = np.expand_dims(image_from_plot, axis=-1)
            torch_image = torch.from_numpy(image_from_plot).permute(2, 0, 1)
            self._explanationLog.addImage(torch_image)
