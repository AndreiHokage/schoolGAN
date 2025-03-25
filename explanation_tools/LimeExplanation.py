from copy import deepcopy
from typing import List

import numpy as np
import torch
from numpy import ndarray
from torch.nn import functional as F

from discriminatorTeachersModels.DiscriminatorTeacher import DiscriminatorTeacher
from explanation_tools.ExplanationAlgorithm import ExplanationAlgorithm
from lime import lime_image
from skimage.segmentation import mark_boundaries

class LimeExplanation(ExplanationAlgorithm):

    def __init__(self, low_prediction_threshold: float):
        super().__init__(low_prediction_threshold)
        self.__discriminatorLime: DiscriminatorTeacher = None

    def _computeAttributions(self, discriminator: DiscriminatorTeacher, data_forward_expl: torch.Tensor, indices_forward_expl: List, training_data: torch.Tensor) -> None:
        explainer = lime_image.LimeImageExplainer()
        self.__discriminatorLime = deepcopy(discriminator)
        self.__discriminatorLime.cpu()
        self.__discriminatorLime.eval()
        for i in range(len(indices_forward_expl)):
            np_ndarray_lime = self.__convert_tensor_to_lime_format(data_forward_expl[i, :])

            originalNumChannels = 1
            if len(np_ndarray_lime.shape) == 3:
                originalNumChannels = np_ndarray_lime.shape[-1]

            exp = explainer.explain_instance(np_ndarray_lime, lambda x: self.__batch_predict(x, originalNumChannels=originalNumChannels), num_samples=100)
            image_post_explain, mask_post_explain = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
            self._gradient_matrix_mask[indices_forward_expl[i], :] = torch.Tensor(mask_post_explain.astype(float))

            if indices_forward_expl[i] == self._worst_counterfeit_entity_idx:
                self.visualiseWorstEntityThroughXAI(image_post_explain, mask_post_explain)

        del self.__discriminatorLime

    def __convert_tensor_to_lime_format(self, image: torch.Tensor) -> np.ndarray:
        if len(image.shape) == 3:
            num_channels = image.shape[0]
            if num_channels == 1: # enough to squeeze to get rid of the channels dim
                tmp_post_channels = image.squeeze() # (1, H, W) -> (H, W)
            else:
                tmp_post_channels = image.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
            tmp_post_channels = tmp_post_channels.detach().cpu().numpy().astype(np.double)
            return tmp_post_channels
        else:
            raise Exception("No handling shape for converting tensor to lime format")

    def __batch_predict(self, images: np.ndarray, originalNumChannels: int):
        # we forward an image of shape (H, W) to the lime explainer. The surrogate model takes that image and creates
        # data of the next shape (L_NUM_BATCHES, H, W, C). Now we have to adapt it to our discriminator to get the loggits (odss)
        if originalNumChannels == 1:
            images = np.mean(images, axis=3) # -> (L_NUM_BATCHES, H, W) (convert the image on grey scale)
        else:
            images = np.transpose(images, (0, 3, 1, 2)) # (L_NUM_BATCHES, H, W ,3) -> (L_NUM_BATCHES, 3, H, W)
        batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
        logits = self.__discriminatorLime(batch)
        if len(logits.shape) == 4:
            probs = F.softmax(logits, dim=1).view(-1).unsqueeze(1) # (L_NUM_BATCHES, 1, 1, 1) -> (L_NUM_BATCHES, 1)
        else:
            probs = F.softmax(logits, dim=1) # already was in (L_NUM_BATCHES, 1)
        return probs.detach().numpy()

    """
    image: ndarray with shape (H, W, C)
    mask: ndarray with shape (H, W)
    """
    def visualiseWorstEntityThroughXAI(self, image: ndarray, mask: ndarray) -> None:
        # image (H, W, C) (0, 1, 2) -> (C, H, W) (2, 0, 1)
        # we need to normalised 0,1
        plotted_image = torch.Tensor(mark_boundaries(image / 2 + 0.5, mask)).permute(2, 0, 1)
        self._explanationLog.addImage(plotted_image)

