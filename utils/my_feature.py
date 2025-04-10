from __future__ import annotations

import tapas
import numpy as np
from tapas.attacks.set_classifiers import SetFeature
import tapas.datasets


class MyFeature(SetFeature):
    """
    Naive set feature F_Naive from Stadler et al. Mean, median, and variance of
    each column is computed.

    """

    def extract(self, datasets: list[tapas.datasets.TabularDataset]) -> np.array:
        np_data = [dataset.as_numeric for dataset in datasets]
        return np.stack(
            [
                np.concatenate(
                    [
                        np.nanmean(data, axis=0),
                        np.nanmedian(data, axis=0),
                        np.nanvar(data, axis=0),
                        np.max(data, axis=0),
                        np.min(data, axis=0),
                        np.percentile(data, 0.75, axis=0),
                        np.percentile(data, 0.25, axis=0)
                    ]
                )
                for data in np_data
            ]
        )

    @property
    def label(self):
        return "F_Naive"