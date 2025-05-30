from reprosyn.generator import PipelineBase

from .data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from .data_synthesiser_utils.datatypes.constants import NUMERICAL

def get_metadata(metadata):
    meta = {}
    columns_list = []
    for col in metadata:
        if col["type"] == 'finite':
            columns_list.append(
                {"name": col["name"],
                 "type": "Categorical",
                 "size": len(col["representation"]),
                 "i2s": col["representation"],
                }
            )
        elif col["type"] in NUMERICAL:
            columns_list.append(
                {"name": col["name"],
                 "type": col["type"],
                 "min": col["min"],
                 "max": col["max"],
                }
            )
    meta["columns"] = columns_list
    return meta


class DS_INDHIST(PipelineBase):
    def __init__(self, histogram_bins=10, **kw):
        parameters = {
            "histogram_bins": histogram_bins,
        }

        self.gen = None

        super().__init__(**kw, **parameters)

    def preprocess(self):

        self.domain = get_metadata(self.dataset.metadata)

    def generate(self, refit=False):

        if (not self.gen) or refit:
            self.gen = IndependentHistogram(self.domain, **self.params)
            self.gen.fit(self.dataset.data)

        self.output = self.gen.generate_samples(self.size)


class DS_BAYNET(PipelineBase):
    def __init__(self, histogram_bins=10, degree=1, seed=None, **kw):
        parameters = {
            "histogram_bins": histogram_bins,
            "degree": degree,
            "seed": seed,
        }

        self.gen = None

        super().__init__(**kw, **parameters)

    def preprocess(self):

        self.domain = get_metadata(self.dataset.metadata)

    def generate(self, refit=False):

        if (not self.gen) or refit:
            self.gen = BayesianNet(self.domain, **self.params)
            self.gen.fit(
                self.dataset.data.astype("object")
            )  # hack to get round a not implemented error when dtype=="category"

        self.output = self.gen.generate_samples(self.size)


class DS_PRIVBAYES(PipelineBase):
    def __init__(
        self, histogram_bins=10, degree=1, epsilon=1, seed=None, **kw
    ):
        parameters = {
            "histogram_bins": histogram_bins,
            "degree": degree,
            "seed": seed,
            "epsilon": epsilon,
        }

        self.gen = None

        super().__init__(**kw, **parameters)

    def preprocess(self):

        self.domain = get_metadata(self.dataset.metadata)

    def generate(self, refit=False):

        if (not self.gen) or refit:
            self.gen = PrivBayes(self.domain, **self.params)
            self.gen.fit(self.dataset.data.astype("object"))

        self.output = self.gen.generate_samples(self.size)
