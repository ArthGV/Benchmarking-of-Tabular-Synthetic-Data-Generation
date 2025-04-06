"""
    Contains the evaluators for the hyperparameter optimization,
    which are used in the objective function.
"""
from omegaconf import DictConfig
import tapas
from tools.synthetic_evaluation.classification_optimizer import ClassificationOptimizer
from tools.utils import tapas_train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tools.tapas.utils import get_categorical_and_numerical_features

def encode_features(dataset):
    # NOTE this is a replication of the code in tapas.utils
    data_description = dataset.description
    df = dataset.data.copy()
    categorical_features, numerical_features = get_categorical_and_numerical_features(data_description)
    categories = {col["name"]: col["representation"] for col in data_description if col['name'] in categorical_features}
    categories = [categories[col] for col in dataset.data.columns if col in categorical_features]
    ohe = OneHotEncoder(sparse_output=False, categories=categories, handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', ohe, categorical_features)
    ])

    return preprocessor.fit_transform(df)


class Evaluator:
    def __init__(self, cfg: DictConfig, generator):
        self.cfg = cfg
        self.generator = generator
        self.fitted = False

    def fit(self, train_ds):
        pass

    def evaluate(self):
        # evaluate receive the train set and the generator, and does whatever
        # want with it. The main idea is that the HPO pipeline manages
        # the train-val-synth splits and the evaluator manages the strategy to evaluate.
        pass

    def test(self, test_ds):
        pass

# class UtilityEvaluator(Evaluator):
class UtilityEvaluator(Evaluator):
    def __init__(self, cfg: DictConfig, generator):
        super().__init__(cfg, generator)
        self.np_rng = np.random.default_rng(self.cfg.random_state)

    def fit(self, train_ds):
        self.train_ds = train_ds
        self.fitted = True

    def evaluate_large(self):
        """
        Returns the generator's ML utility over different train-val splits.
        """
        # generate multiple seeds based on the original random state
        scores = []
        cv = StratifiedKFold(n_splits=self.cfg.n_cv_folds, shuffle=True, random_state=self.cfg.random_state)

        if self.cfg.n_cv_folds == 1:
            train_data, val_data = tapas_train_test_split(self.train_ds, train_size=self.cfg.data.train_size, random_state=self.cfg.random_state, 
                                                          target_column=self.cfg.data.target_column)
            self.generator.fit(train_data)
            synth_ds = self.generator.generate(len(train_data))
            classification_optimizer = ClassificationOptimizer(self.cfg.classification_optimizer, synth_ds, val_data)
            classification_optimizer.fit()
            score = classification_optimizer.test_classifier(self.cfg.n_repetitions_validation)
            print("Validation score:", score)
            scores.append(score)
        else:
            # only y is required in cv.split
            train_data = self.train_ds.data.copy()
            for train_idx, val_idx in tqdm(cv.split(train_data, train_data[self.cfg.data.target_column]), desc="Cross-validation for synthetic data", total=self.cfg.n_cv_folds):
                kfold_train_ds = self.train_ds.get_records(train_idx)
                kfold_val_ds = self.train_ds.get_records(val_idx)

                print("kfold_train_ds", len(kfold_train_ds))
                print("kfold_val_ds", len(kfold_val_ds))

                self.generator.fit(kfold_train_ds)
                kfold_synth_ds = self.generator.generate(len(kfold_train_ds))
                classification_optimizer = ClassificationOptimizer(self.cfg.classification_optimizer, kfold_synth_ds, kfold_val_ds)
                classification_optimizer.fit()
                score = classification_optimizer.test_classifier(self.cfg.n_repetitions_validation)
                print("Validation score:", score)
                scores.append(score)

        return (np.mean([s[0] for s in scores]), np.std([s[0] for s in scores]))


    def evaluate_small(self):
        scores = []
        random_states = self.np_rng.integers(0, 2**32, size=self.cfg.n_repetitions_small)
        for seed in tqdm(random_states, total=len(random_states), desc="Small synthetic data evaluation"):
            subset_train_ds, subset_val_ds = tapas_train_test_split(
                self.train_ds, 
                train_size=self.cfg.num_original_records_small, 
                test_size=self.cfg.n_val_samples_small,
                random_state=seed, 
                target_column=self.cfg.data.target_column)
            
            self.generator.fit(subset_train_ds)
            synth_ds = self.generator.generate(self.cfg.num_synthetic_records_small)
            classification_optimizer = ClassificationOptimizer(self.cfg.classification_optimizer, synth_ds, subset_val_ds)
            classification_optimizer.fit()
            score = classification_optimizer.test_classifier(self.cfg.n_repetitions_validation)
            scores.append(score)

        return (np.mean([s[0] for s in scores]), np.std([s[0] for s in scores]))


    def test(self, test_ds):
        """
        Returns the generator's ML utility over a test set.
        """
        print("train_ds", len(self.train_ds))
        self.generator.fit(self.train_ds)
        synth_ds = self.generator.generate(self.cfg.num_synthetic_records_large)
        classification_optimizer = ClassificationOptimizer(self.cfg.classification_optimizer, synth_ds, test_ds)
        classification_optimizer.fit()

        return classification_optimizer.test_classifier(n_repetitions=self.cfg.n_repetitions_test)
    

class AuthenticityEvaluator(Evaluator):
    def __init__(self, cfg: DictConfig, generator):
        super().__init__(cfg, generator)
        # self.np_rng = np.random.default_rng(self.cfg.random_state)
        self.alpha_precision = AlphaPrecision()

    def fit(self, train_ds):
        """
        Define the train set dataloader
        """
        self.train_ds = train_ds
        self.train_loader = GenericDataLoader(encode_features(train_ds), random_state=self.cfg.random_state)
        self.fitted = True

    def evaluate(self):
        """
        Evaluate the authenticity of the synthetic data over the train set
        """
        self.generator.fit(self.train_ds)
        synth_ds = self.generator.generate(self.cfg.num_synthetic_records_large)
        synth_loader = GenericDataLoader(encode_features(synth_ds), random_state=self.cfg.random_state)
        return self.alpha_precision.evaluate(self.train_loader, synth_loader)['authenticity_OC']

    def test(self, test_ds):
        """
        
        """
        # this function doesn't make sense for this use case
        pass
        