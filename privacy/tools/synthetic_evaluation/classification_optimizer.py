from tqdm import tqdm
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score

from tapas.datasets import TabularDataset
from tools.tapas.utils import get_categorical_and_numerical_features
from tools.utils import tapas_train_test_split


class ClassificationOptimizer:
    def __init__(self, cfg: DictConfig, train_ds: TabularDataset=None, test_ds: TabularDataset=None):
        self.cfg = cfg
        self.random_state = cfg.random_state

        self.dataset = None
        self.data_description = None
        self.pipeline = None
        self.clf = None
        self.fitted = False

        self.target_column = cfg.data.target_column
        if train_ds is None:
            self._load_data()
            # split if not provided
            train_ds, test_ds = tapas_train_test_split(self.dataset, train_size=cfg.pipeline.train_size, 
                                                   target_column=self.target_column, random_state=self.random_state)
            print("Train size:", train_ds.data.shape[0])
            print("Test size:", test_ds.data.shape[0])
        self._set_train_test_ds(train_ds, test_ds)
            
        self.categorical_features, self.numerical_features = get_categorical_and_numerical_features(self.data_description, target_column=self.target_column)
        self.f1_avg = cfg.eval.f1_avg
        self._set_classification_pipeline()

        # not necessary in generator HPO
        try:
            self.n_cv_splits = cfg.eval.n_cv_splits
        except:
            pass


    def _load_data(self):
        # TODO this function is a duplicate of the one in GeneratorHPOPipeline
        tapas_data_processor = instantiate(self.cfg.data.tapas_data_processor)
        self.dataset = TabularDataset.read(self.cfg.data.path, label=self.cfg.data.label)
        self.dataset = tapas_data_processor.process_tapas_tabulardataset(self.dataset)
        if self.cfg.data.drop_duplicates:
            self.dataset.data = self.dataset.data.drop_duplicates()
        self.data_description = self.dataset.description

    def _encode_labels(self):
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.y_train)
        self.y_test = label_encoder.transform(self.y_test)


    def _set_train_test_ds(self, train_ds, test_ds):
        self.data_description = train_ds.description
        self.X_train, self.X_test = train_ds.data.copy(), test_ds.data.copy()    
        self.y_train, self.y_test = self.X_train.pop(self.target_column), self.X_test.pop(self.target_column)
        self._encode_labels()


    def _set_classification_pipeline(self):
        numerical_scaler = instantiate(self.cfg.pipeline.numerical_scaler)

        categories = {col["name"]: col["representation"] 
                      for col in self.data_description 
                      if col['name'] in self.categorical_features}
        # take the categories in the same order as the columns
        categories = [categories[col] for col in self.data_description.columns 
                      if col in self.categorical_features]
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=categories)

        preprocessor = ColumnTransformer([
            ('numerical', numerical_scaler, self.numerical_features),
            ('categorical', ohe, self.categorical_features)
        ])

        self.clf = instantiate(self.cfg.pipeline.classifier)
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', self.clf)
        ])
        
    
    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.fitted = True


    def evaluate_classifier(self):
        # perform cross-validation over the full training set
        cv = StratifiedKFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.pipeline, X=self.X_train, y=self.y_train, 
                                    cv=cv, scoring=f'f1_{self.f1_avg}')
        
        return cv_scores.mean(), cv_scores.std()       


    def test_classifier(self, n_repetitions=1):
        if not self.fitted:
            raise ValueError("The classifier has not been trained yet.")
        
        scores = []
        for _ in range(n_repetitions):
            idx = np.random.choice(self.X_test.shape[0], size=self.X_test.shape[0], replace=True)
            X_test_resample = self.X_test.iloc[idx]
            y_test_resample = self.y_test[idx]
            y_pred = self.pipeline.predict(X_test_resample)
            scores.append(f1_score(y_test_resample, y_pred, average=self.f1_avg))
            
        # compute 95% interval over the metrics
        mean = np.mean(scores)
        lower_ci, upper_ci = np.percentile(scores, [2.5, 97.5])
        return mean, lower_ci, upper_ci