"""
Hyperparameter optimization tools.
"""

# script config
from hydra.utils import instantiate
from omegaconf import DictConfig

# tapas
from tapas.datasets import TabularDataset
from tapas.generators import ReprosynGenerator

# custom
from tools.utils import get_class_object_from_fqdn, tapas_train_test_split
from tools.tapas.generators import SynthcityGenerator
from tools.hpo.evaluators import UtilityEvaluator, AuthenticityEvaluator

def init_generator(cfg):
    # NOTE should be moved in the .toos.tapas repo
    if 'ReprosynGenerator' in cfg.generator._target_:
        # to instantiate a reprosyn object, we need to split the params so that we can recover the reprosyn class
        ReprosynClass = get_class_object_from_fqdn(cfg.generator.reprosyn_class)
        gen_params = {k:v for k,v in cfg.generator.items() if k != 'reprosyn_class' and k!= '_target_'}
        # the reprosyn objects don't use a "random_sate" but a seed
        gen_params['seed'] = gen_params['random_state']
        gen_params.pop('random_state')
        generator = ReprosynGenerator(ReprosynClass, **gen_params)
    else:
        if cfg.generator.plugin_name == "ddpm":
            # model_params: dict = dict(n_layers_hidden=3, n_units_hidden=256, dropout=0.0)
            gen_params = {k:v for k,v in cfg.generator.items() if k != 'reprosyn_class' and k!= '_target_'}

            gen_params['model_params'] = {
                'n_layers_hidden': gen_params['n_layers_hidden'],
                'n_units_hidden': gen_params['n_units_hidden'],
                'dropout': gen_params['dropout']
            }

            # pop the keys
            gen_params.pop('n_layers_hidden')
            gen_params.pop('n_units_hidden')
            gen_params.pop('dropout')
            print("Generator params", gen_params)

            generator = SynthcityGenerator(**gen_params)
        else:
            generator = instantiate(cfg.generator)

    return generator


class GeneratorHPOPipeline:
    def __init__(self, cfg: DictConfig):
        # all variables related to the optimization process are 
        # provided by hydra using the config file
        self.cfg = cfg
        self.random_state = cfg.random_state
        self.generator = init_generator(self.cfg)
        self.evaluator = UtilityEvaluator(cfg.evaluator, self.generator)
        self.authenticity_evaluator = AuthenticityEvaluator(cfg.evaluator, self.generator)
        self.load_data()
        self._split_train_test_datasets()
        self.evaluator.fit(self.train_ds)
        self.authenticity_evaluator.fit(self.train_ds)


    def load_data(self):
        data_processor = instantiate(self.cfg.data.tapas_data_processor)
        self.dataset = TabularDataset.read(self.cfg.data.path, label=self.cfg.data.label)
        self.dataset = data_processor.process_tapas_tabulardataset(self.dataset)

        if self.cfg.data.num_original_records_large != 'all':
            # sample num for train set (must be equal to num_original_records_large)
            # + sample num for test set. To do so, we need to compute the number of test elems
            num_test = int(self.cfg.data.num_original_records_large/self.cfg.data.train_size/10)
            self.dataset = self.dataset.sample(self.cfg.data.num_original_records_large+num_test, random_state=self.random_state)
            print("num_test", num_test)
        print("dataset size", len(self.dataset))


    def _split_train_test_datasets(self):
        """
        The HPO pipeline is responsible for splitting the data into train and test.
        The will access the train and test data in different function. The generator
        is responsible splitting train and validation tests if they are required.
        """
        # stratified split
        self.train_ds, self.test_ds = tapas_train_test_split(self.dataset, 
            train_size=self.cfg.data.train_size, random_state=self.random_state,
            target_column=self.cfg.data.target_column)
        
        # TODO make sure that test_ds is not too large!
        
        if self.cfg.evaluator.num_synthetic_records_large == 'all':
            self.cfg.evaluator.num_synthetic_records_large = len(self.train_ds)

        print("[GenetatorHPOPipeline] train_ds size", len(self.train_ds))
        print("[GenetatorHPOPipeline] num_synth_record_large size", self.cfg.evaluator.num_synthetic_records_large)
        

    def evaluate_generator(self, is_small=False):
        """
        /!\ Use only in the HPO pipeline
        """
        try:
            if is_small:
                return self.evaluator.evaluate_small()
            else:
                return self.evaluator.evaluate_large()
        except:
            return (0.0, 0.0)
        
    # TODO improve that
    def evaluate_authenticity(self):
        """
        /!\ Use only in the HPO pipeline
        """
        return self.authenticity_evaluator.evaluate()    

    def test_generator(self):
        """/!\ Use only when the best hyperparameters are found"""
        return self.evaluator.test(self.test_ds)
