from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu.mmlu_val_0shot_gen import mmlu_datasets

    from ..models.hf_llama.hf_llama3_8b_instruct import model_init as llama3_8b_instruct_model
    from ..models.hf_llama.hf_llama3_8b_instruct import model_mmlu_CRaFT as llama3_8b_instruct_mmlu_CRaFT_model

datasets = mmlu_datasets
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])
work_dir = './results/Eval/' + __file__.split('/')[-1].split('.')[0] + '/' # do NOT modify this line, yapf: disable, pylint: disable
