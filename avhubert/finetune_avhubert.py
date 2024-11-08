import os

def finetune_avhubert(config_dir, config_name, data_dir, label_dir, tokenizer_model, checkpoint_path, run_dir):
    """
    Función para ejecutar el fine-tuning de un modelo AV-HuBERT.

    :param config_dir: Directorio donde se encuentra el archivo de configuración.
    :param config_name: Nombre del archivo de configuración.
    :param data_dir: Ruta al directorio con los datos (.tsv).
    :param label_dir: Ruta al directorio con las etiquetas (.wrd).
    :param tokenizer_model: Ruta al modelo BPE para tokenización.
    :param checkpoint_path: Ruta al modelo preentrenado.
    :param run_dir: Directorio donde se guardarán los resultados del entrenamiento.
    """
    
    # Comando de entrenamiento
    cmd = (
        f"fairseq-hydra-train --config-dir {config_dir} --config-name {config_name} "
        f"task.data={data_dir} "
        f"task.label_dir={label_dir} "
        f"+task.tokenizer_bpe_model={tokenizer_model} "
        f"+model.w2v_path={checkpoint_path} "
        f"hydra.run.dir={run_dir} "
        f"common.user_dir=`pwd`"
    )
    
    # Imprimir el comando para verificación
    print("Ejecutando el fine-tuning con el siguiente comando:")
    print(cmd)
    
    # Ejecutar el comando
    os.system(cmd)

if __name__ == "__main__":
    # Valores por defecto definidos dentro del script
    DEFAULT_CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf/pretrain"
    DEFAULT_CONFIG_NAME = "large_vox_iter5.yaml"
    DEFAULT_DATA_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data"
    DEFAULT_LABEL_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data"
    DEFAULT_TOKENIZER_MODEL = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_EN/lrs3/spm1000/spm_unigram1000.model"
    DEFAULT_CHECKPOINT_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/Pretrained_Models/large_vox_iter5.pt"
    DEFAULT_RUN_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/finetune"

    # Ejecutar el fine-tuning con los valores definidos en las variables
    finetune_avhubert(
        config_dir=DEFAULT_CONFIG_DIR,
        config_name=DEFAULT_CONFIG_NAME,
        data_dir=DEFAULT_DATA_DIR,
        label_dir=DEFAULT_LABEL_DIR,
        tokenizer_model=DEFAULT_TOKENIZER_MODEL,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        run_dir=DEFAULT_RUN_DIR
    )
