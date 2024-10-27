import os
import cv2
import tempfile
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig

def predict(video_path, ckpt_path, user_dir):
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    data_dir = tempfile.mkdtemp()
    
    tsv_cont = ["/\n", f"test-0\t{video_path}\t{audio_path}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
    label_cont = ["DUMMY\n"]
    
    with open(f"{data_dir}/test.tsv", "w") as fo:
        fo.write("".join(tsv_cont))
        
    with open(f"{data_dir}/test.wrd", "w") as fo:
        fo.write("".join(label_cont))
    
    utils.import_user_module(Namespace(user_dir=user_dir))
    
    modalities = ["audio", "video"]
    gen_subset = "test"
    gen_cfg = GenerationConfig(beam=20)
    
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    models = [model.eval().cuda() for model in models]  
    
    
    saved_cfg.task.modalities = modalities
    saved_cfg.task.data = data_dir
    saved_cfg.task.label_dir = data_dir
    task = tasks.setup_task(saved_cfg.task)
    task.load_dataset(gen_subset, task_cfg=saved_cfg.task)
    

    generator = task.build_generator(models, gen_cfg)

    def decode_fn(x):
        dictionary = task.target_dictionary
        symbols_ignore = generator.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)

    itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
    sample = next(itr)
    sample = utils.move_to_cuda(sample)
    hypos = task.inference_step(generator, models, sample)
    
    ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo)
    
    return hypo

if __name__ == "__main__":
    # Variables de ejemplo
    mouth_roi_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/roi.mp4"
    audio_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/clip.wav"
    ckpt_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/large_noise_pt_noise_ft_433h.pt"
    
    # Obtén el directorio actual
    user_dir = ""
    
    # Ejecuta la predicción
    hypo = predict(mouth_roi_path, ckpt_path, user_dir)
   
    # Imprime la salida
    print(f"Prediction: {hypo}")