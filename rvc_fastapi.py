
import subprocess
from typing import Union
import os
import sys
from io import BytesIO
import torch
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import tempfile


# Check if CUDA is available and print the result
cuda_available = torch.cuda.is_available()
print("CUDA is available: ", cuda_available)
# Print the PyTorch version
print("PyTorch version: ", torch.__version__)


# don't like settings paths like this at all but due bad code its necessary
now_dir = os.getcwd()
sys.path.append(now_dir)
print(now_dir)

from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC
import fastapi
import uvicorn

# load_dotenv is also very bad practice but necessary due bad code
load_dotenv()

app = fastapi.FastAPI(
    title="Retrieval-based Voice Conversion FastAPI",
    summary="Infer previously cloned voices",
    version="0.0.2",
    contact={
        "name": "w4hns1nn",
        "url": "https://github.com/w4hns1nn",
    }
)

tags = [
    {
        "name": "voice2voice",
        "description": "Voice2Voice conversion using the pretrained model"
    }
]

class ModelCache:
    """
    This class is used to cache a single model to avoid reloading it for every request.
    It re-initializes itself when a different model is requested.
    """
    def __init__(self):
        self.model = None
        self.model_name = ""

    def load_model(self, model_name: str, device: str = None, is_half: bool = True):
        # Check if the requested model is different from the currently cached model
        if model_name != self.model_name:
            # Re-initialize the cache
            self.model = None
            self.model_name = ""
            # Load the new model
            config = Config()
            config.device = device if device else config.device
            config.is_half = is_half if is_half else config.is_half
            vc = VC(config)
            vc.get_vc(f"{model_name}.pth")
            self.model = vc
            self.model_name = model_name
        return self.model

# Create model cache
model_cache = ModelCache()

def infer(
        input: Union[str, bytes],  # filepath or raw bytes
        model_name: str,
        index_path: str = None,
        f0up_key: int = 0,
        f0method: str = "crepe",
        index_rate: float = 0.66,
        device: str = None,
        is_half: bool = False,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 1,
        protect: float = 0.33,
        **kwargs
    ):
    # Check if input is bytes, if so, write to a temporary file
    if isinstance(input, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(input)
        temp_file.close()
        input_path = temp_file.name
    else:
        input_path = input

    model_name = model_name.replace(".pth", "")

    if index_path is None:
        index_path = os.path.join("logs", model_name, f"added_IVF1254_Flat_nprobe_1_{model_name}_v2.index")
        if not os.path.exists(index_path):
            raise ValueError(f"autinferred index_path {index_path} does not exist. Please provide a valid index_path")

    vc = model_cache.load_model(model_name, device=device, is_half=is_half)

    _, wav_opt = vc.vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=f0up_key,
        f0_file=None,
        f0_method=f0method,
        file_index=index_path,
        file_index2=None,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect
    )

    # using virtual file to be able to return it as response
    wav_path = 'temp.wav'
    mp3_path = 'temp.mp3'
    wavfile.write(wav_path, wav_opt[0], wav_opt[1])

    # Convert to mp3 using ffmpeg
    subprocess.run(['ffmpeg', '-i', wav_path, mp3_path])

    with open(mp3_path, 'rb') as f:
        mp3_data = f.read()

    # Clean up temporary files
    os.remove(wav_path)
    os.remove(mp3_path)
    if isinstance(input, bytes):
        os.remove(input_path)  # Remove the temporary file if input was bytes

    return BytesIO(mp3_data)


@app.post("/voice2voice", tags=["voice2voice"])
async def voice2voice(
    input_file: fastapi.UploadFile,
    model_name: str,
    index_path: str = None,
    f0up_key: int = 0,
    f0method: str = "crepe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    :param input_file: the .wav file to be converted
    :param model_name: the name of the model which was previously trained in the gui like the name in logs folder
    :param index_path: the index file of the previously trained model if none then use default dir logs by rvc.
    :param f0up_key: 0 or 1
    :param f0method: harvest, pm, crepe or rmvpe
    :param index_rate: 0.66
    :param device: if none then use default by rvc. cuda or cpu or specific cuda device "cuda:0", "cuda:1"
    :param is_half: False or True
    :param filter_radius: 3
    :param resample_sr: 0
    :param rms_mix_rate: 1
    :param protect: 0.33
    """
    audio_bytes = await input_file.read()

    kwargs = locals()
    kwargs["input"] = audio_bytes
    del kwargs["input_file"]

    # call the infer function
    try:
        mp3_io = infer(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(mp3_io, media_type="audio/mpeg", headers={"Content-Disposition": f"attachment; filename=rvc.mp3"})

@app.post("/voice2voice_local", tags=["voice2voice"])
async def voice2voice_local(
    input_path: str,
    model_name: str,
    index_path: str = None,
    opt_path: str = None,
    f0up_key: int = 0,
    f0method: str = "crepe",
    index_rate: float = 0.66,
    device: str = None,
    is_half: bool = False,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 1,
    protect: float = 0.33
):
    """
    :param input_path: the .wav file to be converted
    :param model_name: the name of the model which was previously trained in the gui like the name in logs folder
    :param index_path: the index file of the previously trained model if none then use default dir logs by rvc.
    :param opt_path: if not None then save the result to this path
    :param f0up_key: 0 or 1
    :param f0method: harvest, pm, crepe or rmvpe
    :param index_rate: 0.66
    :param device: if none then use default by rvc. cuda or cpu or specific cuda device "cuda:0", "cuda:1"
    :param is_half: False or True
    :param filter_radius: 3
    :param resample_sr: 0
    :param rms_mix_rate: 1
    :param protect: 0.33
    """
    kwargs = locals()

    # call the infer function
    try:
        mp3_io = infer(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # write to file if opt_path is provided
    if opt_path is not None:
        out_name = os.path.basename(opt_path)
        with open(opt_path, "wb") as f:
            f.write(mp3_io.getbuffer())
    else:
        out_name = os.path.basename("input_path")

    return StreamingResponse(mp3_io, media_type="audio/mpeg", headers={"Content-Disposition": f"attachment; filename={out_name}.mp3"})

@app.get("/status")
def status():
    return {"status": "ok"}

# create model cache
model_cache = ModelCache()
# start uvicorn server
uvicorn.run(app, host="0.0.0.0", port=8001)

