print("hello")
import torch

from seamless_communication.inference import Transcriber
print("check1")
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
print("check2")
transcriber = Transcriber (
    model_name,
    device=torch.device("cpu"),
    dtype=torch.float32,
)
print(transcriber.device)

input_audio = "en_example.wav"

txt = transcriber.transcribe(audio=input_audio, src_lang="eng")

print("Translated text: ", txt)
print()

