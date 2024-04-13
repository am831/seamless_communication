# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import Namespace
import torch
import typing as tp
import numpy as np
import warnings

SAMPLING_RATE = 16000

class SileroVADSegmenter:  # type: ignore
    def __init__(self, args: Namespace) -> None:
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        (self.get_speech_timestamps,
        self.save_audio,
        self.read_audio,
        self.VADIterator,
        self.collect_chunks) = utils

        self.sample_rate = getattr(args, "sample_rate", SAMPLING_RATE)
        self.chunk_size_sec = getattr(args, "chunk_size_sec", 10)
        self.pause_length = getattr(args, "pause_length", 0.5)

    def segment_long_input(self, audio: torch.Tensor) -> None:
        """
        Split long input into chunks
        """
        max_segment_length_samples = self.chunk_size_sec * self.sample_rate
        pause_length_samples = self.pause_length * self.sample_rate

        speech_timestamps = self.get_speech_timestamps(
            audio, self.model, sampling_rate=self.sample_rate
        )

        segments = []
        current_segment = []

        for segment in speech_timestamps:
            start_samples = segment[0]
            end_samples = segment[1]

            if current_segment and (
                end_samples - current_segment[0] > max_segment_length_samples
                or start_samples - current_segment[1] > pause_length_samples
            ):
                segments.append(current_segment)
                current_segment = []

            if not current_segment:
                current_segment = [start_samples, end_samples]
            else:
                current_segment[1] = end_samples
        if current_segment:
            segments.append(current_segment) 

        return segments
    
    def get_speech_timestamps(
        self,
        audio: torch.Tensor,
        model,
        threshold: float = 0.5,
        sampling_rate: int = SAMPLING_RATE,
        min_speech_duration_ms: int = 250,
        window_size_samples: int = 1536,
    ) -> tp.List[tp.Tuple[int, int]]:
        probs, _ = self.get_speech_probs(
            audio=audio,
            model=model,
            sampling_rate=sampling_rate,
            window_size_samples=window_size_samples,
        )

        max_segment_length_samples = self.chunk_size_sec * self.sample_rate
        min_segment_length_samples = min_speech_duration_ms / 1000 * sampling_rate

        segments = self.pdac(
            probs=probs,
            max_segment_length=max_segment_length_samples,
            min_segment_length=min_segment_length_samples,
            threshold=threshold,
            window_size_samples=window_size_samples,
        )

        speech_timestamps = [(seg.start, seg.end) for seg in segments]

        return speech_timestamps

    
    def pdac(
            self,
            probs: np.array, 
            max_segment_length: float, 
            min_segment_length: float, 
            threshold: float,
            window_size_samples: float
        ) -> tp.List[Segment]:
        """
        len(probs) may not work for sgm duration because it is for each window, not sample
        
        """
 
        segments = []
        sgm = Segment(0, len(probs)*window_size_samples, probs)
        sgm = self.trim(sgm, threshold)

        def recursive_split(sgm):
            if sgm.duration < max_segment_length:
                segments.append(sgm)
            else:
                j = 0
                sorted_indices = np.argsort(sgm.probs)
                while j < len(sorted_indices):
                    split_idx = sorted_indices[j]
                    sgm_a, sgm_b = self.split_and_trim(sgm, split_idx, threshold)
                    if (
                        sgm_a.duration > min_segment_length
                        and sgm_b.duration > min_segment_length
                    ):
                        recursive_split(sgm_a)
                        recursive_split(sgm_b)
                        break
                    j += 1
                else:
                    if sgm_a.duration > min_segment_length:
                        recursive_split(sgm_a)
                    if sgm_b.duration > min_segment_length:
                        recursive_split(sgm_b)

        recursive_split(sgm)
        return segments
    
    def trim(
            self,
            sgm: Segment, 
            threshold: float
        ) -> Segment:
        included_indices = np.where(sgm.probs >= threshold)[0]

        if not len(included_indices):
            return Segment(sgm.start, sgm.start, np.empty([0]))

        i = included_indices[0]
        j = included_indices[-1] + 1

        sgm = Segment(sgm.start + i, sgm.start + j, sgm.probs[i:j])

        return sgm

    def split_and_trim(
            self,
            sgm: Segment, 
            split_idx: int, 
            threshold: float
        ) -> tp.Tuple[Segment, Segment]:
        probs_a = sgm.probs[:split_idx]
        sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

        probs_b = sgm.probs[split_idx + 1 :]
        sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

        sgm_a = self.trim(sgm_a, threshold)
        sgm_b = self.trim(sgm_b, threshold)

        return sgm_a, sgm_b
    
    @staticmethod
    def get_speech_probs(
        audio: torch.Tensor,
        model,
        sampling_rate: int = SAMPLING_RATE,
        window_size_samples: int = 1536,
    ) -> tp.Tuple[np.ndarray, int]:
        """Get a list of speech probabilities computed with sliding window over the audio using the model."""
        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for _ in range(audio.ndim):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            assert (
                audio.ndim == 1
            ), "More than one dimension in audio. Are you trying to process audio with 2 channels?"

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn(
                "Sampling rate is a multiply of 16000, casting to 16000 manually!"
            )
        else:
            step = 1

        if sampling_rate == 8000 and window_size_samples > 768:
            warnings.warn(
                "window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!"
            )
        if window_size_samples not in [256, 512, 768, 1024, 1536]:
            warnings.warn(
                "Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate"
            )

        model.reset_states()

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            if next(model.parameters()).is_cuda:
                chunk = chunk.cuda()
            speech_prob = model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)

        return np.array(speech_probs), audio_length_samples
    
class Segment:
    def __init__(self, start: int, end: int, probs: np.ndarray):
        self.start = start
        self.end = end
        self.probs = probs
        self.duration = end - start
    