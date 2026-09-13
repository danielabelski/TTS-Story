# IndexTTS 2.5

IndexTTS runs locally in its own environment. Version 2.5 supports English, Chinese, Japanese, Spanish and Arabic; version 2 remains available for existing productions.

## Install and select

1. Open [Settings → IndexTTS](app:settings/index-tts) and install the engine. Existing installations need reinstalling to obtain the 2.5 runtime.
2. Select **IndexTTS-2.5**, choose the manuscript language, and save.
3. Assign a clean reference from the shared Voice Prompts library to each speaker.
4. Generate a short test before a full production.

The installer uses Python 3.11, the official dependency lock and a pinned upstream source revision in an isolated environment. Model files download on first use into separate version folders. Existing version-2 checkpoints are not overwritten by 2.5. No models or environments used by Breeze are changed.

## Emotional direction

Enable **Use passage directions for emotion** and start with **Emotion strength 0.6** or lower.

```text
[direction]Speak with restrained sadness.[/direction]
[narrator]The house was empty when she returned.[/narrator]
```

The direction is not spoken. IndexTTS translates it into its eight emotion dimensions: happy, angry, sad, afraid, disgusted, melancholic, surprised and calm. It is not an unrestricted acting prompt: precise timbre, pitch or accent instructions may not be reproduced. The reference sample supplies the voice identity; TTS-Story does not prepend Breeze's Voice Type instruction.

Zero strength or disabling the option bypasses text-emotion guidance. Without a direction, normal reference-based cloning is used. Repeated directions are cached within a batch and on disk for retries/resumes. The cache stores hashed instruction keys and emotion vectors under the model's checkpoint directory, automatically invalidates when the classifier files change, and applies the current emotion strength at synthesis time. The model stays loaded for the batch.

In **Library → Review Chunks**, edit **Voice Direction** and regenerate. The saved direction is also retained for bulk speaker regeneration and resumed jobs.

For IndexTTS chunks, **IndexTTS emotion strength** appears beneath Voice Direction. Set a value from 0 to 1 and click Regenerate. The value overrides the saved job strength for that chunk only, is saved after successful regeneration, and is reused for later chunk regenerations. Changing global engine settings does not replace this value. Existing chunks start with their saved job strength when available (older items without that information use 0.6). Zero disables guidance; a positive override enables it for the chunk. The control hides when another engine is selected; Breeze does not use this IndexTTS-specific setting. Recompile the chapter/full story afterward to include the updated chunk.

## Performance and compatibility

- **Device:** auto chooses an available accelerator. CUDA is strongly recommended for long jobs. Explicit CUDA requests fail clearly if CUDA is unavailable.
- **BF16:** used by 2.5 on supported GPUs; unsupported devices use full precision. FP16 is a separate version-2 option.
- **Beam width:** start at one.
- **Diffusion steps:** fixed at 25 upstream. The former editable control did not adjust the decoder and is now read-only.
- **Temperature, top-p, top-k, repetition penalty and token limits:** apply consistently to previews and batch generation.
- **DeepSpeed, Accel and torch.compile:** optional, off by default. They require additional compatible dependencies; do not enable all of them without a baseline test. Missing Flash Attention no longer disables ordinary half-precision inference.

Fresh chapter-split jobs use one worker across chapters when eligible. Resumed jobs use the safe section-rendering path to preserve completed filenames; this can reload the model between sections. Initial downloads, model loading and optional compilation are not representative of steady-state synthesis speed.

The worker reuses prepared voice-reference tensors for up to eight speakers, with a 128 MiB cache budget. Replacing a sample invalidates its cached conditioning. This cache lasts only for the worker's lifetime and does not retain models after a batch ends.

Terminal logs report emotion-model device placement, emotion decode time/token count, emotion cache hits, and synthesis time/reference cache hits separately. CPU/disk offloading of the emotion model is reported rather than forcing it onto a full GPU. Emotion decoding uses inference mode and a bounded output budget; truncated responses are retried once with a larger budget, then rejected instead of silently using incomplete directions. Unique directions are still classified individually; emotion batching is not enabled.

## Reference

[Official IndexTTS repository and usage](https://github.com/index-tts/index-tts)
