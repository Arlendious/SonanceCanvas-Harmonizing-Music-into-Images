# Music-to-Image Demo

This demo showcases a pipeline that generates an image based on an input audio file. The pipeline consists of several steps:

1. The audio file is passed through LP-Music-Caps, a model that generates a descriptive caption of the music.
2. If the audio contains lyrics, they are extracted using a separate model.
3. The music description and lyrics (if available) are sent to Llama2, a large language model that generates an image description based on the input text.
4. The image description is then used as a prompt for Stable Diffusion XL to generate the final image.

## Usage

To use the demo, simply upload an audio file and provide a title for the song. You can also specify whether the audio contains lyrics. The demo will then generate an image that visually represents the music. You can also export a video visualizer for the audio with the generated image.

Note that only the first 30 seconds of the audio will be used for inference.

## Demo Video
## Example Video

[![Demo](https://img.youtube.com/vi/okOyY12WaRI/0.jpg)](https://www.youtube.com/watch?v=okOyY12WaRI)


## Requirements

The demo requires the following dependencies:

- gradio
- huggingface
- pydub
- compel
- diffusers
- torch

Make sure to install these packages before running the demo.

## Running the Demo

To run the demo locally, clone the repository and run the following command:

```
python music_to_image.py
```

This will launch the Gradio interface where you can upload audio files and generate images.

## Credits

This demo was created using models and tools from the Hugging Face ecosystem. Special thanks to the following projects:

- LP-Music-Caps: https://huggingface.co/spaces/seungheondoh/LP-Music-Caps-demo
- Llama2: https://huggingface.co/docs/llama/
- Stable Diffusion XL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

Feel free to experiment with the demo and share your results!
