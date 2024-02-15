# Llama2.c port to JavaScript (React Native)

This is a react native JavaScript version of the popular [llama2.c](https://github.com/karpathy/llama2.c) library by Andrej Karpathy.

the llamajs node support repo [llamajs](https://github.com/agershun/llamajs) 

Place the model file (e.g. [stories260k.bin]()) into the ```./assets/models``` directory.
```sh
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
```

Then run the LLM.

```sh
$ yarn install
$ npx expo prebuild
$ expo run:android
```

#### TODOS:

- [] runOnJs or 'worklet' for dont lock the mmain thread on inference time
- [] support for 8 bit quatization
- [] better UI
- [] Supprot native llama.c or cpp version of node if it possible

<img src="./assets/android.gif?raw=true" alt="result" style="width:300px;"/>

## License
MIT


