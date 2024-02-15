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
$ expo run:android #it also works with expo go app
```

#### TODOS:

- [ ] Using `runOnJs` or 'worklet' to avoid locking the main thread during inference time.
- [ ] Support for 8-bit quantization for speed (16 float point is default).
- [ ] Better UI support.
- [ ] Native `llama.c` or `.cpp` version of Node, if possible.
- [ ] Support for chat.


### Showing demo on Android and IOS
<p align="center">
<img src="./assets/android.gif?raw=true" alt="result" style="width:70%;"/>
<img src="./assets/ios.PNG?raw=true" alt="result" style="width:35%;"/>
</p>

## License
MIT


