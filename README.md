# ONNX ort model running in expo app (react-native)
```sh
$ yarn install
$ npx expo prebuild
$ expo run:android
```

### Model: MNIST
<img src="./assets/Screenshot_1701008874.png?raw=true" alt="result" style="width:300px;"/>

# Model conversion:
### [ONNX Github repo](https://github.com/microsoft/onnxruntime/tree/main/js/react_native/e2e/src)

`js/react_native/e2e/src/mnist.onnx` is `onnxruntime/test/testdata/mnist.onnx` updated to opset 15.

```bash
cd <repo root>/js
python -m onnxruntime.tools.update_onnx_opset --opset 15 ../onnxruntime/test/testdata/mnist.onnx ./react_native/e2e/src/mnist.onnx
```

`js/react_native/e2e/src/mnist.ort` and `js/react_native/e2e/android/app/src/main/assets/mnist.ort` are converted from `js/react_native/e2e/src/mnist.onnx`.

```bash
cd <repo root>/js
python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed --output_dir ./react_native/e2e/android/app/src/main/assets ./react_native/e2e/src/mnist.onnx
python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_style=Fixed --output_dir ./react_native/e2e/src ./react_native/e2e/src/mnist.onnx
```

# TODO for create empty expo project
## Steps of building up this project from scratch

### [Original Repo (old code)](https://github.com/fs-eire/ort-rn-hello-world)

1. Install onnxruntime-react-native
    ```sh
    $ yarn add onnxruntime-react-native
    $ npx expo install expo-asset
    ```

2. Add your ONNX model to project

    1. Put the file under `<SOURCE_ROOT>/assets`.
    
       In this tutorial, we use the ORT format ONNX model of MNIST (`mnist.ort`).

    2. add a new file `metro.config.js` under `<SOURCE_ROOT>` and add the following lines to the file:
       ```js
       const { getDefaultConfig } = require('@expo/metro-config');
       const defaultConfig = getDefaultConfig(__dirname);
       defaultConfig.resolver.assetExts.push('ort');
       module.exports = defaultConfig;
       ```

       This step adds extension `ort` to the bundler's asset extension list, which allows the bundler to include the model into assets.

    **NOTE:**
    - There are multiple ways to load model using ONNX Runtime for React Native. In this tutorial, model is built into the app as an asset.
    - It's required to use a ORT format model (ie. a model file with `.ort` extension)

3. Setup Android and iOS project.

    In this step, we setup the generated Android/iOS project to consume ONNX Runtime. There are 2 ways to setup the project:

    - (recommended) using NPM package `onnxruntime-react-native` as an expo plugin.
        1. In `<SOURCE_ROOT>/app.json`, add the following line to section `expo`:
           ```
           "plugins": ["onnxruntime-react-native"],
           ```
        2. Run the following command in `<SOURCE_ROOT>` to generate Android and iOS project:
            ```sh
            expo prebuild
            ```

        The generated project files will be updated automatically.

    - setup manually.

        1. Run the following command in `<SOURCE_ROOT>` to generate Android and iOS project:
            ```sh
            expo prebuild
            ```

            **NOTE:**
            - Expo will ask the Android package name and iOS bundle identifier. In this tutorial we use `com.hootan09.expo_app_ai` as package name and bundle identifier.
            - The package name (Android) and bundle ID (iOS) will be added in your `<SOURCE_ROOT>/app.json` automatically by expo.

        2. Add `onnxruntime-react-native` to gradle depencencies.

            In `<SOURCE_ROOT>/android/app/build.gradle`, add the following line to section `dependencies`:
            ```
            implementation project(':onnxruntime-react-native')
            ```

        3. Add `onnxruntime-react-native` to CocoaPods dependencies.

            In `<SOURCE_ROOT>/ios/Podfile`, add the following line to section `target 'expo_app_ai'`:
            ```
            pod 'onnxruntime-react-native', :path => '../node_modules/onnxruntime-react-native'
            ```

            Run the following command in `<SOURCE_ROOT>/ios` to install:
            ```sh
            pod install
            ```