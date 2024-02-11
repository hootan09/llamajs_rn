import { StatusBar } from 'expo-status-bar';
import { Alert, Button, StyleSheet, Text, View } from 'react-native';

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { useState } from 'react';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';

export default function App() {
  
  const [model, setModel] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [modelStatus, setModelStatus] = useState('');
  const [runResult, setRunResult] = useState('');

  const loadModel = async()=>{
    try {
      const assets = await Asset.loadAsync([require('./assets/models/mnist.ort'),require('./assets/3.jpg')]);

      const modelUri = assets[0]?.localUri;

      const imageUri = assets[1]?.localUri;
      if(imageUri){
        const base64 = await FileSystem.readAsStringAsync(imageUri, { encoding: 'base64' });
        setImageBase64(base64)
      }

      if (!modelUri) {
        // Alert.alert('failed to get model URI', `${assets[0]}`);
        console.log('Failed to get model URI', `${assets[0]}`);
        setModelStatus(`Failed to get model URI ${assets[0]}`)
        setModel(null);
      } else {
        let myModel = await ort.InferenceSession.create(modelUri);
        console.log('Model is loaded: ',`input names: ${myModel.inputNames}, output names: ${myModel.outputNames}`);
        setModelStatus(`Model is loaded: input names: ${myModel.inputNames}, output names: ${myModel.outputNames}`)
        // Alert.alert(
        //   'model loaded successfully',
        //   );
        setModel(myModel);
      }
    } catch (e) {
      // Alert.alert('failed to load model', `${e}`);
      console.log('Failed to load model', `${e}`);
      setModelStatus(`Failed to load model ${e}`)
      setModel(null);
      throw e;
    }
  }

  const runModel = async()=> {
    try {
      // const inputData = new Float32Array(28 * 28);
      // const feeds = {};
      // feeds[model.inputNames[0]] = new ort.Tensor(inputData, [1,1, 28, 28]);
      // const fetches = await model.run(feeds);
      // const output = fetches[model.outputNames[0]];
      
      const bytes = Buffer.from(imageBase64, 'base64');
      const inputData = new Float32Array(784);
      for (let i = 0, len = bytes.length; i < len; i += 4) {
        inputData[i / 4] = bytes[i + 3] / 255;
      }
      const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
      // prepare feeds. use model input names as keys.
      const feeds = { Input3: inputTensor };
      const fetches = await model.run(feeds);
      const output = fetches[model.outputNames[0]];
      
      if (!output) {
        // Alert.alert('failed to get output', `${model.outputNames[0]}`);
        console.log('Failed to get output', `${model.outputNames[0]}`);
        setRunResult(`Failed to get output ${model.outputNames[0]}`);
      } else {
        // Alert.alert(
          //   'model inference successfully',
          //   `output shape: ${output.dims}, output data: ${output.data}`);
        console.log(getPredictedClass(output?.data));
        console.log('Model inference successfully',`output shape: ${output.dims}, output data: ${output.data}`);
        setRunResult(`Model inference successfully output shape: ${output.dims}, output data: ${output.data}`)
      }
    } catch (e) {
      console.log(e);
      setRunResult(`failed to inference model ${e}`);
      // Alert.alert('failed to inference model', `${e}`);
      throw e;
    }
  }

  const getPredictedClass = (output) =>  {
    if (output.reduce((a, b) => a + b, 0) === 0) { 
      return -1;
    }
    return output.reduce((argmax, n, i) => (n > output[argmax] ? i : argmax), 0);
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Using ONNX Runtime for React Native</Text>
      <Button title='Load model' onPress={loadModel}></Button>
      <View style={{padding: 20}}/>
      <Button title='Run' onPress={runModel}></Button>
      <Text style={styles.modetStatusText}>Model Load Status: {modelStatus}</Text>
      <Text style={styles.runResultText}>Run Result: {runResult}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 5,
  },
  modetStatusText: {
    margin: 25,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 25,
  },
  runResultText: {
    margin: 25,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 25,
  }
});
