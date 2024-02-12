import { StatusBar } from 'expo-status-bar';
import { Alert, Button, ScrollView, StyleSheet, Text, View } from 'react-native';

import { Asset } from 'expo-asset';
import { useState } from 'react';
// import * as FileSystem from 'expo-file-system';
// import { Buffer } from 'buffer';
import {
  buildTransformer,
  buildTokenizer,
  buildSampler,
  generate,
} from './llamajs/core';

export default function App() {
  
  const [modelStatus, setModelStatus] = useState('');
  const [runResult, setRunResult] = useState('');
  const [uris, setUris] = useState({
    modelUri: "",
    tokenizerUri: ""
  })

  const loadModel = async()=>{
    try {
      const assets = await Asset.loadAsync([require('./assets/models/stories260K.bin'),require('./assets/models/tok512.bin')]);
      setUris({modelUri: assets[0]?.localUri, tokenizerUri: assets[1]?.localUri});
      // console.log(assets);
      if(!assets[0]?.localUri && !assets[1]?.localUri) {
        // Alert.alert('failed to get model URI', `${assets[0]}`);
        console.log('Failed to get model & tokenizer URI', `${assets[0]}`);
        setModelStatus(`Failed to get model & tokenizer URI ${assets[0]}`)
      } else {
        console.log('Models & Tokenizer loaded!')
        setModelStatus(`Models & Tokenizer loaded!`)
        // Alert.alert(
        //   'model loaded successfully',
        //   );
      }
    } catch (e) {
      // Alert.alert('failed to load model', `${e}`);
      console.log('Failed to load model', `${e}`);
      setModelStatus(`Failed to load model ${e}`)
      throw e;
    }
  }

  const runModel = async()=> {
    try {
      let checkpointPath = uris.modelUri
      let tokenizerPath = uris.tokenizerUri;
      let temperature = 1.0;
      let topp = 0.9;
      let steps = 256;
      let prompt = "";
      let rngSeed = 133742;
      let mode = "generate";
  
      const transformer = {};
      await buildTransformer(transformer, checkpointPath);
    
      const tokenizer = {};
      const vocabSize = transformer.config.vocabSize;
      await buildTokenizer(tokenizer, tokenizerPath, vocabSize);
    
      const sampler = buildSampler(vocabSize, temperature, topp, rngSeed);
    
      if (mode === "generate") {
        generate(transformer, tokenizer, sampler, prompt, steps, (text)=> {
          setRunResult((oldText)=> {
            let newText = oldText + text
            console.log(newText);
            return newText;
          })
        });
      } 
      // else if (mode === "chat") {
      //   await chat(transformer, tokenizer, sampler, prompt, null, steps);
      // }

    } catch (e) {
      console.log(e);
      setRunResult(`failed to inference model ${e}`);
      // Alert.alert('failed to inference model', `${e}`);
      throw e;
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Using LlamaJs for React Native</Text>
      <Button title='Load model' onPress={loadModel}></Button>
      <Text style={styles.modetStatusText}>Model Load Status: {modelStatus}</Text>
      <View style={{padding: 20}}/>
      <Button title='Run Inference' onPress={runModel}></Button>
      <ScrollView contentContainerStyle={{}}>
        <Text style={styles.runResultText}> {'Result:\n\n'+ runResult}</Text>
      </ScrollView>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: 50,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'flex-start',
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    marginBottom: 5,
  },
  modetStatusText: {
    margin: 25,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 25,
    fontStyle: 'italic',
  },
  runResultText: {
    margin: 25,
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 25,
  }
});
