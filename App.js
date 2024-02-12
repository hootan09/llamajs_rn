import { StatusBar } from 'expo-status-bar';
import { Alert, Button, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';

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
  const [temperature, setTemperature] = useState(1.0);
  const [topp, setTopp] = useState(0.9);
  const [steps, setSteps] = useState(256);
  const [prompt, setPrompt] = useState("Once upon a time ");

  const timeout = async(delay)=> {
    return new Promise( res => setTimeout(res, delay) );
  }

  const loadModel = async()=>{
    try {
      const assets = await Asset.loadAsync([require('./assets/models/stories260K.bin'),require('./assets/models/tok512.bin')]);
      setUris({modelUri: assets[0]?.localUri, tokenizerUri: assets[1]?.localUri});
      // console.log(assets);
      if(!assets[0]?.localUri && !assets[1]?.localUri) {
        // Alert.alert('failed to get model URI', `${assets[0]}`);
        console.log('Failed to get model & tokenizer URI', `${assets[0]}`);
        setModelStatus(`Failed to get model & tokenizer`)
      } else {
        console.log('Models & Tokenizer loaded!')
        setModelStatus(`Model & tokenizer loaded!`)
        // Alert.alert(
        //   'model loaded successfully',
        //   );
      }
    } catch (e) {
      // Alert.alert('failed to load model', `${e}`);
      console.log('Failed to load model', `${e}`);
      setModelStatus(`Failed to load model`)
      throw e;
    }
  }

  const runModel = async()=> {
    setRunResult('')
    try {
      let checkpointPath = uris.modelUri
      let tokenizerPath = uris.tokenizerUri;
      // let temperature = 1.0;
      // let topp = 0.9;
      // let steps = 256;
      // let prompt = "";
      let rngSeed = 133742;
      let mode = "generate";
  
      const transformer = {};
      await buildTransformer(transformer, checkpointPath);
    
      const tokenizer = {};
      const vocabSize = transformer.config.vocabSize;
      await buildTokenizer(tokenizer, tokenizerPath, vocabSize);
    
      const sampler = buildSampler(vocabSize, temperature, topp, rngSeed);
    
      if (mode === "generate") {
        generate(transformer, tokenizer, sampler, prompt, steps, async(text)=> {
          await timeout(0.000001)
          setRunResult((oldText)=> {
            // console.log(newText);
            return oldText + text;
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
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Using LlamaJs for React Native</Text>

      <View style={styles.loadModelWrapper}>
        <Text style={styles.modetStatusText}>Temperature:</Text>
        <TextInput style={[styles.inputBox,{ marginLeft: 5}]} defaultValue={temperature.toString()} value={temperature.toString()} keyboardType='numeric' onChangeText={(value)=> setTemperature(+value)}/>
      </View>

      <View style={styles.loadModelWrapper}>
        <Text style={styles.modetStatusText}>Top:</Text>
        <TextInput style={[styles.inputBox,{ marginLeft: 60}]} defaultValue={topp.toString()} value={topp.toString()} keyboardType='numeric' onChangeText={(value)=> setTopp(+value)}/>
      </View>

      <View style={styles.loadModelWrapper}>
        <Text style={styles.modetStatusText}>Steps:</Text>
        <TextInput style={[styles.inputBox,{ marginLeft: 50}]} defaultValue={steps.toString()} value={steps.toString()} keyboardType='numeric' onChangeText={(value)=> setSteps(+value)}/>
      </View>

      <View style={styles.loadModelWrapper}>
      <Text style={styles.modetStatusText}>Prompt:</Text>
        <TextInput style={[styles.inputBox,{ marginLeft: 40, width: 200}]} defaultValue={prompt.toString()} value={prompt.toString()} keyboardType='default' onChangeText={(value)=> setPrompt(value)}/>
      </View>

      <View style={styles.loadModelWrapper}>
        <Button title='Load Model' onPress={loadModel}></Button>
        <Text style={styles.modetStatusText}>Status: {modelStatus}</Text>
      </View>
      {/* <View style={{padding: 20}}/> */}

      <View style={styles.loadModelWrapper}>
        <Button title='Run Inference' onPress={runModel}></Button>
      </View>

      <ScrollView contentContainerStyle={{}}>
        <Text style={styles.runResultText}> {'Result:\n\n'+ runResult}</Text>
      </ScrollView>
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginTop: 25,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'flex-start',
    marginHorizontal: 10,
  },
  loadModelWrapper: {
    width: '100%',
    paddingHorizontal: 10,
    // paddingVertical: 5,
    // borderRadius: 15,
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    backgroundColor: '#EBECED'
  },
  inputBox: {
    borderWidth: 2,
    borderColor: 'black',
    width: 100,
    borderRadius: 10,
    paddingHorizontal: 10,
  },
  title: {
    fontSize: 16,
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
