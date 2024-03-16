/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;
let imgNum = 0;
let hasTrained = false;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// Initialize empty arrays to store images for each label
const labeledImages = {
  0: [], // For label 0 (up)
  1: [], // For label 1 (down)
  2: [], // For label 2 (left)
  3: []  // For label 3 (right)
};

const lowConfidenceImages = [];
const confidenceThreshold = 0.4;
const lowConfidenceFrameBufferMax = 250;
let lowConfidenceFrameBuffer = 0;

const labelDict = {
  0: "UP",
  1: "DOWN",
  2: "LEFT",
  3: "RIGHT"
}

const trainedAmounts = {
  0: 0,
  1: 0,
  2: 0,
  3: 0
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.
ui.setExampleHandler(async label => {
  let img = await getImage();

  // Add the captured image to the corresponding labeledImages array
  labeledImages[label].push({ image: img, num: imgNum });
  imgNum++;

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // Draw the preview thumbnail.
  ui.drawThumb(img, label);
})

function decrementImgNums(fromNum) {
  for (const label in labeledImages) {
    for (let i = 0; i < labeledImages[label].length; i++) {
      if (labeledImages[label][i].num > fromNum) {
        labeledImages[label][i].num--;
      }
    }
  }
}

function addDeleteHandler(label, index, canvas) {
  // Add click event listener to canvas
  canvas.addEventListener('click', () => {
    let currentImgNum = labeledImages[label][index].num;
    console.log(currentImgNum);
    // Remove image from labeledImages array
    labeledImages[label].splice(index, 1);
    // Remove canvas element from DOM
    canvas.remove();
    // Display updated images
    displayImagesForLabel(label);
    // Remove example from dataset
    controllerDataset.removeExample(currentImgNum);
    // Change number under corresponding thumb image
    ui.decrementThumb(label);
    imgNum--;

    decrementImgNums(imgNum);
    //images[i].dispose();
  });
}

// Function to display images for a specific label
async function displayImagesForLabel(label) {
  const images = labeledImages[label];
  // const container = document.getElementById(`label-${label}-container`);
  const trainedContainer = document.getElementById(`label-${label}-trained`);
  const untrainedContainer = document.getElementById(`label-${label}-untrained`);

  // Clear previous canvases
  trainedContainer.innerHTML = 'Data used to train ' + labelDict[label];
  untrainedContainer.innerHTML = 'Data added to ' + labelDict[label] + ' since last training';

  // Create a div container for the grid layout
  const trainedGridContainer = document.createElement('div');
  trainedGridContainer.classList.add('image-grid');
  trainedContainer.appendChild(trainedGridContainer);

  const untrainedGridContainer = document.createElement('div');
  untrainedGridContainer.classList.add('image-grid');
  untrainedContainer.appendChild(untrainedGridContainer);
  
  // Iterate over images and display them on canvases
  for (let i = 0; i < images.length; i++) {
    const imgTensor = images[i].image;
    const canvas = document.createElement('canvas');

    if (i < trainedAmounts[label]) {
      trainedGridContainer.appendChild(canvas);
    } else {
      untrainedGridContainer.appendChild(canvas);
    }
    ui.draw(imgTensor, canvas);

    if (hasTrained == true) {
      const embeddings = truncatedMobileNet.predict(imgTensor);
      // Make a prediction through our newly-trained model using the embeddings
      // from mobilenet as input.
      const predictions = model.predict(embeddings);

      // depending on confidence of prediction,
      // color the outline of the image
      const confidence = predictions.as1D().dataSync()[label];
      const color = ui.getColorFromConfidence(confidence);
      canvas.style.borderColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    }
    
    addDeleteHandler(label, i, canvas);
  }
  updateTrainingDataIndicator();
}

const lowConfidenceContainer = document.getElementById('low-confidence-container');

async function showLowConfidenceImage() {
  // Remove current canvas (if there is one)
  lowConfidenceContainer.innerHTML = '';

  // If there is a lowConfidenceImage, draw it
  if (lowConfidenceImages.length > 0) {
    const canvas = document.createElement('canvas');
    lowConfidenceContainer.appendChild(canvas);
    canvas.style.transform = "scaleX(-1)";
    ui.draw(lowConfidenceImages[0], canvas);

    // When image is clicked, dispose of it and remove from array
    canvas.addEventListener('click', () => {
      lowConfidenceImages[0].dispose();
      lowConfidenceImages.shift();
      canvas.remove();
      showLowConfidenceImage();
    });
  }
}

// document.getElementById(`low-confidence`).addEventListener('click', async () => {
//   showLowConfidenceImage();
// });

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
  hasTrained = true;

  trainedAmounts[0] = labeledImages[0].length;
  trainedAmounts[1] = labeledImages[1].length;
  trainedAmounts[2] = labeledImages[2].length;
  trainedAmounts[3] = labeledImages[3].length;

  ui.hideAllGrids();
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    // Capture the frame from the webcam.
    const img = await getImage();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    const embeddings = truncatedMobileNet.predict(img);

    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    const predictions = model.predict(embeddings);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    const confidence = tf.clone(predictions).as1D().dataSync()[classId];
    
    // If low confidence and enough time has passed, add to low confidence array
    if (confidence < confidenceThreshold && lowConfidenceFrameBuffer >= lowConfidenceFrameBufferMax) {
      console.log("low confidence detected");
      lowConfidenceImages.unshift(img);
      lowConfidenceFrameBuffer = 0;
      showLowConfidenceImage();
    } else {
      img.dispose();
    }

    lowConfidenceFrameBuffer += 1;

    ui.predictClass(classId, confidence);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
  updateTrainingDataIndicator();
  clearLowConfidence();
  document.body.setAttribute('data-active', "none");
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

document.getElementById("toggle-button-0").addEventListener("click", function() {
  displayImagesForLabel(0); // Display images for label 0 (up)
})
document.getElementById("toggle-button-1").addEventListener("click", function() {
  displayImagesForLabel(1); // Display images for label 1 (down)
})
document.getElementById("toggle-button-2").addEventListener("click", function() {
  displayImagesForLabel(2); // Display images for label 2 (left)
})
document.getElementById("toggle-button-3").addEventListener("click", function() {
  displayImagesForLabel(3); // Display images for label 3 (right)
})

document.getElementById('up').addEventListener("click", function() {
  displayImagesForLabel(0); // Display images for label 0 (up)
})
document.getElementById('down').addEventListener("click", function() {
  displayImagesForLabel(1); // Display images for label 1 (down)
})
document.getElementById('left').addEventListener("click", function() {
  displayImagesForLabel(2); // Display images for label 2 (left)
})
document.getElementById('right').addEventListener("click", function() {
  displayImagesForLabel(3); // Display images for label 3 (right)
})

function clearLowConfidence() {
  // Dispose of all the low confidence images
  while (lowConfidenceImages.length > 0) {
    lowConfidenceImages[0].dispose();
    lowConfidenceImages.pop();
  }

  // Display nothing for the low confidence display
  showLowConfidenceImage();      
}

function moveLowConfidenceImage(toLabel) {
  if (lowConfidenceImages.length > 0) {
    labeledImages[toLabel].push({ image: lowConfidenceImages[0], num: imgNum });
    imgNum++;
    controllerDataset.addExample(truncatedMobileNet.predict(lowConfidenceImages[0]), toLabel);

    ui.incrementThumb(toLabel);
    lowConfidenceContainer.innerHTML = '';
    lowConfidenceImages.shift();
    showLowConfidenceImage();

    displayImagesForLabel(toLabel);
  }
}

document.getElementById('confidence-button-0').addEventListener("click", function() {
  moveLowConfidenceImage(0);
})
document.getElementById('confidence-button-1').addEventListener("click", function() {
  moveLowConfidenceImage(1);
})
document.getElementById('confidence-button-2').addEventListener("click", function() {
  moveLowConfidenceImage(2);
})
document.getElementById('confidence-button-3').addEventListener("click", function() {
  moveLowConfidenceImage(3);
})

const trainingDataIndicator = document.getElementById('training-data-indicator');

function updateTrainingDataIndicator() {
  // Count how many images have been trained on
  let trainedAmount = 0;
  for (const key in trainedAmounts) {
    trainedAmount += trainedAmounts[key];
  }

  // Compare to total number of images
  if (trainedAmount < imgNum) {
    // Show "you have images that haven't been trained on"
    trainingDataIndicator.style.display = "block";
  } else {
    trainingDataIndicator.style.display = "none";
  }
}

async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  ui.init();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

// Initialize the application.
init();
