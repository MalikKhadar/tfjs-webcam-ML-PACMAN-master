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

const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = [38, 40, 37, 39];

export function init() {
  document.getElementById('controller').style.display = '';
  statusElement.style.display = 'none';
}

const trainStatusElement = document.getElementById('train-status');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
export const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
export const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
export const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

export function startPacman() {
  google.pacman.startGameplay();
}

const colorMap = [
  // { confidence: 0.0, color: [0, 0, 0] }, // black
  // { confidence: 0.5, color: [0, 0, 255] }, // blue
  // { confidence: 0.9, color: [0, 255, 255] }, // cyan
  // { confidence: 1.0, color: [255, 255, 0] }  // yellow
  { confidence: 0.0, color: [255, 255, 255] }, // white
  { confidence: 0.9, color: [220, 255, 220] }, // light green
  { confidence: 1, color: [0, 255, 0] }, // green
];

function getContrastTextColor(color) {
  // Convert the RGB color to perceived brightness (based on the YIQ color space)
  const brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000;

  // Choose either black or white text color based on the perceived brightness
  return brightness > 128 ? [0, 0, 0] : [255, 255, 255];
}

// Get the color key container element
const colorKeyContainer = document.getElementById('color-key');

// Iterate over the color map array and construct the key entries
colorMap.forEach(entry => {
  const colorEntry = document.createElement('div');
  const textColor = getContrastTextColor(entry.color); // Calculate contrast text color
  colorEntry.style.backgroundColor = `rgb(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
  colorEntry.style.color = `rgb(${textColor[0]}, ${textColor[1]}, ${textColor[2]})`; // Apply text color
  colorEntry.textContent = `Confidence ${entry.confidence}: RGB(${entry.color[0]}, ${entry.color[1]}, ${entry.color[2]})`;
  colorEntry.style.padding = "5px"; /* Add padding for spacing */
  colorKeyContainer.appendChild(colorEntry);
});

export function getColorFromConfidence(confidence) {
  // Sort the color map by confidence values
  colorMap.sort((a, b) => a.confidence - b.confidence);

  // Find the nearest confidence values in the color map based on the given confidence
  let lowerColorIndex = 0;
  let upperColorIndex = colorMap.length - 1;
  for (let i = 0; i < colorMap.length; i++) {
    if (colorMap[i].confidence <= confidence) {
      lowerColorIndex = i;
    }
    if (colorMap[i].confidence >= confidence) {
      upperColorIndex = i;
      break;
    }
  }

  // Calculate the interpolation weight based on the distance between confidence values
  const lowerConfidence = colorMap[lowerColorIndex].confidence;
  const upperConfidence = colorMap[upperColorIndex].confidence;
  const weight = (confidence - lowerConfidence) / (upperConfidence - lowerConfidence);

  // Interpolate between the colors
  const lowerColor = colorMap[lowerColorIndex].color;
  const upperColor = colorMap[upperColorIndex].color;
  return interpolateColor(lowerColor, upperColor, weight);
}

function interpolateColor(color1, color2, weight) {
  // Interpolate between the RGB components
  const interpolatedRgb = [
    Math.round(color1[0] + (color2[0] - color1[0]) * weight),
    Math.round(color1[1] + (color2[1] - color1[1]) * weight),
    Math.round(color1[2] + (color2[2] - color1[2]) * weight)
  ];

  return interpolatedRgb;
}

const confidenceBars = [
  document.getElementById("confidence-up"),
  document.getElementById("confidence-down"),
  document.getElementById("confidence-left"),
  document.getElementById("confidence-right")
]

function showConfidenceBar(classId) {
  for (let i = 0; i < confidenceBars.length; i++) {
    if (i == classId) {
      confidenceBars[i].style.opacity = "1";
    } else {
      confidenceBars[i].style.opacity = "0";
    }
  }
}

export function predictClass(classId, confidence) {
  google.pacman.keyPressed(CONTROL_CODES[classId]);
  document.body.setAttribute('data-active', CONTROLS[classId]);

  const color = getColorFromConfidence(confidence);
  document.documentElement.style.setProperty('--custom-color', `rgb(${color[0]}, ${color[1]}, ${color[2]})`);

  showConfidenceBar(classId);
}

export function isPredicting() {
  statusElement.style.visibility = 'visible';
}
export function donePredicting() {
  statusElement.style.visibility = 'hidden';
}
export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

export let addExampleHandler;
export function setExampleHandler(handler) {
  addExampleHandler = handler;
}
let mouseDown = false;
const totals = [0, 0, 0, 0];

const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

const thumbDisplayed = {};

async function handler(label) {
  mouseDown = true;
  const className = CONTROLS[label];
  const button = document.getElementById(className);
  const total = document.getElementById(className + '-total');
  if (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute('data-active', CONTROLS[label]);
    total.innerText = ++totals[label];
    await tf.nextFrame();
  }
  document.body.removeAttribute('data-active');
}

export function decrementThumb(label) {
  const className = CONTROLS[label];
  const total = document.getElementById(className + '-total');
  total.innerText = --totals[label];
}

upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

export function hideAllGrids() {
  var imageContainers = document.getElementsByClassName("image-container");

  for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].style.display = "none";
  }
}

document.getElementById("toggle-button-0").addEventListener("click", function() {
  var grid = document.getElementById("label-0-container");
  if (grid.style.display !== "block") {
    hideAllGrids();
    grid.style.display = "block";
  } else {
    hideAllGrids();
  }
})

document.getElementById("toggle-button-1").addEventListener("click", function() {
  var grid = document.getElementById("label-1-container");
  if (grid.style.display !== "block") {
    hideAllGrids();
    grid.style.display = "block";
  } else {
    hideAllGrids();
  }
})

document.getElementById("toggle-button-2").addEventListener("click", function() {
  var grid = document.getElementById("label-2-container");
  if (grid.style.display !== "block") {
    hideAllGrids();
    grid.style.display = "block";
  } else {
    hideAllGrids();
  }
})

document.getElementById("toggle-button-3").addEventListener("click", function() {
  var grid = document.getElementById("label-3-container");
  if (grid.style.display !== "block") {
    hideAllGrids();
    grid.style.display = "block";
  } else {
    hideAllGrids();
  }
})

export function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    draw(img, thumbCanvas);
  }
}

export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
