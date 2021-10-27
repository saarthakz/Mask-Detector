const fs = require('fs');
const tf = require("@tensorflow/tfjs-node");

(async () => {

  //Generating a consistent DataSet of 256x256
  const maskImageDataSetURL = "./DataSet/Mask";
  const noMaskImageDataSetURL = "./DataSet/No-Mask";

  const noMaskImageNewDataSetURL = "./NewDataSet/No-Mask";
  const maskImageNewDataSetURL = "./NewDataSet/Mask";

  const maskImageDataSet = fs.readdirSync(maskImageDataSetURL);
  const noMaskImageDataSet = fs.readdirSync(noMaskImageDataSetURL);

  // for (let i = 0;i < maskImageDataSet.length;i++) {
  //   let imageData = fs.readFileSync(maskImageDataSetURL + "/" + maskImageDataSet[i]);
  //   let imageTensor = tf.node.decodeJpeg(imageData, 3);
  //   let newImageTensor = tf.image.resizeBilinear(imageTensor, [50, 50]);
  //   let newImageData = await tf.node.encodePng(newImageTensor);
  //   fs.writeFileSync(`${maskImageNewDataSetURL}/${i}.jpg`, newImageData);
  // };

  // for (let i = 0;i < noMaskImageDataSet.length;i++) {
  //   let imageData = fs.readFileSync(noMaskImageDataSetURL + "/" + noMaskImageDataSet[i]);
  //   let imageTensor = tf.node.decodePng(imageData, 3);
  //   let newImageTensor = tf.image.resizeBilinear(imageTensor, [50, 50]);
  //   let newImageData = await tf.node.encodePng(newImageTensor);
  //   fs.writeFileSync(`${noMaskImageNewDataSetURL}/${i}.jpg`, newImageData);
  // };

  // Creating a model
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [50, 50, 3],
    activation: "relu",
    filters: 50,
    kernelSize: 3,
    batchSize: 2,
  }));
  model.add(tf.layers.maxPool2d({
    poolSize: 2,
    batchSize: 2,
  }));
  model.add(tf.layers.dense({
    activation: "relu",
    units: 24,
    batchSize: 2
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    activation: "softmax",
    units: 2,
    batchSize: 2,
  }));

  model.compile({
    loss: "meanSquaredError",
    optimizer: tf.train.sgd(0.2)
  });

  model.summary();
  console.log("Model Compiled");

  //Importing the new data
  const maskImageNewDataSet = fs.readdirSync(maskImageNewDataSetURL);
  const noMaskImageNewDataSet = fs.readdirSync(noMaskImageNewDataSetURL);

  let allImageTensors = [];
  let allImageLabelTensors = [];

  for (let i = 0;i < maskImageNewDataSet.length;i++) {
    let imageData = fs.readFileSync(maskImageNewDataSetURL + "/" + maskImageNewDataSet[i]);
    let imageTensor = tf.node.decodeJpeg(imageData, 3);
    let imageTensorArr = imageTensor.arraySync();
    imageTensorArr.forEach((pixelArr) => pixelArr.forEach((pixel) => pixel.forEach((color, index) => pixel[index] = color / 255)));
    imageTensor = tf.tensor3d(imageTensorArr);
    allImageTensors.push(imageTensor);
    allImageLabelTensors.push(1);
  };

  for (let i = 0;i < noMaskImageNewDataSet.length;i++) {
    let imageData = fs.readFileSync(noMaskImageNewDataSetURL + "/" + noMaskImageNewDataSet[i]);
    let imageTensor = tf.node.decodeJpeg(imageData, 3);
    let imageTensorArr = imageTensor.arraySync();
    imageTensorArr.forEach((pixelArr) => pixelArr.forEach((pixel) => pixel.forEach((color, index) => pixel[index] = color / 255)));
    imageTensor = tf.tensor3d(imageTensorArr);
    allImageTensors.push(imageTensor);
    allImageLabelTensors.push(0);
  };

  const xs = tf.stack(allImageTensors);
  const ys = tf.oneHot(tf.tensor1d(allImageLabelTensors, "int32"), 2);

  // Fitting the model
  await model.fit(xs, ys, {
    epochs: 100
  });

  console.log("Training complete");

  {
    const testData = fs.readFileSync(maskImageNewDataSetURL + "/" + maskImageNewDataSet[0]);
    let testImageTensor = tf.node.decodeJpeg(testData, 3);
    let testImageTensorArr = testImageTensor.arraySync();
    testImageTensorArr.forEach((pixelArr) => pixelArr.forEach((pixel) => pixel.forEach((color, index) => pixel[index] = color / 255)));
    testImageTensor = tf.tensor3d(testImageTensorArr);
    const prediction = await model.predict(tf.stack([testImageTensor]));
    prediction.print();
  }

  {
    const testData = fs.readFileSync(noMaskImageNewDataSetURL + "/" + noMaskImageNewDataSet[0]);
    let testImageTensor = tf.node.decodeJpeg(testData, 3);
    let testImageTensorArr = testImageTensor.arraySync();
    testImageTensorArr.forEach((pixelArr) => pixelArr.forEach((pixel) => pixel.forEach((color, index) => pixel[index] = color / 255)));
    testImageTensor = tf.tensor3d(testImageTensorArr);
    const prediction = await model.predict(tf.stack([testImageTensor]));
    prediction.print();
  }

})();