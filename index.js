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

  for (let i = 0;i < maskImageDataSet.length;i++) {
    let imageData = fs.readFileSync(maskImageDataSetURL + "/" + maskImageDataSet[i]);
    let imageTensor = tf.node.decodeJpeg(imageData, 3);
    let newImageTensor = tf.image.resizeBilinear(imageTensor, [50, 50]);
    let newImageData = await tf.node.encodePng(newImageTensor);
    fs.writeFileSync(`${maskImageNewDataSetURL}/${i}.jpg`, newImageData);
  };

  for (let i = 0;i < noMaskImageDataSet.length;i++) {
    let imageData = fs.readFileSync(noMaskImageDataSetURL + "/" + noMaskImageDataSet[i]);
    let imageTensor = tf.node.decodePng(imageData, 3);
    let newImageTensor = tf.image.resizeBilinear(imageTensor, [50, 50]);
    let newImageData = await tf.node.encodePng(newImageTensor);
    fs.writeFileSync(`${noMaskImageNewDataSetURL}/${i}.jpg`, newImageData);
  };

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
    optimizer: "sgd"
  });
  console.log("Model Compiled");

  //Importing the new data
  const maskImageNewDataSet = fs.readdirSync(maskImageNewDataSetURL);
  const noMaskImageNewDataSet = fs.readdirSync(noMaskImageNewDataSetURL);

  //Generating Mask image tensors and labels
  let maskImageTensorsArr = [];
  let maskImageLabelsArr = [];
  for (let i = 0;i < maskImageNewDataSet.length;i++) {
    let imageData = fs.readFileSync(maskImageNewDataSetURL + "/" + maskImageNewDataSet[i]);
    let imageTensor = tf.node.decodeJpeg(imageData, 3);
    maskImageTensorsArr.push(imageTensor);
    maskImageLabelsArr.push(1);
  };
  const maskImageTensor = tf.stack(maskImageTensorsArr); //4D Tensor of Input images ( With Mask )
  const maskImageLabelTensor = tf.oneHot(tf.tensor1d(maskImageLabelsArr, "int32"), 2);

  //Generating No Mask image tensors and labels
  let noMaskImageTensorsArr = [];
  let noMaskImageLabelsArr = [];
  for (let i = 0;i < noMaskImageNewDataSet.length;i++) {
    let imageData = fs.readFileSync(noMaskImageNewDataSetURL + "/" + noMaskImageNewDataSet[i]);
    let imageTensor = tf.node.decodeJpeg(imageData, 3);
    noMaskImageTensorsArr.push(imageTensor);
    noMaskImageLabelsArr.push(0);
  };
  const noMaskImageTensor = tf.stack(noMaskImageTensorsArr); //4D Tensor of Input images ( Without Mask )
  const noMaskImageLabelTensor = tf.oneHot(tf.tensor1d(noMaskImageLabelsArr, "int32"), 2);

  //Fitting the model
  await model.fit(maskImageTensor, maskImageLabelTensor, {
    epochs: 100
  });

  await model.fit(noMaskImageTensor, noMaskImageLabelTensor, {
    epochs: 100
  });

  console.log("Training complete");

})();