var fs = require('fs')

// Formatt CSV to JSON function.
function formatCSV(filePath) {
  var dataArray = [];
  var fileContents = fs.readFileSync(filePath);
  var lines = fileContents.toString().split('\n');
  // split lines from original string
  for (var i = 1; i < lines.length; i++) {
	  dataArray.push(lines[i].toString().split(','));
  }

  // need to remove row delimitter
  for (let i = 0; i < dataArray.length; i++) {
    dataArray[i][8] = dataArray[i][8].replace(/[\r\n]/g, "")
  }

  // parse each data point to an integer
  for (let i = 0; i < dataArray.length; i++) {
    for(let j = 0; j < dataArray[i].length; j++) {
      dataArray[i][j] = parseInt(dataArray[i][j])
    }
  }
  return dataArray
}

// Splits data into training and testing sets, randomly selecting training entries.
function split(baselineData) {
  const trainingLength = Math.round((baselineData.length/100)*70)
  let trainingData = []
  let trainingLabels = []
  let testingLabels = []
  // make training set, by pulling random indexs from the baselineData.
  // this would leave random data left in the test data.
  for (i = 0; i < trainingLength; i++) {
    trainingData.push(baselineData.splice(Math.round(Math.random()*10), 1)[0])
  }
  // get training labels by removing from traning data.
  for (i = 0; i < trainingData.length; i++) {
    trainingLabels.push(trainingData[i].splice(8, 1)[0])
  }
  // get testing labels by removing from testing data.
  for (i = 0; i < baselineData.length; i++) {
    testingLabels.push(baselineData[i].splice(8, 1)[0])
  }
  return {
    trainingData,
    testingData: baselineData,
    trainingLabels,
    testingLabels
  }
}

// method to call to measure how accurate on your Knn algo.
function evaluator(csvPath, algo) {
  // read data from csv and format
  baseData = formatCSV(csvPath)
  // split data
  let { trainingData, testingData, testingLabels, trainingLabels } = split(baseData)

  algo.fit(trainingData, trainingLabels)

  const predictions = algo.predict(testingData)

  return evaluate(predictions, testingLabels)
}

// evaluates predicted results of testing vs traning data, returning 4 different types of accuracy/precision
// metrics.
function evaluate(predictions, labels) {
  let truePos = 0, falseNeg = 0, falsePos = 0, trueNeg = 0
  for (i = 0; i < labels.length; i++) {
    if(labels[i] === 1) {
      if(predictions[i][0] === 1) {
        truePos++
      } else {
        falseNeg++
      }
    } else {
      if (predictions[i][0] === 1) {
        falsePos++
      } else {
        trueNeg++
      }
    }
  }

  const accuracy = (trueNeg + truePos) / labels.length
  const precision = truePos / (truePos + falsePos)
  const recall = truePos / (truePos + falseNeg)
  const f1 = 2 * ((precision * recall) / (precision + recall))

  return {
    accuracy: accuracy,
    precision: precision,
    recall: recall,
    f1: f1
  }
}

module.exports = {
  evaluator: evaluator
}