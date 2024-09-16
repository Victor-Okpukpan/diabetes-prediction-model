import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import PRROCChart from "./PRROCChart"; // Import the new component

export default function Home() {
  const [model, setModel] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [inputData, setInputData] = useState({
    pregnancies: 0,
    glucose: 0,
    bloodPressure: 0,
    skinThickness: 0,
    insulin: 0,
    bmi: 0,
    diabetesPedigreeFunction: 0,
    age: 0,
  });
  const [prediction, setPrediction] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [prData, setPrData] = useState([]);
  const [rocData, setRocData] = useState([]);
  const [showModal, setShowModal] = useState(false);

  console.log(rocData, prData);

  const loadCSV = async () => {
    const response = await fetch("/diabetes.csv");
    const csvData = await response.text();
    return new Promise((resolve) => {
      Papa.parse(csvData, {
        header: true,
        dynamicTyping: true,
        complete: (results) => resolve(results.data),
      });
    });
  };

  const preprocessData = (data) => {
    const inputs = data.map((row) => [
      row.Pregnancies,
      row.Glucose,
      row.BloodPressure,
      row.SkinThickness,
      row.Insulin,
      row.BMI,
      row.DiabetesPedigreeFunction,
      row.Age,
    ]);
    const labels = data.map((row) => row.Outcome);
    const inputTensor = tf.tensor2d(inputs);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    return { inputTensor, labelTensor };
  };

  const calculatePRAndROC = (trueLabels, predictedScores) => {
    const thresholds = Array.from(Array(100).keys()).map((i) => i / 100);
    const prData = [];
    const rocData = [];

    thresholds.forEach((threshold) => {
      let tp = 0,
        fp = 0,
        tn = 0,
        fn = 0;

      trueLabels.forEach((trueLabel, index) => {
        const predictedLabel = predictedScores[index] >= threshold ? 1 : 0;
        if (predictedLabel === 1 && trueLabel === 1) {
          tp += 1;
        } else if (predictedLabel === 1 && trueLabel === 0) {
          fp += 1;
        } else if (predictedLabel === 0 && trueLabel === 0) {
          tn += 1;
        } else if (predictedLabel === 0 && trueLabel === 1) {
          fn += 1;
        }
      });

      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const tpr = tp / (tp + fn) || 0; // True Positive Rate
      const fpr = fp / (fp + tn) || 0; // False Positive Rate

      prData.push({ precision, recall });
      rocData.push({ tpr, fpr });
    });

    return { prData, rocData };
  };

  const trainModel = async () => {
    setIsTraining(true);
    const data = await loadCSV();
    const { inputTensor, labelTensor } = preprocessData(data);

    const model = tf.sequential();
    model.add(
      tf.layers.dense({ units: 8, inputShape: [8], activation: "relu" })
    );
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    const history = await model.fit(inputTensor, labelTensor, {
      epochs: 100,
      validationSplit: 0.2,
      shuffle: true,
    });

    const predictedScores = model.predict(inputTensor).dataSync();
    const trueLabels = await labelTensor.data();

    // Calculate PR and ROC values
    const { prData, rocData } = calculatePRAndROC(trueLabels, predictedScores);
    setPrData(prData);
    setRocData(rocData);

    setAccuracy(history.history.acc.pop());
    setModel(model);
    setIsTraining(false);

    inputTensor.dispose();
    labelTensor.dispose();
  };

  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!model) {
      alert("Please train the model first!");
      return;
    }
    const inputTensor = tf.tensor2d([Object.values(inputData)]);
    const output = model.predict(inputTensor);
    const result = await output.data();
    setPrediction(result[0] >= 0.5 ? "Positive" : "Negative");
    setShowModal(true);
  };

  return (
    <div className="flex flex-col text-black items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4">Diabetes Prediction System</h1>
      <button
        onClick={trainModel}
        className={`bg-blue-500 text-white px-4 py-2 rounded mb-4 ${
          isTraining ? "opacity-50 cursor-not-allowed" : ""
        }`}
        disabled={isTraining}
      >
        {isTraining ? "Training Model..." : "Train Model"}
      </button>
      {accuracy && !isTraining && (
        <h2 className="text-lg font-medium text-green-500 mb-4">
          Model Accuracy: {(accuracy * 100).toFixed(2)}%
        </h2>
      )}
      {prData.length > 0 && rocData.length > 0 && (
        <PRROCChart prData={prData} rocData={rocData} />
      )}
      <form
        onSubmit={handleSubmit}
        className="bg-white p-6 rounded shadow-md w-full max-w-md"
      >
        <h2 className="text-xl font-semibold mb-4">Input Data</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.keys(inputData).map((key) => (
            <div key={key}>
              <label className="block text-sm font-medium text-gray-700">
                {key}
              </label>
              <input
                type="number"
                name={key}
                value={inputData[key]}
                onChange={handleChange}
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              />
            </div>
          ))}
        </div>
        <button
          type="submit"
          className={`bg-green-500 text-white px-4 py-2 rounded mt-4 w-full ${
            isTraining ? "opacity-50 cursor-not-allowed" : ""
          }`}
          disabled={isTraining}
        >
          Predict
        </button>
      </form>

      {showModal && (
        <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center">
          <div className="bg-white mx-10 p-6 rounded shadow-lg max-w-sm w-full text-center">
            <h2 className="text-2xl font-semibold mb-4">Prediction Result</h2>
            <p className="text-lg mb-4">
              The model predicts:{" "}
              <span className="text-blue-500 font-bold">{prediction}</span>
            </p>
            <button
              onClick={() => setShowModal(false)}
              className="bg-red-500 text-white px-4 py-2 rounded"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
