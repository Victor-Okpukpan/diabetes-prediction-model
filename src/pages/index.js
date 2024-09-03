import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';

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
    const [isPredicting, setIsPredicting] = useState(false);
    const [showModal, setShowModal] = useState(false);

    const loadCSV = async () => {
        console.log('Loading CSV data...');
        const response = await fetch('/diabetes.csv');
        const csvData = await response.text();
        return new Promise((resolve) => {
            Papa.parse(csvData, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    console.log('CSV data loaded successfully.');
                    resolve(results.data);
                },
            });
        });
    };

    const preprocessData = (data) => {
        console.log('Preprocessing data...');
        const inputs = data.map((row) => [
            row.Pregnancies, row.Glucose, row.BloodPressure,
            row.SkinThickness, row.Insulin, row.BMI,
            row.DiabetesPedigreeFunction, row.Age,
        ]);

        const labels = data.map((row) => row.Outcome);

        const inputTensor = tf.tensor2d(inputs);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        console.log('Data preprocessed successfully.');
        return { inputTensor, labelTensor };
    };

    const trainModel = async () => {
        setIsTraining(true);
        console.log('Starting model training...');
        const data = await loadCSV();
        const { inputTensor, labelTensor } = preprocessData(data);

        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 8, inputShape: [8], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({
            optimizer: tf.train.adam(),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });

        console.log('Model structure created. Starting training...');
        const history = await model.fit(inputTensor, labelTensor, {
            epochs: 100,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: accuracy = ${logs.acc}`);
                },
            },
        });

        setAccuracy(history.history.acc.pop());
        setModel(model);
        setIsTraining(false);

        console.log('Model training completed.');
        inputTensor.dispose();
        labelTensor.dispose();
    };

    const handleChange = (e) => {
        setInputData({
            ...inputData,
            [e.target.name]: parseFloat(e.target.value)
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!model) {
            alert('Please train the model first!');
            return;
        }
        setIsPredicting(true);
        console.log('Starting prediction...');
        const inputTensor = tf.tensor2d([Object.values(inputData)]);
        const output = model.predict(inputTensor);
        const result = await output.data();
        console.log('Prediction completed.');
        console.log(`Prediction result: ${result[0]}`);
        setPrediction(result[0] >= 0.5 ? 'Positive' : 'Negative');
        setShowModal(true); // Show the modal with the prediction result
        setIsPredicting(false);
    };

    return (
        <div className="flex flex-col text-black items-center justify-center min-h-screen bg-gray-100 p-4">
            <h1 className="text-2xl font-bold mb-4">Diabetes Prediction System</h1>
            <button 
                onClick={trainModel} 
                className={`bg-blue-500 text-white px-4 py-2 rounded mb-4 ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isTraining}
            >
                {isTraining ? 'Training Model...' : 'Train Model'}
            </button>
            {accuracy && !isTraining && (
                <h2 className="text-lg font-medium text-green-500 mb-4">
                    Model Accuracy: {(accuracy * 100).toFixed(2)}%
                </h2>
            )}
            {!accuracy && !isTraining && (
                <div className="text-red-500 mb-4">
                    Train the model before you begin
                </div>
            )}
            {isTraining && (
                <div className="text-blue-500 mb-4">Training in progress... Please wait.</div>
            )}
            <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow-md w-full max-w-md">
                <h2 className="text-xl font-semibold mb-4">Input Data</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {Object.keys(inputData).map(key => (
                        <div key={key}>
                            <label className="block text-sm font-medium text-gray-700">{key}</label>
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
                    className={`bg-green-500 text-white px-4 py-2 rounded mt-4 w-full ${isPredicting ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={isPredicting}
                >
                    {isPredicting ? 'Predicting...' : 'Predict'}
                </button>
            </form>
            {isPredicting && (
                <div className="text-blue-500 mt-4">Calculating prediction... Please wait.</div>
            )}

            {/* Modal for showing the prediction */}
            {showModal && (
                <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center">
                    <div className="bg-white mx-10 p-6 rounded shadow-lg max-w-sm w-full text-center">
                        <h2 className="text-2xl font-semibold mb-4">Prediction Result</h2>
                        <p className="text-lg mb-4">The model predicts: <span className="text-blue-500 font-bold">{prediction}</span></p>
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
