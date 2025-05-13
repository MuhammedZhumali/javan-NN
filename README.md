# Java Neural Network from Scratch

A simple feedforward neural network written in pure Java.  
Supports:
- Multiple hidden layers
- ReLU & Sigmoid activations (per layer)
- Backpropagation training
- XOR training from CSV file

---

## ✅ Features

- Fully connected feedforward network
- CSV dataset loader
- Activation functions: `sigmoid`, `relu`
- Custom layer-wise activations
- Easy-to-extend for larger datasets

---

## 📁 Structure
├ xor.csv

├── src/

│ └── nn/

│ ├── NeuralNetwork.java

│ ├── Matrix.java

│ ├── ActivationFunction.java

│ ├── DatasetLoader.java

│ └── Main.java


---

## 📊 Example: XOR Training

Contents of `xor.csv`:

0,0,0

0,1,1

1,0,1

1,1,0


Main.java example:
```java
NeuralNetwork nn = new NeuralNetwork(2, new int[]{6, 4}, 1);
nn.setLearningRate(0.1);
nn.setActivations(new String[]{"relu", "relu", "sigmoid"});

List<double[]>[] data = DatasetLoader.load("xor.csv", 2);
List<double[]> inputs = data[0];
List<double[]> outputs = data[1];

for (int epoch = 0; epoch < 5000; epoch++) {
    for (int i = 0; i < inputs.size(); i++) {
        nn.train(inputs.get(i), outputs.get(i));
    }
}
```

📦 Future Ideas

GUI with JavaFX

Model save/load

Support for softmax, tanh

MNIST or Iris dataset

