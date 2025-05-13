package nn;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    int inputNodes;
    int outputNodes;
    int[] hiddenLayers;

    List<Matrix> weights = new ArrayList<>();
    List<Matrix> biases = new ArrayList<>();
    private List<String> activations = new ArrayList<>();

    double learningRate = 0.1;

    public NeuralNetwork(int inputNodes, int[] hiddenLayers, int outputNodes) {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
        this.hiddenLayers = hiddenLayers;

        int previousNodes = inputNodes;

        // Create weights and biases for each layer
        for (int hiddenNodes : hiddenLayers) {
            weights.add(new Matrix(hiddenNodes, previousNodes));
            biases.add(new Matrix(hiddenNodes, 1));
            previousNodes = hiddenNodes;
        }

        // Output layer
        weights.add(new Matrix(outputNodes, previousNodes));
        biases.add(new Matrix(outputNodes, 1));
    }

    public void setActivations(String[] types) {
        if (types.length != weights.size()) {
            throw new IllegalArgumentException("Activation count must match layer count (" + weights.size() + ")");
        }
        activations = new ArrayList<>(List.of(types));
    }

    public double[] feedforward(double[] inputArray) {
        Matrix current = Matrix.fromArray(inputArray);

        for (int i = 0; i < weights.size(); i++) {
            Matrix weight = weights.get(i);
            Matrix bias = biases.get(i);

            current = Matrix.multiply(weight, current);
            current.add(bias);

            String act = activations.size() > i ? activations.get(i) : "sigmoid";
            switch (act.toLowerCase()) {
                case "relu":
                    current.applyFunction(ActivationFunction::relu);
                    break;
                case "sigmoid":
                    current.applyFunction(ActivationFunction::sigmoid);
                    break;
                default:
                    throw new RuntimeException("Unknown activation: " + act);
            }
        }

        return current.toArray();
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void train(double[] inputArray, double[] targetArray) {
        // === FORWARD PASS ===
        Matrix input = Matrix.fromArray(inputArray);
        List<Matrix> activationsList = new ArrayList<>();
        activationsList.add(input);

        Matrix current = input;
        for (int i = 0; i < weights.size(); i++) {
            Matrix z = Matrix.multiply(weights.get(i), current);
            z.add(biases.get(i));

            String act = activations.size() > i ? activations.get(i) : "sigmoid";
            switch (act.toLowerCase()) {
                case "relu":
                    z.applyFunction(ActivationFunction::relu);
                    break;
                case "sigmoid":
                    z.applyFunction(ActivationFunction::sigmoid);
                    break;
                default:
                    throw new RuntimeException("Unknown activation: " + act);
            }

            current = z;
            activationsList.add(current);
        }

        // === BACKWARD PASS ===
        Matrix target = Matrix.fromArray(targetArray);
        Matrix error = Matrix.subtract(target, activationsList.get(activationsList.size() - 1));

        for (int i = weights.size() - 1; i >= 0; i--) {
            Matrix output = activationsList.get(i + 1);
            Matrix inputToLayer = activationsList.get(i);

            Matrix gradient = new Matrix(output.rows, output.cols);
            for (int r = 0; r < output.rows; r++) {
                for (int c = 0; c < output.cols; c++) {
                    double y = output.data[r][c];
                    double grad;
                    String actType = (activations.size() > i) ? activations.get(i).toLowerCase() : "sigmoid";
                    switch (actType) {
                        case "sigmoid":
                            grad = ActivationFunction.dsigmoid(y);
                            break;
                        case "relu":
                            grad = ActivationFunction.drelu(y);
                            break;
                        default:
                            throw new RuntimeException("Unknown activation: " + actType);
                    }
                    gradient.data[r][c] = grad * error.data[r][c] * learningRate;
                }
            }

            Matrix inputT = Matrix.transpose(inputToLayer);
            Matrix delta = Matrix.multiply(gradient, inputT);

            weights.get(i).add(delta);
            biases.get(i).add(gradient);

            if (i != 0) {
                Matrix weightsT = Matrix.transpose(weights.get(i));
                error = Matrix.multiply(weightsT, error);
            }
        }
    }
}
