package nn;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, new int[]{6, 4}, 1);
        nn.setLearningRate(0.1);
        nn.setActivations(new String[]{"relu", "relu", "sigmoid"});


        List<double[]>[] data = DatasetLoader.load("xor.csv", 2);
        List<double[]> inputs = data[0];
        List<double[]> outputs = data[1];

        for (int epoch = 0; epoch < 3000; epoch++) {
            for (int i = 0; i < inputs.size(); i++) {
                nn.train(inputs.get(i), outputs.get(i));
            }
        }

        System.out.println("Results:");
        for (int i = 0; i < inputs.size(); i++) {
            double[] prediction = nn.feedforward(inputs.get(i));
            System.out.printf("Input: %s → %.4f (Expected: %.1f)%n",
                    Arrays.toString(inputs.get(i)), prediction[0], outputs.get(i)[0]);
        }
    }
}
