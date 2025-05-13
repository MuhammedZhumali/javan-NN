package nn;

import java.io.*;
import java.util.*;

public class DatasetLoader {

    public static List<double[]>[] load(String path, int inputCount) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.trim().split(",");
                if (tokens.length <= inputCount || line.trim().isEmpty()) {
                    continue; // skip invalid or empty line
                }


                double[] input = new double[inputCount];
                double[] output = new double[tokens.length - inputCount];

                for (int i = 0; i < inputCount; i++) {
                    input[i] = Double.parseDouble(tokens[i]);
                }

                for (int i = inputCount; i < tokens.length; i++) {
                    output[i - inputCount] = Double.parseDouble(tokens[i]);
                }

                inputs.add(input);
                outputs.add(output);
            }
        } catch (IOException e) {
            throw new RuntimeException("Could not load dataset: " + path, e);
        }

        return new List[]{inputs, outputs};
    }
}
