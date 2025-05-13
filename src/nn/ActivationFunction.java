package nn;

public class ActivationFunction {

    public static double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    public static double dsigmoid(double y) {
        // Derivative of sigmoid in terms of output y
        return y * (1 - y);
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double drelu(double x) {
        return x > 0 ? 1 : 0;
    }
}
