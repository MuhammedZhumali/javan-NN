package nn;

import java.util.Random;

public class Matrix {
    public int rows, cols;
    public double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
        randomize();
    }

    public void randomize() {
        Random rand = new Random();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = rand.nextDouble() * 2 - 1; // from -1 to 1
    }

    public static Matrix fromArray(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    public double[] toArray() {
        double[] arr = new double[rows];
        for (int i = 0; i < rows; i++) {
            arr[i] = data[i][0];
        }
        return arr;
    }

    public static Matrix subtract(Matrix a, Matrix b) {
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < result.rows; i++)
            for (int j = 0; j < result.cols; j++)
                result.data[i][j] = a.data[i][j] - b.data[i][j];
        return result;
    }

    public void add(Matrix m) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] += m.data[i][j];
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        Matrix result = new Matrix(a.rows, b.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                double sum = 0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public void applyFunction(Function f) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = f.apply(data[i][j]);
    }


    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }


    public interface Function {
        double apply(double x);
    }
}
