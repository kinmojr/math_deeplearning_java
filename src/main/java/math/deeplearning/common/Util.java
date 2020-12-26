package math.deeplearning.common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class Util {
    // #########################
    // # Constants
    // #########################
    private static final String BOSTON_BASE_URL = "http://lib.stat.cmu.edu/datasets";
    private static final String BOSTON_FILE_NAME = "boston";
    private static final String BOSTON_DIR_NAME = "./dataset/boston";

    private static final String IRIS_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris";
    private static final String IRIS_FILE_NAME = "bezdekIris.data";
    private static final String IRIS_DIR_NAME = "./dataset/iris";

    private static final String MNIST_BASE_URL = "http://yann.lecun.com/exdb/mnist";
    public static final String MNIST_TRAIN_IMAGE_FILE_NAME = "train-images-idx3-ubyte.gz";
    public static final String MNIST_TRAIN_LABEL_FILE_NAME = "train-labels-idx1-ubyte.gz";
    public static final String MNIST_TEST_IMAGE_FILE_NAME = "t10k-images-idx3-ubyte.gz";
    public static final String MNIST_TEST_LABEL_FILE_NAME = "t10k-labels-idx1-ubyte.gz";
    private static final String MNIST_DIR_NAME = "./dataset/mnist";

    // #########################
    // # Math Functions
    // #########################
    public static RealMatrix add(RealMatrix aM, RealMatrix bM) {
        return aM.add(bM);
    }

    public static RealMatrix add(RealMatrix aM, double b) {
        return aM.scalarAdd(b);
    }

    public static RealVector add(RealVector aV, double b) {
        return aV.mapAdd(b);
    }

    public static RealMatrix sub(RealMatrix aM, RealMatrix bM) {
        return aM.subtract(bM);
    }

    public static RealVector sub(RealVector aV, RealVector bV) {
        return aV.subtract(bV);
    }

    public static RealMatrix mult(RealMatrix aM, double b) {
        return aM.scalarMultiply(b);
    }

    public static RealMatrix mult(RealMatrix aM, RealMatrix bM) {
        double[][] aA2 = aM.getData();
        double[][] bA2 = bM.getData();
        double[][] retA2 = new double[aA2.length][aA2[0].length];
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                retA2[i][j] = aA2[i][j] * bA2[i][j];
            }
        }
        return MatrixUtils.createRealMatrix(retA2);
    }

    public static RealVector mult(RealVector aV, double b) {
        return aV.mapMultiply(b);
    }

    public static RealMatrix div(RealMatrix aM, double b) {
        return aM.scalarMultiply(1 / b);
    }

    public static RealVector div(RealVector aV, double b) {
        return aV.mapMultiply(1 / b);
    }

    public static RealMatrix dot(RealMatrix aM, RealMatrix bM) {
        return aM.multiply(bM);
    }

    public static RealVector dot(RealMatrix aM, RealVector bV) {
        return aM.operate(bV);
    }

    public static RealMatrix pow(RealMatrix aM, double b) {
        double[][] pA2 = aM.getData();
        for (int i = 0; i < pA2.length; i++) {
            for (int j = 0; j < pA2[i].length; j++) {
                pA2[i][j] = Math.pow(pA2[i][j], b);
            }
        }
        return MatrixUtils.createRealMatrix(pA2);
    }

    public static RealVector pow(RealVector aV, double b) {
        double[] pA = aV.toArray();
        for (int i = 0; i < pA.length; i++) {
            pA[i] = Math.pow(pA[i], b);
        }
        return MatrixUtils.createRealVector(pA);
    }

    public static RealMatrix t(RealMatrix aM) {
        return aM.transpose();
    }

    public static double mean(RealVector aV) {
        double[] aA = aV.toArray();
        double sum = 0.0;
        for (double val : aA) {
            sum += val;
        }
        return sum / aA.length;
    }

    public static RealMatrix sigmoid(RealMatrix aM) {
        double[][] aA2 = aM.getData();
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                aA2[i][j] = sigmoid(aA2[i][j]);
            }
        }
        return MatrixUtils.createRealMatrix(aA2);
    }

    public static RealVector sigmoid(RealVector aV) {
        double[] aA = aV.toArray();
        for (int i = 0; i < aA.length; i++) {
            aA[i] = sigmoid(aA[i]);
        }
        return MatrixUtils.createRealVector(aA);
    }

    public static double sigmoid(double a) {
        return 1 / (1 + Math.exp(-a));
    }

    public static RealVector softmax(RealVector aV) {
        double[] aA = aV.toArray();
        double sum = 0.0;
        for (int i = 0; i < aA.length; i++) {
            aA[i] = Math.exp(aA[i]);
            sum += aA[i];
        }
        for (int i = 0; i < aA.length; i++) {
            aA[i] /= sum;
        }
        return MatrixUtils.createRealVector(aA);
    }

    public static RealMatrix softmax(RealMatrix aM) {
        double[][] aA2 = aM.getData();
        for (int i = 0; i < aA2.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < aA2[i].length; j++) {
                aA2[i][j] = Math.exp(aA2[i][j]);
                sum += aA2[i][j];
            }
            for (int j = 0; j < aA2[i].length; j++) {
                aA2[i][j] /= sum;
            }
        }
        return MatrixUtils.createRealMatrix(aA2);
    }

    public static RealMatrix reLU(RealMatrix aM) {
        double[][] aA2 = aM.getData();
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                if (aA2[i][j] <= 0) aA2[i][j] = 0.0;
            }
        }
        return MatrixUtils.createRealMatrix(aA2);
    }

    public static RealMatrix step(RealMatrix aM) {
        double[][] aA2 = aM.getData();
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                aA2[i][j] = (aA2[i][j] <= 0) ? 0.0 : 1.0;
            }
        }
        return MatrixUtils.createRealMatrix(aA2);
    }

    public static double crossEntropy(RealMatrix tM, RealMatrix pM) {
        double[][] tA2 = tM.getData();
        double[][] pA2 = pM.getData();
        double ce = 0.0;
        for (int i = 0; i < tA2.length; i++) {
            for (int j = 0; j < tA2[i].length; j++) {
                ce -= tA2[i][j] * Math.log(pA2[i][j]);
            }
        }
        return ce / tA2.length;
    }

    public static double crossEntropy(RealVector tV, RealVector pV) {
        double[] tA = tV.toArray();
        double[] pA = pV.toArray();
        double sum = 0.0;
        for (int i = 0; i < tA.length; i++) {
            sum -= tA[i] * Math.log(pA[i]) + (1 - tA[i]) * Math.log(1 - pA[i]);
        }
        return sum / tA.length;
    }

    // #########################
    // # Data Functions
    // #########################
    private static void download(String baseUrl, String dirName, String fileName) throws IOException {
        if (!new File(dirName).exists()) new File(dirName).mkdirs();

        if (!new File(dirName + "/" + fileName).exists()) {
            System.out.println("Downloading " + baseUrl + "/" + fileName + " ...");
            URLConnection conn = new URL(baseUrl + "/" + fileName).openConnection();
            File file = new File(dirName + "/" + fileName);
            try (InputStream is = conn.getInputStream();
                 FileOutputStream os = new FileOutputStream(file, false)) {
                byte[] data = new byte[1024];
                while (true) {
                    int ret = is.read(data);
                    if (ret == -1) {
                        break;
                    }
                    os.write(data, 0, ret);
                }
            }
        }
    }

    public static RealMatrix loadBoston() throws IOException {
        download(BOSTON_BASE_URL, BOSTON_DIR_NAME, BOSTON_FILE_NAME);

        BufferedReader br = new BufferedReader(new FileReader(new File(BOSTON_DIR_NAME + "/" + BOSTON_FILE_NAME)));
        for (int i = 0; i < 22; i++) br.readLine();

        double[][] valA2 = new double[506][14];
        String line;
        int i = 0;
        while ((line = br.readLine()) != null && !"".equals(line)) {
            line = line.trim().replaceAll(" +", " ");
            line += " " + br.readLine().trim().replaceAll(" +", " ");
            String[] valA = line.split(" ");
            for (int j = 0; j < valA.length; j++) {
                valA2[i][j] = Double.parseDouble(valA[j]);
            }
            i++;
        }
        return MatrixUtils.createRealMatrix(valA2);
    }

    public static RealMatrix loadIris() throws IOException {
        download(IRIS_BASE_URL, IRIS_DIR_NAME, IRIS_FILE_NAME);

        BufferedReader br = new BufferedReader(new FileReader(new File(IRIS_DIR_NAME + "/" + IRIS_FILE_NAME)));

        double[][] valA2 = new double[150][5];
        for (int i = 0; i < 150; i++) {
            String[] valA = br.readLine().split(",");
            for (int j = 0; j < valA.length; j++) {
                if (j == 4) {
                    if ("Iris-setosa".equals(valA[j])) {
                        valA2[i][j] = 0.0;
                    } else if ("Iris-versicolor".equals(valA[j])) {
                        valA2[i][j] = 1.0;
                    } else {
                        valA2[i][j] = 2.0;
                    }
                } else {
                    valA2[i][j] = Double.parseDouble(valA[j]);
                }
            }
        }
        return MatrixUtils.createRealMatrix(valA2);
    }

    public static RealMatrix loadIris(int startRow, int endRow) throws IOException {
        return extractRowCol(loadIris(), startRow, endRow - 1, 0, 4);
    }

    public static RealMatrix loadMnistImage(String fileName) throws IOException {
        download(MNIST_BASE_URL, MNIST_DIR_NAME, fileName);

        DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(MNIST_DIR_NAME + "/" + fileName)));
        dis.readInt();
        int numImg = dis.readInt();
        int numDim = dis.readInt() * dis.readInt();

        double[][] imgA2 = new double[numImg][numDim];
        for (int i = 0; i < numImg; i++) {
            for (int j = 0; j < numDim; j++) {
                imgA2[i][j] = (double) dis.readUnsignedByte();
            }
        }
        return MatrixUtils.createRealMatrix(imgA2);
    }

    public static RealVector loadMnistLabel(String fileName) throws IOException {
        download(MNIST_BASE_URL, MNIST_DIR_NAME, fileName);

        DataInputStream dis = new DataInputStream(new GZIPInputStream(new FileInputStream(MNIST_DIR_NAME + "/" + fileName)));
        dis.readInt();
        int numLabel = dis.readInt();

        double[] labelA = new double[numLabel];
        for (int i = 0; i < numLabel; i++) {
            labelA[i] = dis.readUnsignedByte();
        }
        return MatrixUtils.createRealVector(labelA);
    }


    // #########################
    // # Other Utility Functions
    // #########################
    public static RealMatrix addBiasCol(RealMatrix aM) {
        double[][] aA2 = new double[aM.getRowDimension()][aM.getColumnDimension() + 1];
        for (int i = 0; i < aM.getRowDimension(); i++) {
            aA2[i][0] = 1.0;
            for (int j = 0; j < aM.getColumnDimension(); j++) {
                aA2[i][j + 1] = aM.getEntry(i, j);
            }
        }
        return MatrixUtils.createRealMatrix(aA2);
    }

    public static RealMatrix removeBias(RealMatrix aM) {
        return aM.getSubMatrix(1, aM.getRowDimension() - 1, 0, aM.getColumnDimension() - 1);
    }

    public static RealMatrix extractRowCol(RealMatrix aM, int startRow, int endRow, int startCol, int endCol) {
        return aM.getSubMatrix(startRow, endRow, startCol, endCol);
    }

    public static RealMatrix extractRowCol(RealMatrix aM, int startRow, int endRow, int[] cols) {
        return extractCol(extractRow(aM, startRow, endRow), cols);
    }

    public static RealMatrix extractRow(RealMatrix aM, int startRow, int endRow) {
        return extractRowCol(aM, startRow, endRow, 0, aM.getColumnDimension() - 1);
    }

    public static RealMatrix extractCol(RealMatrix aM, int startCol, int endCol) {
        return extractRowCol(aM, 0, aM.getRowDimension() - 1, startCol, endCol);
    }

    public static RealMatrix extractCol(RealMatrix aM, int[] cols) {
        int[] rows = new int[aM.getRowDimension()];
        for (int i = 0; i < aM.getRowDimension(); i++) rows[i] = i;
        return aM.getSubMatrix(rows, cols);
    }

    public static RealVector extractRowCol(RealMatrix aM, int startRow, int endRow, int col) {
        return aM.getColumnVector(col).getSubVector(startRow, endRow - startRow + 1);
    }

    public static RealVector extract(RealVector aV, int start, int end) {
        return aV.getSubVector(start, end - start + 1);
    }

    public static RealMatrix oneHotEncode(RealVector aV, int classNum) {
        double[] aA = aV.toArray();
        RealMatrix aM = MatrixUtils.createRealMatrix(aV.getDimension(), classNum);
        for (int i = 0; i < aA.length; i++) {
            aM.setEntry(i, (int) aA[i], 1.0);
        }
        return aM;
    }

    public static void print(RealVector aV, String varName) {
        double[] aA = aV.toArray();
        for (int i = 0; i < aA.length; i++) {
            System.out.println(varName + i + " = " + aA[i]);
        }
    }

    public static void print(RealMatrix aM, String varName) {
        double[][] aA2 = aM.getData();
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                System.out.print(varName + i + "" + j + " = " + aA2[i][j] + ", ");
            }
            System.out.println();
        }
    }

    public static RealMatrix shuffle(RealMatrix aM) {
        double[][] aA2 = aM.getData();
        RealMatrix retM = MatrixUtils.createRealMatrix(aM.getRowDimension(), aM.getColumnDimension());

        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < aM.getRowDimension(); i++) {
            indexes.add(i);
        }

        Random rand = new Random();
        for (int i = 0; i < aM.getRowDimension(); i++) {
            int index = rand.nextInt(indexes.size());
            retM.setRow(indexes.get(index), aA2[i]);
            indexes.remove(index);
        }
        return retM;
    }

    public static List<Integer> randIndex(List<Integer> indexes, int total, int size) {
        if (indexes.size() < size) {
            indexes.clear();
            for (int i = 0; i < total; i++) indexes.add(i);
        }

        Random rand = new Random();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            int index = rand.nextInt(indexes.size());
            list.add(indexes.get(index));
            indexes.remove(index);
        }
        return list;
    }

    public static RealMatrix sampling(RealMatrix aM, List<Integer> index) {
        RealMatrix rM = MatrixUtils.createRealMatrix(index.size(), aM.getColumnDimension());
        for (int i = 0; i < index.size(); i++) {
            rM.setRowVector(i, aM.getRowVector(index.get(i)));
        }
        return rM;
    }

    public static double calcAccuracy(RealVector tV, RealVector pV) {
        double[] tA = tV.toArray();
        double[] pA = pV.toArray();
        double sum = 0;
        for (int i = 0; i < tA.length; i++) {
            if (pA[i] <= 0.5) {
                if (tA[i] == 0.0) sum++;
            } else {
                if (tA[i] == 1.0) sum++;
            }
        }
        return sum / tA.length;
    }

    public static double calcAccuracy(RealMatrix tM, RealMatrix pM) {
        double[][] tA2 = tM.getData();
        double[][] pA2 = pM.getData();
        double ans = 0;
        for (int i = 0; i < tA2.length; i++) {
            int maxIndex = 0;
            double max = 0.0;
            for (int j = 0; j < tA2[i].length; j++) {
                if (pA2[i][j] > max) {
                    maxIndex = j;
                    max = pA2[i][j];
                }
            }
            if (tA2[i][maxIndex] == 1.0) {
                ans++;
            }
        }
        return ans / tA2.length;
    }

    public static RealMatrix initW(int inDim, int outDim) {
        double[][] wA2 = new double[inDim][outDim];
        Random rand = new Random();
        for (int i = 0; i < wA2.length; i++) {
            for (int j = 0; j < wA2[i].length; j++) {
                wA2[i][j] = rand.nextGaussian() / Math.sqrt(inDim / 2);
            }
        }

        return MatrixUtils.createRealMatrix(wA2);
    }
}
