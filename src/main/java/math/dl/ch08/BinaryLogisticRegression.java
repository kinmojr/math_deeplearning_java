package math.dl.ch08;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.IOException;

import static math.dl.common.Util.*;

public class BinaryLogisticRegression {
    // 学習率
    private double alpha;
    // 学習回数
    private int iters;
    // 学習データ
    private RealMatrix x;
    // 評価用学習データ
    private RealMatrix xTest;
    // 正解データ
    private RealVector yt;
    // 評価用正解データ
    private RealVector ytTest;
    // 入力データ行数
    private int M;
    // 入力データ列数
    private int D;
    // 重みベクトル
    private RealVector w;

    /**
     * 初期化処理.
     *
     * @param iters 学習回数
     * @param alpha 学習率
     */
    public BinaryLogisticRegression(int iters, double alpha) throws IOException {
        this.iters = iters;
        this.alpha = alpha;

        RealMatrix iris = shuffle(loadIris(0, 100));
        x = addBiasCol(extractRowCol(iris, 0, 69, 0, 1));
        xTest = addBiasCol(extractRowCol(iris, 70, 99, 0, 1));
        yt = extractRowCol(iris, 0, 69, 4);
        ytTest = extractRowCol(iris, 70, 99, 4);
        M = x.getRowDimension();
        D = x.getColumnDimension();

        // 重みベクトルを1で初期化する
        w = add(MatrixUtils.createRealVector(new double[D]), 1.0);
    }

    public static void main(String[] args) throws Exception {
        BinaryLogisticRegression blr = new BinaryLogisticRegression(10000, 0.01);
        blr.learn();
    }

    /**
     * 学習する.
     */
    public void learn() {
        // 学習する
        for (int i = 0; i < iters; i++) {
            RealVector yp = sigmoid(dot(x, w));
            RealVector yd = yp.subtract(yt);
            w = sub(w, mult(div(dot(t(x), yd), M), alpha));

            // 一定回数学習するごとに誤差と精度を表示する
            if (i % 10 == 0) {
                System.out.print("iter = " + i + "\tloss = " + crossEntropy(ytTest, sigmoid(dot(xTest, w))));
                System.out.println("\tscore = " + calcAccuracy(ytTest, sigmoid(dot(xTest, w))));
            }
        }
    }
}
