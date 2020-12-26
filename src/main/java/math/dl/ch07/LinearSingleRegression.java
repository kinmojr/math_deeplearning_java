package math.dl.ch07;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.IOException;

import static math.dl.common.Util.*;

/**
 * 線形単回帰モデル.
 */
public class LinearSingleRegression {
    // 学習率
    private double alpha;
    // 学習回数
    private int iters;
    // 学習データ
    private RealMatrix x;
    // 正解データ
    private RealVector yt;
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
    public LinearSingleRegression(int iters, double alpha) throws IOException {
        this.iters = iters;
        this.alpha = alpha;

        RealMatrix boston = loadBoston();
        x = addBiasCol(extractCol(boston, new int[]{5}));
        yt = boston.getColumnVector(13);
        M = x.getRowDimension();
        D = x.getColumnDimension();

        // 重みベクトルを1で初期化する
        w = add(MatrixUtils.createRealVector(new double[D]), 1.0);
    }

    public static void main(String[] args) throws Exception {
        LinearSingleRegression lsr = new LinearSingleRegression(50000, 0.01);
        lsr.learn();
    }

    /**
     * 学習する.
     */
    public void learn() {
        for (int i = 0; i < iters; i++) {
            // 予測値計算
            RealVector yp = dot(x, w);
            // 誤差計算
            RealVector yd = yp.subtract(yt);
            // 勾配計算
            w = sub(w, mult(div(dot(t(x), yd), M), alpha));

            // 一定回数学習するごとに誤差を表示する
            if (i % 100 == 0)
                System.out.println(i + " " + mean(pow(yd, 2)) / 2);
        }
    }
}
