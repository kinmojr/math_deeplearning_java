package math.deeplearning.ch07;

import org.apache.commons.math3.linear.*;
import java.io.IOException;
import static math.deeplearning.common.Util.*;

/**
 * 線形重回帰モデル.
 */
public class LinearMultipleRegression {
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
    private RealVector W;

    /**
     * 初期化処理.
     *
     * @param iters 学習回数
     * @param alpha 学習率
     */
    public LinearMultipleRegression(int iters, double alpha) throws IOException {
        this.iters = iters;
        this.alpha = alpha;

        // The Boston Housing Datasetを読み込む
        RealMatrix boston = loadBoston();
        // 学習データとして部屋数(RM)列と低所得者率(LSTAT)列を抽出し、ダミー変数1を付加する
        x = addBiasCol(extractCol(boston, new int[]{5, 12}));
        // 正解データとして物件価格を抽出する
        yt = boston.getColumnVector(13);
        // 学習データの行数
        M = x.getRowDimension();
        // 学習データの列数
        D = x.getColumnDimension();

        // 重みベクトルを1で初期化する
        W = add(MatrixUtils.createRealVector(new double[D]), 1.0);
    }

    public static void main(String[] args) throws Exception {
        // 学習回数を2000、学習率を0.001に設定する
        LinearMultipleRegression lmr = new LinearMultipleRegression(2000, 0.001);
        // 学習する
        lmr.learn();
    }

    /**
     * 学習する.
     */
    public void learn() {
        for (int i = 0; i < iters; i++) {
            // 予測値ypを計算
            RealVector yp = dot(x, W);
            // 誤差ydを計算
            RealVector yd = sub(yp, yt);
            // 勾配に学習率を掛けて重みを更新
            W = sub(W, mult(div(dot(trans(x), yd), M), alpha));

            // 一定回数学習するごとに誤差を表示する
            if (i % 100 == 0)
                System.out.println(i + " " + mean(pow(yd, 2)) / 2);
        }
    }
}
