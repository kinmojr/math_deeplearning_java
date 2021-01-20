package math.deeplearning.ch08;

import org.apache.commons.math3.linear.*;
import java.io.IOException;
import static math.deeplearning.common.Util.*;

/**
 * ロジスティック回帰モデル(2値分類).
 */
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
    private RealVector W;

    /**
     * 初期化処理.
     *
     * @param iters 学習回数
     * @param alpha 学習率
     */
    public BinaryLogisticRegression(int iters, double alpha) throws IOException {
        this.iters = iters;
        this.alpha = alpha;

        // Iris Data SetからSetosaとVersicolourの2種類のアヤメのデータを読み込む
        RealMatrix iris = shuffle(loadIris(0, 100));
        // 学習データとしてがく片の長さの列とがく片の幅の列を抽出し、ダミー変数1を付加する
        x = addBiasCol(extractRowCol(iris, 0, 69, 0, 1));
        // テストデータとしてがく片の長さの列とがく片の幅の列を抽出し、ダミー変数1を付加する
        xTest = addBiasCol(extractRowCol(iris, 70, 99, 0, 1));
        // 学習の正解データとしてアヤメの種類を抽出する
        yt = extractRowCol(iris, 0, 69, 4);
        // テストの正解データとしてアヤメの種類を抽出する
        ytTest = extractRowCol(iris, 70, 99, 4);
        // 正解データの行数
        M = x.getRowDimension();
        // 正解データの列数
        D = x.getColumnDimension();

        // 重みベクトルを1で初期化する
        W = add(MatrixUtils.createRealVector(new double[D]), 1.0);
    }

    public static void main(String[] args) throws Exception {
        // 学習回数を10000、学習率を0.01に設定する
        BinaryLogisticRegression blr = new BinaryLogisticRegression(10000, 0.01);
        // 学習する
        blr.learn();
    }

    /**
     * 学習する.
     */
    public void learn() {
        // 学習する
        for (int i = 0; i < iters; i++) {
            // 予測値ypを計算
            RealVector yp = sigmoid(dot(x, W));
            // 誤差ydを計算
            RealVector yd = sub(yp, yt);
            // 勾配に学習率を掛けて重みを更新
            W = sub(W, mult(div(dot(trans(x), yd), M), alpha));

            // 一定回数学習するごとに誤差と精度を表示する
            if (i % 10 == 0) {
                RealVector p = sigmoid(dot(xTest, W));
                System.out.print("iter = " + i + "\tloss = " + crossEntropy(ytTest, p));
                System.out.println("\tscore = " + calcAccuracy(ytTest, p));
            }
        }
    }
}
