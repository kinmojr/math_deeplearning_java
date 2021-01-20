package math.deeplearning.ch09;

import org.apache.commons.math3.linear.*;
import java.io.IOException;
import static math.deeplearning.common.Util.*;

/**
 * ロジスティック回帰モデル(多クラス分類).
 */
public class MultipleLogisticRegression {
    // 学習率
    private double alpha;
    // 学習回数
    private int iters;
    // 学習データ
    private RealMatrix x;
    // 評価用学習データ
    private RealMatrix xTest;
    // 正解データ
    private RealMatrix yt;
    // 評価用正解データ
    private RealMatrix ytTest;
    // 入力データ行数
    private int M;
    // 入力データ列数
    private int D;
    // 重み行列
    private RealMatrix W;

    /**
     * 初期化処理.
     *
     * @param iters 学習回数
     * @param alpha 学習率
     */
    public MultipleLogisticRegression(int iters, double alpha) throws IOException {
        this.iters = iters;
        this.alpha = alpha;

        // Iris Data SetからSetosaとVersicolourの2種類のアヤメのデータを読み込む
        RealMatrix iris = shuffle(loadIris());
        // 学習データとしてがく片の長さの列と花弁の長さの列を抽出し、ダミー変数1を付加する
        x = addBiasCol(extractRowCol(iris, 0, 74, new int[]{0, 2}));
        // テストデータとしてがく片の長さの列と花弁の長さの列を抽出し、ダミー変数1を付加する
        xTest = addBiasCol(extractRowCol(iris, 75, 149, new int[]{0, 2}));
        // 4変数すべてを使う場合
        // x = addBiasCol(extractRowCol(iris, 0, 74, 0, 3));
        // xTest = addBiasCol(extractRowCol(iris, 75, 149, 0, 3));
        // 学習の正解データとしてアヤメの種類を抽出し、OneHotVector形式に変換する
        yt = oneHotEncode(extractRowCol(iris, 0, 74, 4), 3);
        // テストの正解データとしてアヤメの種類を抽出し、OneHotVector形式に変換する
        ytTest = oneHotEncode(extractRowCol(iris, 75, 149, 4), 3);
        // 正解データの行数
        M = x.getRowDimension();
        // 正解データの列数
        D = x.getColumnDimension();

        // 重み行列を1で初期化する
        W = add(MatrixUtils.createRealMatrix(D, 3), 1.0);
    }

    public static void main(String[] args) throws Exception {
        // 学習回数を10000、学習率を0.01に設定する
        MultipleLogisticRegression mlr = new MultipleLogisticRegression(10000, 0.01);
        // 学習する
        mlr.learn();
    }

    /**
     * 学習する.
     */
    public void learn() {
        // 学習する
        for (int i = 0; i < iters; i++) {
            // 予測値ypを計算
            RealMatrix yp = softmax(dot(x, W));
            // 誤差ydを計算
            RealMatrix yd = sub(yp, yt);
            // 勾配に学習率を掛けて重みを更新
            W = sub(W, mult(div(dot(trans(x), yd), M), alpha));

            // 一定回数学習するごとに誤差と精度を表示する
            if (i % 10 == 0) {
                RealMatrix p = softmax(dot(xTest, W));
                System.out.print("iter = " + i + "\tloss = " + crossEntropy(ytTest, p));
                System.out.println("\tscore = " + calcAccuracy(ytTest, p));
            }
        }
    }
}
