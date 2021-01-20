package math.deeplearning.ch10;

import org.apache.commons.math3.linear.RealMatrix;
import java.io.IOException;
import java.util.*;
import static math.deeplearning.common.Util.*;

/**
 * ディープラーニング(隠れ層1層).
 */
public class DeepLearning {
    // 学習データの行数
    private int M;
    // 学習データの列数(画像のピクセル数)
    private int D;
    // 分類クラス数
    private int N;
    // 学習回数
    private int iters;
    // バッチデータサイズ
    private int batchSize;
    // 学習率
    private double alpha;

    // MNIST画像データ
    private RealMatrix xAll;
    private RealMatrix xTest;
    private RealMatrix ytAll;
    private RealMatrix ytTest;

    // 重み行列
    private RealMatrix V;
    private RealMatrix W;

    public DeepLearning(int iters, int H, int batchSize, double alpha) throws IOException {
        // MNISTデータセットを読み込む
        xAll = addBiasCol(div(loadMnistImage(MNIST_TRAIN_IMAGE_FILE_NAME), 255));
        xTest = addBiasCol(div(loadMnistImage(MNIST_TEST_IMAGE_FILE_NAME), 255));
        ytAll = oneHotEncode(loadMnistLabel(MNIST_TRAIN_LABEL_FILE_NAME), 10);
        ytTest = oneHotEncode(loadMnistLabel(MNIST_TEST_LABEL_FILE_NAME), 10);

        M = xAll.getRowDimension();
        D = xAll.getColumnDimension();
        N = ytAll.getColumnDimension();

        this.iters = iters;
        this.batchSize = batchSize;
        this.alpha = alpha;

        // 重み行列をHe Normalで初期化
        V = initW(D, H);
        W = initW(H + 1, N);
    }

    public static void main(String... args) throws Exception {
        // 学習回数を10000、隠れ層のニューロン数を128、バッチサイズを512、学習率を0.01に設定する
        DeepLearning dl = new DeepLearning(10000, 128, 512, 0.01);
        // 学習する
        dl.learn();
    }

    public void learn() {
        // ランダムサンプリングのindexを初期化
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < M; i++) indexes.add(i);

        for (int i = 0; i < iters; i++) {
            // 学習データをサンプリング
            List<Integer> index = randIndex(indexes, M, batchSize);
            RealMatrix x = sampling(xAll, index);
            RealMatrix yt = sampling(ytAll, index);

            // 各層の出力値を計算
            RealMatrix a = dot(x, V);
            RealMatrix b = reLU(a);
            RealMatrix b1 = addBiasCol(b);
            RealMatrix u = dot(b1, W);
            RealMatrix yp = softmax(u);
            // 各層の誤差を計算
            RealMatrix yd = sub(yp, yt);
            RealMatrix bd = mult(step(a), dot(yd, trans(removeBias(W))));
            // 勾配に学習率を掛けて各層の重みを更新
            W = sub(W, mult(div(dot(trans(b1), yd), batchSize), alpha));
            V = sub(V, mult(div(dot(trans(x), bd), batchSize), alpha));

            // 一定回数学習するごとに誤差と精度を表示
            if (i % 100 == 0) {
                RealMatrix p = softmax(dot(addBiasCol(reLU(dot(xTest, V))), W));
                System.out.print(i + " " + crossEntropy(ytTest, p) + " ");
                System.out.println(calcAccuracy(ytTest, p));
            }
        }
    }
}
