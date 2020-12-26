package math.deeplearning.ch10;

import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static math.deeplearning.common.Util.*;

public class DeepLearning2 {
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
    private RealMatrix U;
    private RealMatrix V;
    private RealMatrix W;

    public DeepLearning2(int iters, int H, int batchSize, double alpha) throws IOException {
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
        U = initW(D, H);
        V = initW(H + 1, H);
        W = initW(H + 1, N);
    }

    public static void main(String... args) throws Exception {
        DeepLearning2 dl2 = new DeepLearning2(10000, 128, 512, 0.01);
        dl2.learn();
    }

    public void learn() {
        // ランダムサンプリングのindexを初期化
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < M; i++) indexes.add(i);

        for (int i = 0; i < iters; i++) {
            // 学習データのサンプリング
            List<Integer> index = randIndex(indexes, M, batchSize);
            RealMatrix x = sampling(xAll, index);
            RealMatrix yt = sampling(ytAll, index);

            // 予測値計算
            RealMatrix a = dot(x, U);
            RealMatrix b = reLU(a);
            RealMatrix b1 = addBiasCol(b);
            RealMatrix c = dot(b1, V);
            RealMatrix d = reLU(c);
            RealMatrix d1 = addBiasCol(d);
            RealMatrix u = dot(d1, W);
            RealMatrix yp = softmax(u);
            // 誤差計算
            RealMatrix yd = sub(yp, yt);
            RealMatrix dd = mult(step(c), dot(yd, t(removeBias(W))));
            RealMatrix bd = mult(step(a), dot(dd, t(removeBias(V))));
            // 勾配計算
            W = sub(W, mult(div(dot(t(d1), yd), batchSize), alpha));
            V = sub(V, mult(div(dot(t(b1), dd), batchSize), alpha));
            U = sub(U, mult(div(dot(t(x), bd), batchSize), alpha));

            // 一定回数学習するごとに誤差と精度を表示
            if (i % 100 == 0) {
                RealMatrix p = softmax(dot(addBiasCol(reLU(dot(addBiasCol(reLU(dot(xTest, U))), V))), W));
                System.out.print(i + " " + crossEntropy(ytTest, p) + " ");
                System.out.println(calcAccuracy(ytTest, p));
            }
        }
    }
}
