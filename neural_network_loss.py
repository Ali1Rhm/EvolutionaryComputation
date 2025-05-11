import math
from typing import List

def total_loss(weights_flat: List[float],
               inputs: List[List[float]],
               labels: List[int]) -> float:
    idx = 0
    W1 = [weights_flat[idx + i*5 : idx + (i+1)*5] for i in range(21)]
    idx += 21*5
    b1 = weights_flat[idx : idx+5]
    idx += 5
    W2 = [weights_flat[idx + i*5 : idx + (i+1)*5] for i in range(5)]
    idx += 5*5
    b2 = weights_flat[idx : idx+5]
    idx += 5
    W3 = [weights_flat[idx + i*3 : idx + (i+1)*3] for i in range(5)]
    idx += 5*3
    b3 = weights_flat[idx : idx+3]

    def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
        in_dim = len(mat)
        out_dim = len(mat[0])
        out = [0.0] * out_dim
        for i in range(in_dim):
            xi = vec[i]
            row = mat[i]
            for j in range(out_dim):
                out[j] += row[j] * xi
        return out

    def add(a: List[float], b: List[float]) -> List[float]:
        return [x + y for x, y in zip(a, b)]

    def relu(vec: List[float]) -> List[float]:
        return [x if x > 0 else 0.0 for x in vec]

    def softmax(logits: List[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def xent(pred: List[float], true_label: int) -> float:
        return -math.log(max(pred[true_label], 1e-15))

    total = 0.0
    for x_vec, y in zip(inputs, labels):
        h1      = relu(add(matvec(W1, x_vec), b1))
        h2      = relu(add(matvec(W2, h1), b2))
        logits  =     add(matvec(W3, h2), b3)
        y_pred  = softmax(logits)
        total  += xent(y_pred, y)

    return total
