package main

import "math"

func ArgMax(logits []float32) int64 {
    if len(logits) == 0 {
        return -1
    }

    maxVal := logits[0]
    maxIdx := 0

    for i, val := range logits {
        if val > maxVal {
            maxVal = val
            maxIdx = i
        }
    }

    return int64(maxIdx)
}

func CalculateQKVvectors(vector []float32, queryWeight []float32, keyWeight []float32, valueWeight []float32, hidden_dim int) ([]float32, []float32, []float32) {
    qOutDim := len(queryWeight) / hidden_dim  // 576
    kOutDim := len(keyWeight) / hidden_dim    // 192
    vOutDim := len(valueWeight) / hidden_dim  // 192

    queryVector := GeneralVectorMatrixMultiply(vector, queryWeight, hidden_dim, qOutDim)
    keyVector   := GeneralVectorMatrixMultiply(vector, keyWeight,   hidden_dim, kOutDim)
    valueVector := GeneralVectorMatrixMultiply(vector, valueWeight, hidden_dim, vOutDim)

    return queryVector, keyVector, valueVector
}


func GeneralVectorMatrixMultiply(vector []float32, matrix []float32, rows int, cols int) []float32 {
    result := make([]float32, cols)
    for i := 0; i < cols; i++ {
        var sum float32 = 0
        for j := 0; j < rows; j++ {
            sum += vector[j] * matrix[i*rows+j]
        }
        result[i] = sum
    }
    return result
}

func DotProduct(a, b []float32) float32 {
    var sum float32
    for i := 0; i < len(a); i++ {
        sum += a[i] * b[i]
    }
    return sum
}

func Softmax(scores []float32) []float32 {
    if len(scores) == 0 {
        return nil
    }

    maxScore := scores[0]
    for _, s := range scores {
        if s > maxScore {
            maxScore = s
        }
    }

    exps := make([]float32, len(scores))
    var sumExps float32
    for i, s := range scores {
        // e^(s - max)
        val := float32(math.Exp(float64(s - maxScore)))
        exps[i] = val
        sumExps += val
    }

    for i := range exps {
        exps[i] = exps[i] / sumExps
    }

    return exps
}

func FlattenHeads(heads [][]float32, headDim int) []float32 {
    numHeads := len(heads)
    flat := make([]float32, numHeads * headDim)
    
    for i, head := range heads {
        copy(flat[i*headDim : (i+1)*headDim], head)
    }
    
    return flat
}

func silu(x float32) float32 {
    return x * (1.0 / (1.0 + float32(math.Exp(float64(-x)))))
}

func MLP(normedVector []float32, gateW []float32, upW []float32, downW []float32, hiddenDim int, intermediateDim int) []float32 {
    gate := GeneralVectorMatrixMultiply(normedVector, gateW, hiddenDim, intermediateDim)
    up   := GeneralVectorMatrixMultiply(normedVector, upW, hiddenDim, intermediateDim)

    for i := 0; i < intermediateDim; i++ {
        gate[i] = silu(gate[i]) * up[i]
    }

    output := GeneralVectorMatrixMultiply(gate, downW, intermediateDim, hiddenDim)

    return output
}

func CalculateAttention(allQ [][]float32, allK [][]float32, allV [][]float32, o_proj []float32, numQHeads int,numKVHeads int,headDim int) [][]float32{
    seqLen := len(allQ)
    output := make([][]float32, 0, seqLen)
    sqrt64 := float32(math.Sqrt(64))
    
    for i := range seqLen{
        concatenated := make([][]float32, 0, numQHeads) 

        for h := range numQHeads{
            kvHead := h / (numQHeads / numKVHeads) 
            q := allQ[i][h*64 : h*64+64]

            scores := make([]float32, 0, i+1)

            for j := 0; j <= i; j++ {
                k := allK[j][kvHead*64 : kvHead*64+64]

                score := DotProduct(q, k) / sqrt64
                scores = append(scores, score)
            }

            // softmax
            softmax_scores := Softmax(scores)

            headOutput := make([]float32, 64)

            for j := 0; j <= i; j++ {
                v := allV[j][kvHead*64 : kvHead*64+64]
                
                prob := softmax_scores[j] 

                for d := 0; d < 64; d++ {
                    headOutput[d] += prob * v[d]
                }
            }

            concatenated = append(concatenated,headOutput)
        }

        fullVector := FlattenHeads(concatenated,headDim)

        attnOut := GeneralVectorMatrixMultiply(fullVector, o_proj, 576, 576)

        output = append(output,attnOut)
    } 

    return output
}