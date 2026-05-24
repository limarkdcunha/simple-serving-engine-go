package main

import "math"

func applyRoPE(vector []float32, position int, headDim int) []float32 {
    for h := 0; h < len(vector); h += headDim {
        for i := 0; i < headDim/2; i++ {
            theta := float64(position) / math.Pow(10000, float64(2*i)/float64(headDim))

            cosTheta := float32(math.Cos(theta))
            sinTheta := float32(math.Sin(theta))

            idx0 := h + i              // first half
            idx1 := h + i + headDim/2  // second half

            x0 := vector[idx0]
            x1 := vector[idx1]

            vector[idx0] = x0*cosTheta - x1*sinTheta
            vector[idx1] = x0*sinTheta + x1*cosTheta
        }
    }
    return vector
}