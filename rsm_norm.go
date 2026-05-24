package main

import "math"

func RMSNorm(vector []float32,weight []float32 ) []float32 {
    var sumOfSqrs float32
    n := float32(len(vector))

    for _,val := range vector {
        sumOfSqrs += val * val
    }

    rms := float32(math.Sqrt(float64(sumOfSqrs/n + 1e-6)))

    output := make([]float32, len(vector))
    
    for i, v := range vector {
        output[i] = (v / rms) * weight[i]
    }

    return output
}