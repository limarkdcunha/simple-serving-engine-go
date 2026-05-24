package main

import "fmt"


func ForwardPass(tensors map[string]*Tensor,ReverseVocab map[int]string ,encodedIds []int64) int64 {
	model_weights := tensors["model.embed_tokens.weight"].Data
    hidden_dim := tensors["model.embed_tokens.weight"].Shape[1]

    vectors := make([][]float32, 0, len(encodedIds))

    for _,encodedId := range encodedIds{
        start := encodedId * hidden_dim
        end := encodedId * hidden_dim + hidden_dim

        vectorSlice := make([]float32, hidden_dim)
		copy(vectorSlice, model_weights[start:end])
		vectors = append(vectors, vectorSlice)
    }

	// fmt.Println("Go after embed, last token:", vectors[len(vectors)-1][:5])

    for i := 0; i < 30; i++ {
        // RMSNorm all token vectors
        normedVectors := make([][]float32, 0, len(vectors))

        inputNorm := tensors[fmt.Sprintf("model.layers.%d.input_layernorm.weight", i)].Data
        postNorm := tensors[fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i)].Data
        qW := tensors[fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i)].Data
        kW := tensors[fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i)].Data
        vW := tensors[fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i)].Data
        oW := tensors[fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i)].Data
        gateProj := tensors[fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i)].Data
        upProj := tensors[fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i)].Data
        downProj := tensors[fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i)].Data
        
        for _, vector := range vectors {
            normedVectors = append(normedVectors, RMSNorm(vector,inputNorm))
        }

        var allQ [][]float32
        var allK [][]float32
        var allV [][]float32

        for pos, normedVector := range normedVectors {
            q, k, v := CalculateQKVvectors(normedVector, qW, kW, vW,int(hidden_dim))
            
            applyRoPE(q, pos, 64)
            applyRoPE(k, pos, 64)

            allQ = append(allQ, q)
            allK = append(allK, k)
            allV = append(allV, v)
        }

		// if i == 0 {
		// 	fmt.Println("Go embedding[0:5]:", vectors[0][0:5])
		// 	fmt.Println("Go normed[0:5]:", normedVectors[0][0:5])
		// 	fmt.Println("Go q[0:5]:", allQ[0][0:5])
		// }

        // attention
        attnResults := CalculateAttention(allQ,allK,allV,oW,9,3,64)

        // Residual connection
        for idx := range vectors {
            for d := 0; d < int(hidden_dim); d++ {
                vectors[idx][d] += attnResults[idx][d]
            }
        }

		if i == 0 {
			// fmt.Println("after attn residual, last token:", vectors[len(vectors)-1][:5])
			// put this AFTER the first residual, BEFORE the MLP
		}
            
        normedVectors = make([][]float32, 0, len(vectors))
        for idx, vector := range vectors {
            normedVector := RMSNorm(vector, postNorm)

            mlpOut := MLP(normedVector, gateProj, upProj, downProj, 576, 1536)
    
            // Second Residual 
            for d := 0; d < 576; d++ {
                vectors[idx][d] += mlpOut[d]
            }
        }

		if i == 0 {
			// fmt.Println("Go after layer 0, last token:", vectors[len(vectors)-1][:5])
		}
    }

	// fmt.Println("Go after layer 29, last token:", vectors[len(vectors)-1][:5])
    // Final norm
    finalNorm := tensors["model.norm.weight"].Data
    lastVector := RMSNorm(vectors[len(vectors)-1], finalNorm)

    // lmhead layer
    //  projecting 576-dimensional vector into a 49152-dimensional vector
    // lmHead := tensors["model.embed_tokens.weight"].Data 
    logits := GeneralVectorMatrixMultiply(lastVector, model_weights, 576,49152 )

    nextTokenID := ArgMax(logits)

	return nextTokenID
}