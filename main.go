package main

import "fmt"

func main() {
    // read_tensors()
    input := "Hello world, it's 42 degrees!"

    // Tokenizer START
    Vocab,ReverseVocab,MergesMap := LoadTokenizerMetadata()
   
    tokens := Pretokenize(input,)

    encodedIds  := PerformBPE(tokens,Vocab,MergesMap)
    // Tokenizer END

    Decode(encodedIds,ReverseVocab)
    // fmt.Println(decodedString)

    tensors := ReadTensors()

    fmt.Println(len(tensors["model.embed_tokens.weight"].Data))
    fmt.Println(tensors["model.embed_tokens.weight"].Shape)
    fmt.Println(tensors["model.embed_tokens.weight"].Data[0:5] )
    
}
