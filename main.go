package main

import "fmt"

func main() {
    // read_tensors()
    input := "Hi, today is quite rainy isnt it?"

    // Tokenizer START
    Vocab,ReverseVocab,MergesMap := LoadTokenizerMetadata()
   
    tokens := Pretokenize(input)

    encodedIds  := PerformBPE(tokens,Vocab,MergesMap)
    // Tokenizer END

    tensors := ReadTensors()
    unicodeToByteMap := GetUnicodeToByteMap()

    for count := 0; count < 10; count++ {
        nextTokenID := ForwardPass(tensors, ReverseVocab, encodedIds)
        
        if nextTokenID == 0 || nextTokenID == 2 {
            break
        }

        nextWord := Decode(int(nextTokenID), ReverseVocab, unicodeToByteMap)
        fmt.Print(nextWord)

        encodedIds = append(encodedIds, nextTokenID)
    }
        
    // fmt.Println("nextWord:", nextWord)
}
