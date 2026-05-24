package main

import (
	"encoding/json"
	"os"
)

const TOKENIZER_FILEPATH =  MODEL_PATH + "//tokenizer.json"


type TokenizerMetadata struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
}

func LoadTokenizerMetadata() (Vocab map[string]int, ReverseVocab map[int]string, MergesMap map[string]int) {
	fileData, err := os.ReadFile(TOKENIZER_FILEPATH)
	if err != nil {
		panic(err)
	}

	var metadata TokenizerMetadata
	if err = json.Unmarshal(fileData, &metadata); err != nil {
		panic(err)
	}

	reverseVocab := make(map[int]string, len(metadata.Model.Vocab))
	
	for str, id := range metadata.Model.Vocab {
		reverseVocab[id] = str
	}

	mergesMap := make(map[string]int, len(metadata.Model.Merges))
    for rank, pair := range metadata.Model.Merges {
        mergesMap[pair] = rank
    }
	
	return metadata.Model.Vocab, reverseVocab, mergesMap
}

func GetUnicodeToByteMap() map[rune]byte {
    forward := GetByteToUnicodeMap()
    reverse := make(map[rune]byte, len(forward))
    for b, s := range forward {
        r := []rune(s)[0]
        reverse[r] = b
    }
    return reverse
}


func Decode(encodedId int, ReverseVocab map[int]string,UnicodeToByteMap map[rune]byte) string {
    token := ReverseVocab[encodedId]
    
    // convert each unicode character back to its original byte
    // uToB := GetUnicodeToByteMap()
    var bytes []byte
    for _, r := range token {
        if b, ok := UnicodeToByteMap[r]; ok {
            bytes = append(bytes, b)
        }
    }
    
    return string(bytes)
}

func PerformBPE(PretokenizedStrings []string,Vocab map[string]int,MergesMap map[string]int) []int64 {
	var tokenIDs []int64

	for _,preString := range PretokenizedStrings {
		var parts []string
		for _, r := range preString {
			parts = append(parts, string(r))
		}
		
		for {
			lowestRank := -1 
			bestPairIdx := -1

			for i := 0; i < len(parts)-1; i++ {
				pairKey := parts[i] + " " + parts[i+1]
				
				if rank, ok := MergesMap[pairKey]; ok {
					if lowestRank == -1 || rank < lowestRank {
						lowestRank = rank
						bestPairIdx = i
					}
				}
			}

			if bestPairIdx == -1 {
				break
			}
			
			// Merging part look at diagrams to understand this
			newParts := make([]string, 0, len(parts)-1)
			newParts = append(newParts, parts[:bestPairIdx]...)
			newParts = append(newParts, parts[bestPairIdx]+parts[bestPairIdx+1])
			newParts = append(newParts, parts[bestPairIdx+2:]...)

			parts = newParts
		}

        for _, p := range parts {
            if id, ok := Vocab[p]; ok {
                tokenIDs = append(tokenIDs,int64( id))
            }
        }
	}

	return tokenIDs
}