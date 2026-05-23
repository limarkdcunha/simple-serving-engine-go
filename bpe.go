package main

import (
	"encoding/json"
	"os"
	"strings"
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


func Decode(encodedIds []int,ReverseVocab map[int]string) string{
	var result string

    for _, id := range encodedIds {
        token := ReverseVocab[id]
		
		// TO DO: This can be ehanced
        result += strings.ReplaceAll(token, "Ġ", " ")
    }
    return result
}

func PerformBPE(PretokenizedStrings []string,Vocab map[string]int,MergesMap map[string]int) []int {
	var tokenIDs []int

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
                tokenIDs = append(tokenIDs, id)
            }
        }
	}

	return tokenIDs
}