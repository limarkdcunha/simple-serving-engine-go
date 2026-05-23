package main

import (
	"regexp"
	"unicode"
)


const GPT2_REGEX = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`

func GetByteToUnicodeMap() map[byte]string {
	m := make(map[byte]string)

	for b := 0; b < 256; b++ {
		m[byte(b)] = string(rune(b))
	}

	m[' '] = "Ġ" 

	return m
}

func ByteLevelSplit(seq string) []string {
	var chunks []string
	var buffer string

	for _, r := range seq {
		if unicode.IsDigit(r) {
			if buffer != "" {
				chunks = append(chunks, buffer)
				buffer = ""
			}
			chunks = append(chunks, string(r))
		} else {
			buffer += string(r)
		}
	}
	if buffer != "" {
		chunks = append(chunks, buffer)
	}
	return chunks
}

func Pretokenize(seq string) []string {
	re := regexp.MustCompile(GPT2_REGEX)
	byteMap := GetByteToUnicodeMap()
	var finalTokens []string

	chunks := ByteLevelSplit(seq)

	for _, chunk := range chunks {
		rawTokens := re.FindAllString(chunk, -1)

		for _, token := range rawTokens {
			encodedToken := ""
			for i := 0; i < len(token); i++ {
				encodedToken += byteMap[token[i]]
			}
			finalTokens = append(finalTokens, encodedToken)
		}
	}
	

	return finalTokens
}