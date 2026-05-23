package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
)

const MODEL_PATH = "SmolLM2-135M"

type TensorInfo struct {
	Dtype   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	DataOff []uint64 `json:"data_offsets"`
}

type Tensor struct {
    Shape []int64
    Data  []float32
}


func bytesToFloat32(b []byte) []float32 {
	// BF16 uses 2 bytes per value
	// TO DO need to learn more about this BF16 thingy
    size := len(b) / 2
    floats := make([]float32, size)

    for i := 0; i < size; i++ {
        bits16 := binary.LittleEndian.Uint16(b[i*2 : (i+1)*2])
		
		bits32 := uint32(bits16) << 16

        floats[i] = math.Float32frombits(bits32)
    }
    return floats
}

func ReadTensors() map[string]*Tensor{
	tensorMap := make(map[string]*Tensor)

	f, err := os.Open(MODEL_PATH+"/model.safetensors")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Reading first 8 bytes of safetensors file gives us N
	buf := make([]byte, 8)
	_, err = f.Read(buf)
	if err != nil {
		panic(err)
	}
	// Byte to int
	headerLength := binary.LittleEndian.Uint64(buf)

	// Reading next N bytes
	headerBuf := make([]byte, headerLength)

	// Starting after first 8 bytes
	_, err = f.ReadAt(headerBuf, 8)
	if err != nil {
		panic(err)
	}

	// parsing as JSON
	var header map[string]json.RawMessage
	err = json.Unmarshal(headerBuf, &header)
	if err != nil {
		panic(err)
	}


	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		
		var info TensorInfo
		json.Unmarshal(raw, &info)

		dataStart := 8 + headerLength + info.DataOff[0]
		dataEnd := 8 + headerLength + info.DataOff[1]
		dataLen := dataEnd - dataStart

		tensorBytes := make([]byte, dataLen)
		f.ReadAt(tensorBytes, int64(dataStart))

		tensorMap[name] = &Tensor{
			Data: bytesToFloat32(tensorBytes),
			Shape: info.Shape,
		}
	}

	return tensorMap
}