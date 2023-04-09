package main

import (
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/cpmech/gosl/mpi"
	"github.com/dereklstinson/gocudnn"
	"github.com/mumax/3/plum"
	"golang.org/x/crypto/aes"
)

func main() {
	mpi.Init()
	defer mpi.Finalize()

	size := mpi.Size()
	rank := mpi.Rank()

	// calculate start and end indices for this rank
	rowsPerRank := 40000000 / size
	chunkSize := rowsPerRank / size
	start := rank * chunkSize
	end := (rank + 1) * chunkSize

	// read excel file
	rows, err := readExcel("data.xlsx")
	if err != nil {
		panic(err)
	}

	// apply filter using go-plum
	plum.Init()
	defer plum.Close()

	filter := plum.NewFilter("myfilter.cu")
	defer filter.Close()

	for i := start; i < end; i++ {
		row := rows[i]

		// apply filter to each row using go-cudnn and CUDA
		image := imgio.Load("image.png")
		image = transform.Resize(image, 256, 256, transform.Linear)
		tensor := gocudnn.Image2Tensor(image, gocudnn.DefaultTensorFormat, true)
		defer tensor.Free()

		err := filter.Apply(tensor, tensor)
		if err != nil {
			panic(err)
		}

		// apply AES encryption using golang.org/x/crypto
		key := []byte("my-secret-key-123")
		block, err := aes.NewCipher(key)
		if err != nil {
			panic(err)
		}

		encrypted := make([]byte, len(row))
		block.Encrypt(encrypted, row)

		// send result to master process
		mpi.Send(encrypted, 0, 0)
	}

	// master process receives results from all workers
	if rank == 0 {
		results := make([][]byte, 0)
		for i := 0; i < size; i++ {
			result := make([]byte, chunkSize)
			mpi.Recv(result, i, 0)
			results = append(results, result)
		}

		// write results to output excel file
		writeExcel("output.xlsx", results)
	}
}

func readExcel(filename string) ([][]byte, error) {
	// read excel file
	// TODO: implement
}

func writeExcel(filename string, data [][]byte) error {
	// write excel file
	// TODO: implement
}
